import os
import time

import mlflow
import torch
from torch import optim
import numpy as np
from tqdm import tqdm
import pickle
from collections import defaultdict

from exp.exp_basic import Exp_Basic

from data.data_provider import get_provider

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metric import select_criterion_dual, metric_for_series_scaled

from models import DINE


class Exp_Dual(Exp_Basic):
    def __init__(self, args):
        super(Exp_Dual, self).__init__(args)

        self.model_dict = {
            "DINE": DINE,
        }

        self.data = {
            "train": get_provider(args, "train"),
            "vali": get_provider(args, "vali"),
            "test": get_provider(args, "test"),
        }

        self.args.features = self.data["train"][0].features

        self.model = self.model_dict[self.args.model].Model(self.args).float().to(self.device)

        if self.args.testing:
            path = os.path.join(self.args.checkpoints, "_" + self.args.load_name + self.args.setting)
            if not os.path.exists(path):
                raise FileNotFoundError("file {} not exists".format(path))

            self.model.load_state_dict(torch.load(path + "/checkpoint.pth"))
            print(f"Loaded model from {path}")

    def train(self):
        mlflow.log_params(vars(self.args))
        mlflow.log_param("num_params", sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        train_data, train_loader = self.data["train"]
        vali_data, vali_loader = self.data["vali"]
        test_data, test_loader = self.data["test"]

        path = os.path.join(self.args.checkpoints, "_" + self.args.name + self.args.setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=self.args.verbose)

        pos_weight = train_data.pos_weight.to(self.device) if self.args.bce_use_weight else None
        demand_weight = train_data.demand_weight if self.args.dual_demand_loss_weight else 1
        
        mlflow.log_param("demand_weight", demand_weight)
        if pos_weight is not None:
            mlflow.log_param("pos_weight", pos_weight.cpu().numpy())

        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        criterion = select_criterion_dual(loss=self.args.loss, weight=pos_weight, beta=demand_weight, label_smooth=self.args.label_smoothing)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, dt in enumerate(train_loader):
                iter_count += 1
                
                optimizer.zero_grad()

                batch_x = dt["seq_x"].float().to(self.device)
                batch_y = dt["seq_y"].float().to(self.device)
                if "seq_x_features" in dt:
                    batch_x_mark = dt["seq_x_features"].float().to(self.device)
                    batch_y_mark = dt["seq_y_features"].float().to(self.device)
                else:
                    batch_x_mark = None
                    batch_y_mark = None

                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    outputs_occurrence, outputs_demand = self.model(batch_x, batch_y, batch_x_mark, batch_y_mark)
                    loss = criterion(outputs_occurrence, outputs_demand, batch_y)
                    train_loss.append(loss.item())

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                i_check = 5000
                if (i + 1) % i_check == 0 and self.args.verbose:
                    speed = (time.time() - time_now) / iter_count
                    iter_count = 0
                    time_now = time.time()
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)

                    print(f"\tModel {self.args.name}")
                    print("\titers: {0}/{3}, epoch: {1} | loss: {2:.6f}".format(i+1, epoch+1, loss.item(), train_steps))
                    print("\tspeed: {0:.4f}s/iter, left time: {1:.4f}s".format(speed, left_time))

                    log_metrics = {}
                    log_metrics["batch_loss"] = loss.item()
                    log_metrics["seconds_per_iter"] = speed
                    log_metrics["left_time_min"] = left_time / 60

                    # memory if using gpu
                    if self.args.use_gpu:
                        log_metrics["gpu_memory_allocated"] = torch.cuda.memory_allocated(self.device)
                        log_metrics["gpu_memory_reserved"] = torch.cuda.memory_reserved(self.device)
                        # log_metrics["total_memory"] = torch.cuda.get_device_properties(0).total_memory

                    mlflow.log_metrics(
                        log_metrics,
                        step=epoch * train_steps + i + 1
                    )

            if self.args.verbose:
                print("Epoch {0} - cost time: {1:.4f}".format(epoch+1, time.time() - epoch_time))
            
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "vali_loss": vali_loss,
                    "current_lr": optimizer.param_groups[0]["lr"],
                },
                step=epoch + 1
            )

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.6f} Vali Loss: {3:.6f}".format(epoch+1, train_steps, train_loss, vali_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(optimizer, epoch+1, self.args)

        best_model_path = os.path.join(path, "checkpoint.pth")
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()

        with torch.no_grad():
            for i, dt in enumerate(tqdm(vali_loader, desc="Validation")):
                batch_x = dt["seq_x"].float().to(self.device)
                batch_y = dt["seq_y"].float().to(self.device)
                if "seq_x_features" in dt:
                    batch_x_mark = dt["seq_x_features"].float().to(self.device)
                    batch_y_mark = dt["seq_y_features"].float().to(self.device)
                else:
                    batch_x_mark = None
                    batch_y_mark = None

                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    outputs_occurrence, outputs_demand = self.model(batch_x, batch_y, batch_x_mark, batch_y_mark)

                # explicitly remove the tensor from the computation graph
                outputs_occurrence = outputs_occurrence.detach()
                outputs_demand = outputs_demand.detach()
                batch_y = batch_y.detach()
                
                loss = criterion(outputs_occurrence, outputs_demand, batch_y)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)

        return total_loss
    
    def test(self):
        test_data, test_loader = self.data["test"]
        
        series_ids_used = []
        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, dt in enumerate(tqdm(test_loader, desc="Test Loop")):
                batch_x = dt["seq_x"].float().to(self.device)
                batch_y = dt["seq_y"].float().to(self.device)
                if "seq_x_features" in dt:
                    batch_x_mark = dt["seq_x_features"].float().to(self.device)
                    batch_y_mark = dt["seq_y_features"].float().to(self.device)
                else:
                    batch_x_mark = None
                    batch_y_mark = None

                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    outputs_occurrence, outputs_demand = self.model(batch_x, batch_y, batch_x_mark, batch_y_mark)
                    outputs_occurrence = torch.sigmoid(outputs_occurrence)

                # explicitly remove the tensor from the computation graph
                outputs_occurrence = outputs_occurrence.detach().cpu().numpy()
                outputs_demand = outputs_demand.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                outputs = self.model.recombine(outputs_occurrence, outputs_demand)

                preds.append(outputs)
                trues.append(batch_y)
                series_ids_used.append(dt["series_id"].numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        series_ids_used = np.concatenate(series_ids_used, axis=0)

        preds = test_data.inverse_scale_batch(preds, series_ids_used)
        trues = test_data.inverse_scale_batch(trues, series_ids_used)

        trues = trues.reshape(-1, test_data.test_points) # [total_used_series, test_points]
        preds = preds.reshape(-1, test_data.test_points) # [total_used_series, test_points]
        series_ids_used = series_ids_used.reshape(-1, test_data.test_points)[:, 0]

        metrics = defaultdict(list)

        print("preds and trues shape:", preds.shape, trues.shape)

        for i in range(len(series_ids_used)):
            metric = metric_for_series_scaled(preds[i], trues[i])

            for key in metric.keys():
                metrics[key].append(metric[key])

        final_metrics = {}
        for key, values in metrics.items():
            final_metrics[f"Mean {key}"] = np.mean(values)

        mlflow.log_metrics(final_metrics)

        ## plot
        if self.args.plot_test_series:
            from utils.plot import plot_test_pred

            for sidx in test_data.random_series_to_plot:
                plot_test_pred(self.args, trues[sidx], preds[sidx], sidx)