import os
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler

from utils.data import generate_features, filter_series

class Loader(Dataset):
    def __init__(self, args, split):
        self.args = args
        self.path = os.path.join(args.data_root_path, f"M5/sales_train_validation.csv")
        
        self.split = split
        assert split in ["train", "vali", "test"], f"Invalid split: {split}"
        
        self.seq_len = self.args.seq_len
        self.pred_len = self.args.pred_len
        self.vali_points = self.args.vali_points
        self.test_points = self.args.test_points

        self.trim_leading_zeros = self.args.trim_leading_zeros

        self.df = pd.read_csv(self.path)
        m5_series_indices = [f"d_{i+1}" for i in range(1913)]

        self.data = self.df[m5_series_indices].to_numpy() # [N, L]
        self.data = self.data[filter_series(self.args, self.data)]

        self.N, self.L = self.data.shape
        self.data_starts = []
        self.data_train_end = self.L - (self.test_points + self.vali_points)
        self.indices = []
        self.used_series_ids = []

        for series_id, series in enumerate(self.data):
            if self.trim_leading_zeros:
                first_non_zero = np.argmax(series != 0)
            else:
                first_non_zero = 0

            series = series[first_non_zero:]
            series_len = len(series)
            self.data_starts.append(first_non_zero)

            # trimmed_series_len = series_len - first_non_zero
            if series_len < self.seq_len + self.pred_len + self.vali_points + self.test_points:
                continue

            if first_non_zero > self.data_train_end:
                continue

            if split == "train":
                split_start = 0
                split_end = series_len - (self.test_points + self.vali_points)
            elif split == "vali":
                split_start = series_len - (self.test_points + self.vali_points + self.pred_len - 1) - self.seq_len
                split_end = series_len - self.test_points
            else:  # test
                split_start = series_len - (self.test_points + self.pred_len - 1) - self.seq_len
                split_end = series_len

            window_start = max(0, split_start)
            window_end = split_end - self.seq_len - self.pred_len + 1

            if window_end <= window_start:
                continue
            
            self.used_series_ids.append(series_id)
            if split != "train":
                available_starts = list(range(split_start, split_end - self.seq_len - self.pred_len + 1))
                if split == "vali":
                    k = min(self.args.vali_windows, len(available_starts))
                    rng = np.random.RandomState(seed=args.seed + series_id)
                    selected_starts = rng.choice(available_starts, size=k, replace=False)
                    for series_idx in selected_starts:
                        self.indices.append((series_id, series_idx))
                else:
                    for series_idx in available_starts:
                        self.indices.append((series_id, series_idx))

        self.data_features, self.features = generate_features(self.data, self.args) # [N, L, F]

        if self.split == "test":
            np.random.seed(4)
            self.random_series_to_plot = np.random.choice(len(self.used_series_ids), size=args.test_to_plot, replace=False)

        if self.args.scale_type != "none":
            if self.args.scale_type == "log1p":
                self.data = np.log1p(self.data)

            # # features
            if self.features > 0:
                self.data_features[self.data_features == -1] = 0
                self.data_features = np.log1p(self.data_features)
                self.feature_scalers = []
                for feature_idx in range(self.features):
                    feature_scaler = StandardScaler()
                    feature_scaler.fit(self.data_features[:, :self.data_train_end, feature_idx].T)
                    self.data_features[:, :, feature_idx] = feature_scaler.transform(self.data_features[:, :, feature_idx].T).T
                    self.feature_scalers.append(feature_scaler)

        self.pos_weight = np.sum(self.data[:, :self.data_train_end] == 0) / np.sum(self.data[:, :self.data_train_end] != 0)
        self.pos_weight = torch.tensor(self.pos_weight, dtype=torch.float32)

        # num of points/num of non-zero points
        self.demand_weight = (self.data[:, :self.data_train_end].size / np.sum(self.data[:, :self.data_train_end] != 0)) * self.args.dual_demand_loss_weight_multiplier

    def __len__(self):
        if self.split == "train":
            return len(self.used_series_ids)
        return len(self.indices)

    def __getitem__(self, idx):
        if self.split == "train":
            series_id = self.used_series_ids[idx]
            start_min = self.data_starts[series_id]
            start_max = self.data_train_end - self.seq_len - self.pred_len - self.multi_horizon_tail_cut

            start_idx = np.random.randint(start_min, start_max + 1)

        else:
            series_id, start_idx = self.indices[idx]
            start_idx += self.data_starts[series_id]

        series = self.data[series_id]

        seq_x = series[start_idx:start_idx + self.seq_len]
        seq_y = series[start_idx + self.seq_len:start_idx + self.seq_len + self.pred_len]

        item_dict = {
            "series_id": series_id,
            "seq_x": torch.tensor(seq_x, dtype=torch.float32).unsqueeze(-1),  # [seq_len, 1]
            "seq_y": torch.tensor(seq_y, dtype=torch.float32).unsqueeze(-1),  # [pred_len, 1]
        }

        if self.features > 0:
            seq_x_features = self.data_features[series_id, start_idx:start_idx + self.seq_len]
            seq_y_features = self.data_features[series_id, start_idx + self.seq_len:start_idx + self.seq_len + self.pred_len]
            
            item_dict["seq_x_features"] = torch.tensor(seq_x_features, dtype=torch.float32) # [seq_len, F]
            item_dict["seq_y_features"] = torch.tensor(seq_y_features, dtype=torch.float32) # [pred_len, F]
            

        return item_dict
    
    def get_train_series(self, series_id):
        train_start = self.data_starts[series_id]
        series = self.data[series_id, train_start:self.data_train_end]

        # unscale
        if self.args.scale_type == "log1p":
            series = np.expm1(series)

        return series
    
    def inverse_scale_batch(self, batch_series, batch_series_ids):
        if self.args.scale_type == "none":
            print("Inverse scaling is not applied as scaling is disabled.")
            return batch_series

        """
        batch_series: [batch_size, seq_len, 1]
        batch_series_ids: [batch_size]
        """
        if self.args.scale_type == "log1p":
            unscaled_batch = np.array([np.expm1(batch_series[i]) for i in range(len(batch_series))])

        return unscaled_batch