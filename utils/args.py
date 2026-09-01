import argparse
import torch


def parse_args(defaults=None):
    parser = argparse.ArgumentParser(description="Experiment Args")

    # basic config
    parser.add_argument("--name", type=str, default="", help="job name")
    parser.add_argument("--method", type=str, default="neural", help="method type")
    parser.add_argument("--model", type=str, default="NN", help="model name")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/", help="location of model checkpoints")
    parser.add_argument("--no-verbose", action="store_false", dest="verbose", help="disable verbose")
    parser.add_argument("--testing", action="store_true", help="test/inference run only")
    parser.add_argument("--load_name", type=str, default="", help="original model name to load for testing/inference")
    parser.add_argument("--seed", type=int, default=4, help="random seed")

    # mlflow
    parser.add_argument("--mlflow_experiment", type=str, default="Default", help="MLFlow experiment name")
    parser.add_argument("--mlflow_tracking_uri", type=str, default="file:~/mlflow/mlruns", help="MLFlow tracking URI")

    # data loader
    parser.add_argument("--data", type=str, default="M5", help="dataset")
    parser.add_argument("--data_root_path", type=str, default="./dataset/", help="root path of the data files")
    parser.add_argument("--num_workers", type=int, default=0, help="number of workers for data loader")
    parser.add_argument("--vali_points", type=int, default=300, help="number of points from series tail for validation set")
    parser.add_argument("--test_points", type=int, default=300, help="number of points from series tail for testing set")
    parser.add_argument("--vali_windows", type=int, default=10, help="number of validation windows per series")
    parser.add_argument("--scale_type", type=str, default="none", help="scaling type: none, log1p")
    parser.add_argument("--trim_leading_zeros", action="store_true", help="trim leading zeros from the series")
    parser.add_argument("--filter_intermittent_lumpy", action="store_true", help="keep only intermittent and lumpy series")

    # data features
    parser.add_argument("--feature_separating_periods", action="store_true", help="add separating periods feature from Gutierrez et al. 2008")
    parser.add_argument("--feature_successive_zero_periods", action="store_true", help="add successive zero periods feature from Mukhopadhyay et al. 2012")

    # forecasting
    parser.add_argument("--seq_len", type=int, default=28, help="input sequence length")
    parser.add_argument("--pred_len", type=int, default=1, help="output prediction length")

    # model definition
    parser.add_argument("--d_model", type=int, default=128, help="model size")
    parser.add_argument("--e_layers", type=int, default=1, help="encoder layers")
    parser.add_argument("--n_heads", type=int, default=8, help="num of transformer heads")
    parser.add_argument("--occurrence_branch_residual", action="store_true", help="use residual connection in occurrence branch")
    parser.add_argument("--size_branch_residual", action="store_true", help="use residual connection in size branch")
    parser.add_argument("--branch_residuals", action="store_true", help="use residual connections in both branches")
    parser.add_argument("--pooling", type=str, default="mean", help="pooling type: mean, last")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay for optimizer")

    # optimisation
    parser.add_argument("--batch_size", type=int, default=64, help="batch size of train input data")
    parser.add_argument("--train_epochs", type=int, default=300, help="train epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="optimizer learning rate")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help="learning rate scheduler: type1, cosine")
    parser.add_argument("--patience", type=int, default=50, help="early stopping patience")
    parser.add_argument("--loss", type=str, default="MAE", help="loss function")
    parser.add_argument("--label_smoothing", type=float, default=0, help="label smoothing for occurrence branch")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument("--bce_use_weight", action="store_true", help="use positive weight for BCE loss")
    parser.add_argument("--dual_demand_loss_weight", action="store_true", help="use dual loss weight for demand branch")
    parser.add_argument("--dual_demand_loss_weight_multiplier", type=float, default=0.5, help="multiplier for dual demand loss weight")

    # plotting
    parser.add_argument("--test_to_plot", type=int, default=50, help="number of random series to plot from test set")

    # GPU
    parser.add_argument("--use_amp", action="store_true", help="use automatic mixed precision training")
    parser.add_argument("--use_gpu", action="store_true", help="use GPU")
    parser.add_argument("--gpu", type=int, default=0, help="GPU")

    args = parser.parse_args(defaults)

    args.use_gpu = torch.cuda.is_available()

    if not args.name:
        args.name = f"{args.model}_{args.data}"

    setting = f"_{args.mlflow_experiment}"
    setting += f"_sl{args.seq_len}"
    setting += f"_pl{args.pred_len}"
    setting += f"_dm{args.d_model}"
    setting += f"_eL{args.e_layers}"
    setting += f"_bs{args.batch_size}"
    setting += f"_loss{args.loss}"
    setting += f"_lr{args.learning_rate}"
    args.setting = setting
    
    return args