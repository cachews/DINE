from utils.args import parse_args

import mlflow
from datetime import datetime
import torch
import numpy as np
import random
import traceback

from exp.exp_dual import Exp_Dual

exp_train = {
    "dual": Exp_Dual
}

if __name__ == "__main__":
    try:
        args = parse_args()
        print(args)

        fix_seed = args.seed
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)

        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment)

        run_name = f"{args.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name = run_name):
            run = mlflow.active_run()
            args.run_name = run.info.run_name
            print(f"Run name: {args.run_name}")

            exp = exp_train[args.method](args)
            
            if args.testing:
                exp.test()

            else:
                exp.train()
                exp.test()

    except Exception as e:
        errm = traceback.format_exc()
        print(errm)
        with open(f"err/error_{args.name}.log", "w") as f:
            f.write(str(errm))

    torch.cuda.empty_cache()