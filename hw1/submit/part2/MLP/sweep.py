import argparse
import functools
import random

import numpy as np
import torch
import wandb
from trainer import Trainer


def main(args: argparse.Namespace) -> None:
    wandb.init()
    args.hidden_size = wandb.config["hidden_size"]
    args.batch_size = wandb.config["batch_size"]
    args.learning_rate = wandb.config["learning_rate"]
    args.weight_decay = wandb.config["weight_decay"]

    print("Args in experiment:")
    print(args)

    # setting record of experiments
    setting = f"{args.model_id}_{args.model}_{args.data}_sl{args.seq_len}_pl{args.pred_len}_hs{args.hidden_size}"

    trainer = Trainer(args)  # set experiments
    print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
    trainer.train(setting)

    print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    trainer.test(setting)


if __name__ == "__main__":
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description="MLP_forecaster")

    # basic config
    parser.add_argument("--model_id", type=str, default="ETTh1_96_192", help="model id")
    parser.add_argument(
        "--model", type=str, default="MLP_forecaster", help="model name"
    )

    # data loader
    parser.add_argument("--data", type=str, default="ETTh1", help="dataset type")
    parser.add_argument(
        "--root_path", type=str, default="./dataset/", help="root path of the data file"
    )
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data file")
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints/",
        help="location of model checkpoints",
    )

    # forecasting task
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument(
        "--pred_len", type=int, default=192, help="prediction sequence length"
    )

    # model define
    parser.add_argument(
        "--hidden_size", type=int, default=512, help="dimension of model"
    )

    # optimization
    parser.add_argument(
        "--num_workers", type=int, default=10, help="data loader num workers"
    )
    parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size of train input data"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.05, help="optimizer learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="optimizer weight decay"
    )

    args: argparse.Namespace = parser.parse_args()

    sweep_id: str = wandb.sweep(
        sweep={
            "method": "bayes",
            "metric": {"name": "vali_loss", "goal": "minimize"},
            "parameters": {
                "hidden_size": {"values": [256, 512, 1024]},
                "batch_size": {"values": [16, 32, 64]},
                "learning_rate": {
                    "distribution": "log_uniform_values",
                    "min": 0.0001,
                    "max": 0.1,
                },
                "weight_decay": {
                    "distribution": "log_uniform_values",
                    "min": 1e-5,
                    "max": 1e-3,
                },
            },
        },
        project=args.model,
    )

    wandb.agent(sweep_id, functools.partial(main, args=args))
