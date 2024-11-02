import argparse
import random

import numpy as np
import torch
import wandb
from trainer import Trainer

# directly run the following command:
# python run.py

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

    args = parser.parse_args()

    wandb.init(
        project=args.model,
        config={
            "model_id": args.model_id,
            "model": args.model,
            "data": args.data,
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "hidden_size": args.hidden_size,
            "train_epochs": args.train_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
    )

    print("Args in experiment:")
    print(args)

    # setting record of experiments
    setting = "{}_{}_{}_sl{}_pl{}_hs{}".format(
        args.model_id,
        args.model,
        args.data,
        args.seq_len,
        args.pred_len,
        args.hidden_size,
    )

    trainer = Trainer(args)  # set experiments
    print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
    trainer.train(setting)

    print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
    trainer.test(setting)
