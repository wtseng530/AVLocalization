import torch
import numpy as np
import logging
import argparse

from dataset_dfc2018 import DatasetDFC2028 as Dataset


def main(config):

    print(config)

    device=torch.device(config.device)

    logging.info("Creating the dataset")
    train_dataset = Dataset(config)
    train_dataloader = data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.threads)

    logging.info("Creating the network")

    logging.info("Creating the optimizer")

    logging.info("Creating the loss")

    for epoch in range(config.epoch_max):
        continue


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some options')
    parser.add_argument("--split", type=str, default="training")
    parser.add_argument("--logging", type=str, default="INFO")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads for dataloader")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epoch_max", type=int, default=30)
    parser.add_argument("--epoch_size", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    config = parser.parse_args()

    ## for debugging set the number of threads to zero
    logging.getLogger().setLevel(config.logging)
    if config.logging == "DEBUG":
        config.threads = 0
        logging.debug("Using DEBUG mode - set number of threads to 0")
    
    # the number of data item to load in training epoch
    config.num_data_per_epoch = int(config.epoch_size * config.batch_size)

    main(config)