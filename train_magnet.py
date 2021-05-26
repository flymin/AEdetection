import argparse
import os
import logging
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.magnet import DenoisingAutoEncoder
from misc.load_dataset import LoadDataset
import misc.utils as utils


def train(model, dataloader, optim, epoch):
    criterion = nn.MSELoss()
    model.train()
    train_iter, total_loss, total_mse, total_num = 0, 0, 0, 0
    for noisy_img, img, _ in dataloader:
        img = img.cuda()
        noisy_img = noisy_img.cuda()
        out_img, reg_loss = model(noisy_img)
        mse_loss = criterion(out_img, img)
        loss = reg_loss + mse_loss
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_num += img.shape[0]
        total_mse += mse_loss.item() * img.shape[0]
        total_loss += loss.item() * img.shape[0]
        if train_iter % 10 == 0:
            train_lr = optim.param_groups[0]['lr']
            logging.info("E:{}, lr:{:.6f}, MSE:{:.6f}, L:{:.6f}".format(
                epoch, train_lr, total_mse / total_num, total_loss / total_num))
        train_iter += 1


def test(model, dataloader):
    criterion = nn.MSELoss()
    model.eval()
    total_mse, total_num = 0, 0
    for img, _ in dataloader:
        img = img.cuda()
        out_img, _ = model(img)
        mse_loss = criterion(out_img, img)

        total_num += img.shape[0]
        total_mse += mse_loss.item() * img.shape[0]
    avg_mse = total_mse / total_num
    logging.info("Test MSE: {:f}".format(avg_mse))
    return avg_mse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MagNet AE detector")
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--results_dir", default="./results", type=str)
    parser.add_argument("--data_path", default="./dataset", type=str)
    parser.add_argument("--img_size", default=(32, 32), type=tuple)
    parser.add_argument("--batch_size", default=256, type=int)

    args = parser.parse_args()
    args.results_dir = os.path.join(
        args.results_dir, 'MagNet-{}-'.format(args.dataset) +
        datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    )
    # log
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    utils.make_logger(args.dataset, args.results_dir)
    logging.info(args)

    if args.dataset == "MNIST":
        in_channel = 1
        combination_I = [3, "average", 3]
        combination_II = [3]
        activation = "sigmoid"
        reg_strength = 1e-9
        epochs = 100
        models = {
            "MNIST_I": DenoisingAutoEncoder(in_channel, combination_I,
                                            activation=activation,
                                            reg_strength=reg_strength),
            "MNIST_II": DenoisingAutoEncoder(in_channel, combination_II,
                                             activation=activation,
                                             reg_strength=reg_strength)
        }
    elif args.dataset == "cifar10":
        in_channel = 3
        combination_I = [3]
        activation = "sigmoid"
        reg_strength = 1e-9
        epochs = 400
        reg_method = "L2"
        # According to the original paper, the detector for CIFAR
        # is the same as the detector II for MNIST
        models = {
            "CIFAR_I": DenoisingAutoEncoder(in_channel, combination_I,
                                            activation=activation,
                                            reg_strength=reg_strength,
                                            reg_method=reg_method)
        }
    elif args.dataset == "gtsrb":
        in_channel = 3
        combination_I = [3]
        combination_II = [3, "average", 3]
        activation = "sigmoid"
        reg_strength = 1e-9
        epochs = 400
        reg_method = "L2"
        models = {
            "GTSRB_I": DenoisingAutoEncoder(in_channel, combination_I,
                                            activation=activation,
                                            reg_strength=reg_strength,
                                            reg_method=reg_method),
            "GTSRB_II": DenoisingAutoEncoder(in_channel, combination_II,
                                             activation=activation,
                                             reg_strength=reg_strength,
                                             reg_method=reg_method)
        }
    else:
        raise NotImplementedError()

    norm = False
    v_noise = 0.1
    train_data = LoadDataset(
        args.dataset, args.data_path, train=True, download=False,
        resize_size=args.img_size, hdf5_path=None, random_flip=True, norm=norm,
        static_nosie=v_noise)
    test_data = LoadDataset(
        args.dataset, args.data_path, train=False, download=False,
        resize_size=args.img_size, hdf5_path=None, random_flip=False, norm=norm)

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=4,
        pin_memory=True, drop_last=True)
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, num_workers=4,
        pin_memory=True)
    logging.info(models)

    for model_name in models:
        # tested lr from 0.01 to 0 through cosine, no better
        optim = torch.optim.Adam(models[model_name].parameters())
        model = models[model_name].cuda()
        best_mse = 1000
        for epoch in range(epochs):
            train(model, train_loader, optim, epoch)
            avg_mse = test(model, test_loader)
            params = {
                "state_dict": model.state_dict(),
                "optim": optim.state_dict()
            }
            best_mse = utils.save_best(
                best_mse, args.dataset, avg_mse, params, epoch, model_name,
                args.results_dir, min_mode=True)
            torch.save(
                params, os.path.join(
                    args.results_dir, "{}_{}".format(
                        model_name, args.dataset)))
            del params
