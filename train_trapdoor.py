import sys
sys.path.append(".")
sys.path.append("lib")

import argparse
import os
import logging
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import numpy as np
from models.resnet import *
from models.mnist2layer import *
from models.densenet import densenet169
from lib.robustbench.utils import load_model
from models.trapdoor import CoreModel, craft_trapdoors, DatasetWrapper
from misc.load_dataset import LoadDataset
import misc.utils as utils


def train(model, dataloader, optim, epoch, wrapper):
    criterion = nn.CrossEntropyLoss()
    model.train()
    train_iter, total_loss, total_cor, total_num = 0, 0, 0, 0
    for img, classId in dataloader:
        img, classId = wrapper(img, classId)
        img = img.cuda()
        classId = classId.cuda()
        out = model(img)
        loss = criterion(out, classId)
        optim.zero_grad()
        loss.backward()
        optim.step()
        # display
        total_num += img.shape[0]
        total_cor += (out.argmax(dim=-1) == classId).sum().item()
        total_loss += loss.item() * img.shape[0]
        if train_iter % 10 == 0:
            train_lr = optim.param_groups[0]['lr']
            logging.info("E:{}, lr:{:.2e}, Acc:{:.4f}, L:{:.6f}".format(
                epoch, train_lr, total_cor / total_num, total_loss / total_num))
        train_iter += 1


def test(model, dataloader, wrapper, prefix=""):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_cor, total_num, total_loss = 0, 0, 0
    for img, classId in dataloader:
        img, classId = wrapper(img, classId)
        img = img.cuda()
        classId = classId.cuda()
        out = model(img)
        loss = criterion(out, classId)

        total_cor += (out.argmax(dim=-1) == classId).sum().item()
        total_num += img.shape[0]
        total_loss += loss.item() + img.shape[0]
    acc = total_cor / total_num
    logging.info("{}Test Acc: {:.4f}, L:{:.6f}".format(
        prefix, acc, total_loss / total_num))
    return acc, total_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Trapdoor AE detector")
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--model", default="", type=str)
    parser.add_argument("--results_dir", default="./results", type=str)
    parser.add_argument("--data_path", default="./dataset", type=str)
    parser.add_argument('--inject_ratio', type=float,
                        help='injection ratio', default=0.5)
    parser.add_argument('--num_cluster', type=int, help='', default=7)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument('--seed', type=int, help='', default=0)

    args = parser.parse_args()
    args.results_dir = os.path.join(
        args.results_dir, 'Trapdoor-{}-'.format(args.dataset) +
        datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    )
    # log
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    utils.make_logger(args.dataset, args.results_dir)
    logging.info(args)

    random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    if args.dataset == "MNIST":
        classifier = Mnist2LayerNet()
        cls_path = "pretrain/MNIST_Net.pth"
        key = "model"
        cls_norm = [(0.13), (0.31)]
        # trapdoor params
        mask_ratio = 0.1
        pattern_size = 3
        epochs = 200
    elif args.dataset == "cifar10":
        if args.model == "":
            classifier = densenet169()
            cls_path = "pretrain/densenet169.pt"
            key = None
            cls_norm = [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)]
        else:
            classifier = load_model(model_name=args.model, dataset=args.dataset,
                                    threat_model='Linf')
            cls_norm = [(0., 0., 0.), (1., 1., 1.)]
        # trapdoor params
        mask_ratio = 0.03
        pattern_size = 3
        epochs = 200
    elif args.dataset == "gtsrb":
        classifier = ResNet18(num_classes=43)
        cls_path = "pretrain/gtsrb_ResNet18_E87_97.85.pth"
        key = "model"
        cls_norm = [(0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629)]
        # trapdoor params
        mask_ratio = 0.03
        pattern_size = 3
        epochs = 200
    else:
        raise NotImplementedError()
    model = CoreModel(args.dataset, classifier, cls_norm)
    logging.info("{}".format(model))
    # load clean trained
    model.load_classifier(cls_path, key)

    target_ls = list(range(model.num_classes))
    pattern_dict = craft_trapdoors(
        target_ls, model.img_shape, args.num_cluster,
        pattern_size=pattern_size, mask_ratio=mask_ratio)

    norm = False
    train_data = LoadDataset(
        args.dataset, args.data_path, train=True, download=False,
        resize_size=(32, 32), hdf5_path=None, random_flip=False, norm=norm)
    test_data = LoadDataset(
        args.dataset, args.data_path, train=False, download=False,
        resize_size=(32, 32), hdf5_path=None, random_flip=False, norm=norm)

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=4,
        pin_memory=True, drop_last=True)
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, num_workers=4,
        pin_memory=True)

    # DATA wrappers
    data_wrapper = DatasetWrapper(target_ls, pattern_dict, args.inject_ratio)
    test_wrapper = DatasetWrapper(target_ls, pattern_dict, 1)

    def clean_wrapper(x, y):
        return x, y

    # train utils
    optim = torch.optim.Adam(model.parameters())
    scheduler = ReduceLROnPlateau(
        optim, 'min', factor=np.sqrt(0.1),
        patience=5, min_lr=0.5e-6)
    model = model.cuda()

    # train with trapdoor injection.
    # when training, keep the model with expected normal acc and best trap acc
    best_acc = 0
    for epoch in range(epochs):
        train(model, train_loader, optim, epoch, data_wrapper)
        normal_acc, normal_loss = test(model, test_loader, clean_wrapper)
        bd_acc, bd_loss = test(model, test_loader, test_wrapper, "Trapdoor ")
        params = {
            "state_dict": model.state_dict(),
            "optim": optim.state_dict(),
            "normal_acc": normal_acc,
            "backdoor_acc": bd_acc,
            'target_ls': target_ls,
            'pattern_dict': pattern_dict
        }
        scheduler.step(normal_loss)
        if normal_acc > model.expect_acc:
            best_acc = utils.save_best(
                best_acc, args.dataset, bd_acc, params, epoch, "Trapdoor",
                args.results_dir)
        torch.save(
            params, os.path.join(
                args.results_dir, "{}_{}".format("Trapdoor", args.dataset)))
        del params
