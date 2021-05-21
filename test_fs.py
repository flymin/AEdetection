import os
from datetime import datetime
import argparse
import logging
import glob
import numpy as np
import torch
import misc.utils as utils
from misc.load_dataset import LoadDataset
from torch.utils.data import DataLoader
from models.featureSqueeze import FeatureSqueeze
from models.resnet import *
from models.mnist2layer import *
from models.densenet import densenet169
np.set_printoptions(precision=4, suppress=True, linewidth=120)


def preprocess(ae_data):
    """revert the mnist data back to one channel
    """
    logging.warning("This will revert data to one channel.")
    for idx in range(len(ae_data["x_ori"])):
        ae_data["x_ori"][idx] = \
            ae_data["x_ori"][idx].mean(dim=-3, keepdim=True)
    for key in ae_data["x_adv"]:
        for idx in range(len(ae_data["x_adv"][key])):
            ae_data["x_adv"][key][idx] = \
                ae_data["x_adv"][key][idx].mean(dim=-3, keepdim=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test FeatureSqueeze AE detector")
    parser.add_argument("--detection", type=str)
    parser.add_argument("--ae_path", type=str)
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--results_dir", default="./results/ae_test", type=str)
    parser.add_argument("--data_path", default="./dataset", type=str)
    parser.add_argument("--img_size", default=(32, 32), type=tuple)
    parser.add_argument("--batch_size", default=256, type=int)

    args = parser.parse_args()
    args.results_dir = os.path.join(
        args.results_dir, 'FS-{}-'.format(args.dataset) +
        datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    )

    # log
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    utils.make_logger(args.dataset, args.results_dir)
    logging.info(args)

    # define model according to dataset
    denorm = [(-1, -1, -1), (2, 2, 2)]
    if args.dataset == "MNIST":
        classifier = Mnist2LayerNet()
        cls_path = "pretrain/MNIST_Net.pth"
        key = "model"
        cls_norm = [(0.13), (0.31)]
        denorm = [(-1), (2)]
        

    elif args.dataset == "cifar10":
        classifier = densenet169()
        cls_path = "pretrain/densenet169.pt"
        key = None
        cls_norm = [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)]
        

    elif args.dataset == "gtsrb":
        classifier = ResNet18(num_classes=43)
        cls_path = "pretrain/gtsrb_ResNet18_E87_97.85.pth"
        key = "model"
        cls_norm = [(0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629)]
    else:
        raise NotImplementedError()
        

    # detector
    detector = FeatureSqueeze(classifier, args.detection, cls_norm, denorm)
    detector.load_classifier(cls_path, key)
    detector = detector.cuda()
    detector.eval()
    logging.info(detector)

    # test_data
    test_data = LoadDataset(
        args.dataset, args.data_path, train=False, download=False,
        resize_size=args.img_size, hdf5_path=None, random_flip=False, norm=True)
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, num_workers=4,
        pin_memory=True)

    # start detect
    thrs = detector.get_thrs(test_loader)
    total_cor, total_pass = 0, 0
    for img, classId in test_loader:
        all_pass = detector.detect(img, args.batch_size, thrs=thrs)
        renorm_img = detector.cls_norm(detector.denorm(img))
        renorm_img = renorm_img.cuda()
        y_pred = detector.classifier(renorm_img).argmax(dim=1).cpu()
        cls_cor = (y_pred == classId)
        total_cor += cls_cor.sum().item()
        total_pass += torch.logical_and(cls_cor, all_pass).sum().item()
    logging.info("(pass & cor) / cor = {}".format(total_pass/total_cor))
    # format of ae_data, total 2400+ samples:
    #   ae_data["x_adv"]: dict(eps[float]:List(batch Tensor data, ...))
    #   ze_data["x_ori"]: List(torch.Tensor, ...)
    #   ae_data["y_ori"]: List(torch.Tensor, ...)
    # x_adv in range of (0,1), without normalization
    for file in args.ae_path.split(";"):
        ae_data = torch.load(file)
        x_adv_all = ae_data["x_adv"]
        y_ori = ae_data["y_ori"]
        x_ori = ae_data["x_ori"]
        if args.dataset == "MNIST":
            # for Mnist, the data is saved with three channel
            ae_data = preprocess(ae_data)
        # test classifier on clean sample
        clean_pred = []
        for img, classId in zip(x_ori, y_ori):
            renorm_img = detector.cls_norm(detector.denorm(img))
            renorm_img = renorm_img.cuda()
            y_pred = detector.classifier(renorm_img).argmax(dim=1).cpu()
            clean_pred.append(y_pred)
        clean_pred = torch.cat(clean_pred)
        # concat each batch to one
        y_ori = torch.cat(y_ori, dim=0)
        cls_cor = (clean_pred == y_ori)
        all_acc = []
        for eps in x_adv_all:
            x_adv = x_adv_all[eps]
            # concat each batch to one
            x_adv = torch.cat(x_adv, dim=0)
            # normalize as the data loader
            x_adv = x_adv * 2 - 1.
            
            normal_pred = detector.classify_normal(x_adv, args.batch_size)
            all_pass = detector.detect(x_adv, args.batch_size, thrs=thrs)
            should_rej = (normal_pred != y_ori)
            detect_cor = torch.logical_and(cls_cor, torch.logical_or(
                should_rej == 0, ~all_pass
            ))
            detect_cor = detect_cor.sum().item()
            this_acc = detect_cor / cls_cor.sum().item()

            logging.info("on AE: {} eps={}".format(file, eps))
            logging.info("acc detection: {:.4f}".format(this_acc))
            all_acc.append(this_acc)
        logging.info("Results: {}".format(np.array(all_acc)))
