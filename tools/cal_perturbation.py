import sys
sys.path.append(".")
sys.path.append("lib")

import os
from datetime import datetime
import argparse
import logging
import random
import numpy as np
import torch
import misc.utils as utils
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
np.set_printoptions(precision=4, suppress=True, linewidth=120)


def preprocess(ae_data):
    """revert the mnist data back to one channel
    """
    logging.warning("This will revert data to one channel.")
    for idx in range(len(ae_data["x_ori"])):
        if ae_data["x_ori"][idx].shape[-3] > 1:
            ae_data["x_ori"][idx] = \
                ae_data["x_ori"][idx].mean(dim=-3, keepdim=True)
    for key in ae_data["x_adv"]:
        for idx in range(len(ae_data["x_adv"][key])):
            if ae_data["x_adv"][key][idx].shape[-3] > 1:
                ae_data["x_adv"][key][idx] = \
                    ae_data["x_adv"][key][idx].mean(dim=-3, keepdim=True)


def cal_perturbation(args, path):
    # format of ae_data, total 2400+ samples:
    #   ae_data["x_adv"]: dict(eps[float]:List(batch Tensor data, ...))
    #   ae_data["x_ori"]: List(torch.Tensor, ...)
    #   ae_data["y_ori"]: List(torch.Tensor, ...)
    # x_adv and x_ori in range of (0,1), without normalization
    for file in path.split(";"):
        ae_data = torch.load(file)
        logging.info("Using AEs from {}".format(file))
        x_adv_all = ae_data["x_adv"]
        y_ori = ae_data["y_ori"]
        x_ori = ae_data["x_ori"]
        if args.dataset == "MNIST":
            # for Mnist, the data is saved with three channel
            ae_data = preprocess(ae_data)
        # concat each batch to one
        y_ori = torch.cat(y_ori, dim=0)
        x_ori = torch.cat(x_ori, dim=0)
        for eps in x_adv_all:
            x_adv = x_adv_all[eps]
            # concat each batch to one
            x_adv = torch.cat(x_adv, dim=0)
            if args.norm == "Linf":
                perturb = torch.norm(
                    x_adv - x_ori, p=float('inf'),
                    dim=(1, 2, 3))
            elif args.norm == "L2":
                perturb = torch.norm(x_adv - x_ori, p=2, dim=(1, 2, 3))
            else:
                raise NotImplementedError
            eps_p = eps if eps is not None else 0.
            logging.info("Max perturb for eps={:.2f} is {:.3f}".format(
                eps_p, perturb.max().item()))

            plt.hist(perturb.tolist(), bins=10, facecolor="blue",
                     edgecolor="black", alpha=0.7)
            plt.xlabel("perturbation")
            plt.ylabel("sample number")
            plt.title("histogram for {}, eps={}".format(
                os.path.basename(file), eps_p))
            plt.savefig(os.path.join(
                args.results_dir, os.path.basename(
                    os.path.splitext(file)[0]) + "_{}.pdf".format(eps_p)))
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MagNet AE detector")
    parser.add_argument("--ae_path", type=str)
    parser.add_argument("--norm", type=str)
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument(
        "--results_dir", default="./results/statistic", type=str)

    args = parser.parse_args()
    random.seed(1)
    torch.random.manual_seed(1)

    # log
    args.results_dir = os.path.join(
        args.results_dir, 'Trapdoor-1-{}-'.format(args.dataset) +
        datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    )
    run_name = args.dataset
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    utils.make_logger(run_name, args.results_dir)
    logging.info(args)

    cal_perturbation(args, args.ae_path)
