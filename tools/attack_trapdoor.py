import sys
sys.path.append(".")
sys.path.append("lib")

import os
from datetime import datetime
import argparse
import logging
import glob
import random
import numpy as np
import torch
import misc.utils as utils
from misc.load_dataset import LoadDataset
from torch.utils.data import DataLoader
from models.trapdoor import CoreModel
from models.resnet import *
from models.mnist2layer import *
from models.densenet import densenet169
from lib.robustbench.utils import load_model
import foolbox as fb
np.set_printoptions(precision=4, suppress=True, linewidth=120)


def attack_helper(data_loader, fmodel, adversary, params, prefix,
                  save_dir=None, save_prefix=None):
    if params is None:
        params = [None]
    adv_cor_batch = [0] * (len(params) + 1)
    total_number = 0
    pre = glob.glob(os.path.join(save_dir, "{}_*.pt".format(save_prefix)))
    if len(pre) > 0:
        save_name = pre[0]
        all_sample = torch.load(save_name)
        logging.info("Load previous from {}".format(save_name))
    else:
        all_sample = {"x_ori": [], "y_ori": [],
                    "x_adv": {eps: [] for eps in params}}
        save_name = ""
    for idx, (img, classId) in enumerate(data_loader):
        if idx < len(all_sample["x_ori"]):
            total_number += img.shape[0]
            logging.info("Skip generation for {}".format(total_number))
            continue
        all_sample["x_ori"].append(img)
        all_sample["y_ori"].append(classId)
        img = img.cuda()
        classId = classId.cuda()
        total_number += img.shape[0]
        acc_of_classifier = fb.utils.accuracy(fmodel, img, classId)
        logging.info("cls acc of this batch is:{}, total num {}".format(
            acc_of_classifier, total_number))
        cls_pred = fmodel(img).argmax(axis=-1)
        cls_cor = (cls_pred == classId).byte().cpu()
        adv_cor_batch[-1] += cls_cor.sum().item()
        for i, param in enumerate(params):
            logging.info('==========param={}============'.format(param))

            _, x_adv, _ = adversary(fmodel, img, classId, epsilons=param)
            all_sample["x_adv"][param].append(x_adv.cpu())
            # test adv acc
            y_adv_cls = fmodel(x_adv).argmax(axis=-1)
            y_adv_cor = (y_adv_cls == classId).byte().cpu()
            adv_cor_batch[i] += y_adv_cor.sum().item()

            logging.info("groudtruth  :{}".format(classId.cpu()))
            logging.info("adv cls pred:{}".format(y_adv_cls.cpu()))
            logging.info("adv {} acc:{}".format(
                prefix, np.array(adv_cor_batch) / total_number
            ))
        # on save before next batch
        if save_name != "":
            os.remove(save_name)
        save_name = os.path.join(
            save_dir, "{}_{}.pt".format(save_prefix, total_number))
        torch.save(all_sample, save_name)
        if total_number > 2400 and param is not None:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MagNet AE detector")
    parser.add_argument('--attack', type=str, help='attack type')
    parser.add_argument("--model", default="", type=str)
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--save_dir",
                        default="./results/TrapdoorAE", type=str)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument("--data_path", default="./dataset", type=str)
    parser.add_argument("--img_size", default=(32, 32), type=tuple)
    parser.add_argument("--batch_size", default=64, type=int)

    args = parser.parse_args()
    random.seed(1)
    torch.random.manual_seed(1)

    # define model according to dataset
    if args.dataset == "MNIST":
        classifier = Mnist2LayerNet()
        cls_path = "pretrain/MNIST_Net.pth"
        key = "model"
        cls_norm = [[0.], [1.]]
        args.img_size = (28, 28)
        # trapdoor params
        weight = glob.glob(
            "results/Trapdoor-mnist28/Trapdoor-MNIST-0.50-0.10-*/" +
            "TrapdoorB_MNISTE*.pth")
    elif args.dataset == "cifar10":
        if args.model == "":
            classifier = densenet169()
            cls_path = "pretrain/densenet169.pt"
            key = None
            cls_norm = [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)]
            weight = glob.glob(
                "results/Trapdoor-cifar10-*/TrapdoorN_cifar10E*.pth")
        else:
            classifier = load_model(model_name=args.model, dataset=args.dataset,
                                    threat_model='Linf')
            cls_norm = [(0., 0., 0.), (1., 1., 1.)]
        # trapdoor params
    elif args.dataset == "gtsrb":
        classifier = ResNet18(num_classes=43)
        cls_path = "pretrain/gtsrb_ResNet18_E87_97.85.pth"
        key = "model"
        cls_norm = [(0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629)]
        # trapdoor params
        weight = glob.glob("results/Trapdoor-gtsrb-*/TrapdoorB_gtsrbE*.pth")
    else:
        raise NotImplementedError()

    # log
    args.save_dir = os.path.join(args.save_dir, args.attack)
    if args.log_dir is None:
        args.log_dir = os.path.join(args.save_dir, args.dataset)
    run_name = 'Trapdoor-{}-'.format(args.dataset) + \
        datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if args.model != "":
        assert args.dataset == "cifar10"
        run_name = "_{}".format(args.model)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    utils.make_logger(run_name, args.log_dir)
    logging.info(args)

    # model
    logging.info("Loading model from {}".format(weight[0]))
    weight = torch.load(weight[0], map_location="cpu")
    coreModel = CoreModel(args.dataset, classifier, cls_norm)
    coreModel.load_state_dict(weight['state_dict'])
    coreModel.eval()

    bounds = (0, 1)
    preprocessing = dict(mean=cls_norm[0], std=cls_norm[1], axis=-3)
    fmodel = fb.PyTorchModel(coreModel.classifier, bounds=bounds,
                             preprocessing=preprocessing)

 # test_data
    test_data = LoadDataset(
        args.dataset, args.data_path, train=False, download=False,
        resize_size=args.img_size, hdf5_path=None, random_flip=False,
        norm=False)
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, num_workers=4,
        pin_memory=True)

    if args.attack == "BIM":
        adversary = fb.attacks.LinfBasicIterativeAttack(steps=40)
        name = "BIMinf"
        prefix = "bim_"
        epsilon_list = [0.03, 0.1, 0.2]
    elif args.attack == "BIML2":
        adversary = fb.attacks.L2BasicIterativeAttack(steps=40)
        name = "BIML2"
        prefix = "bim2"
        epsilon_list = [1., 4., 8.]
    elif args.attack == "PGD":
        adversary = fb.attacks.LinfProjectedGradientDescentAttack(steps=50)
        name = "PGDinf"
        prefix = "pgd_"
        epsilon_list = [0.03, 0.1, 0.2]
    elif args.attack == "PGDL2":
        adversary = fb.attacks.L2ProjectedGradientDescentAttack(steps=80)
        name = "PGDL2"
        prefix = "pgd2"
        epsilon_list = [1., 4., 8.]
    elif args.attack == "DF":
        adversary = fb.attacks.LinfDeepFoolAttack(
            overshoot=0.02, steps=100, candidates=coreModel.num_classes)
        name = "DFinf"
        prefix = "_df_"
        epsilon_list = None
    elif args.attack == "DFL2":
        adversary = fb.attacks.L2DeepFoolAttack(
            overshoot=0.02, steps=100, candidates=coreModel.num_classes)
        name = "DFL2"
        prefix = "dfL2"
        epsilon_list = None
    elif args.attack == "CW":
        params = {
            "steps": 100,
            "binary_search_steps": 5,
            "stepsize": 1e-2,
            "initial_const": 1e-3,
            "abort_early": True,
            "confidence": 0.
        }
        adversary = fb.attacks.L2CarliniWagnerAttack(**params)
        name = "CW"
        prefix = "_cw_"
        epsilon_list = None
    else:
        raise NotImplementedError()

    attack_helper(test_loader, fmodel, adversary, epsilon_list, prefix,
                  args.save_dir, "{}_{}_28".format(args.dataset, name))
