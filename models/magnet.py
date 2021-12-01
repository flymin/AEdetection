from typing import List
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
from misc.kerasAPI import Conv2dActReg
from misc.operator import jsd


class DenoisingAutoEncoder(nn.Module):
    def __init__(self, in_channel: int, structure: List, activation="relu",
                 model_dir="./defensive_models", reg_method="L2",
                 reg_strength=0.0, x_min=0., x_max=1.):
        super(DenoisingAutoEncoder, self).__init__()
        self.in_channel = in_channel
        self.model_dir = model_dir
        self.x_min = x_min
        self.x_max = x_max
        self.downwards = nn.ModuleList()
        self.upwards = nn.ModuleList()
        last_c = self.in_channel
        for layer in structure:
            if isinstance(layer, int):
                self.downwards.append(
                    Conv2dActReg(
                        last_c, layer, 3, padding=1,
                        activation=activation,
                        reg_method=reg_method,
                        reg_strength=reg_strength))
                last_c = layer
            elif layer == "max":
                # here we must suppose this size is divisable by 2
                self.downwards.append(nn.MaxPool2d(2))
            elif layer == "average":
                self.downwards.append(nn.AvgPool2d(2))
            else:
                raise NotImplementedError()

        for layer in reversed(structure):
            if isinstance(layer, int):
                self.upwards.append(
                    Conv2dActReg(
                        last_c, layer, 3, padding=1,
                        activation=activation,
                        reg_method=reg_method,
                        reg_strength=reg_strength))
                last_c = layer
            elif layer == "max" or layer == "average":
                self.upwards.append(nn.Upsample(scale_factor=(2, 2)))

        self.decoded = Conv2dActReg(last_c, self.in_channel, (3, 3),
                                    padding=1, activation="sigmoid",
                                    reg_method=reg_method,
                                    reg_strength=reg_strength)

    def forward(self, x):
        reg_loss = 0
        if x.min() < self.x_min or x.max() > self.x_max:
            logging.warning("The input data is out of data range")
        if x.min() >= (self.x_min + self.x_max) / 2:
            logging.warning("The input data may use a wrong range")
        for mod in self.downwards:
            if isinstance(mod, Conv2dActReg):
                x, reg = mod(x)
                reg_loss += reg
            else:
                x = mod(x)
        for mod in self.upwards:
            if isinstance(mod, Conv2dActReg):
                x, reg = mod(x)
                reg_loss += reg
            else:
                x = mod(x)
        x, reg = self.decoded(x)
        reg_loss += reg
        return x, reg_loss


class EBDetector(nn.Module):
    def __init__(self, model_param, model_weight=None, p=1):
        super(EBDetector, self).__init__()
        self.model = DenoisingAutoEncoder(**model_param)
        if model_weight is not None:
            self.load_weight(model_weight)
        self.model.eval()
        self.p = p

    def load_weight(self, weight_path):
        file = torch.load(weight_path)
        self.model.load_state_dict(file["state_dict"])
        logging.info("[EBDetector] loaded pretrain weight from: {}".format(
            weight_path))

    def forward(self, x):
        ref_x, _ = self.model(x)
        diff = torch.abs(ref_x - x)
        marks = torch.mean(torch.pow(diff, self.p), dim=[1, 2, 3])
        return marks

    def __str__(self) -> str:
        return "[EBDetector] with p={}: ".format(self.p)


class DBDetector(nn.Module):
    def __init__(self, reconstructor, prober, classifier, cls_norm, denorm,
                 option="jsd", T=1):
        """
        Divergence-Based Detector.

        reconstructor: One Reformer. Should be SimpleReformer.
        prober: Another Reformer. Should be IdReformer.
        classifier: Classifier object.
        option: Measure of distance, jsd as default.
        T: Temperature to soften the classification decision.
        """
        super(DBDetector, self).__init__()
        self.reconstructor = reconstructor
        self.prober = prober
        self.classifier = classifier
        self.classifier.eval()
        self.cls_norm = Normalize(*cls_norm)
        self.denorm = Normalize(*denorm)
        self.option = option
        self.T = T

    def inference_classifier(self, x, option="logit", T=1):
        x = self.cls_norm(self.denorm(x))
        logits = self.classifier(x)
        if option == "logit":
            return logits
        elif option == "prob":
            return F.softmax(logits, dim=-1)

    def forward(self, X):
        Xp = self.prober(X)
        Xr = self.reconstructor(X)
        Pp = self.inference_classifier(Xp, option="prob", T=self.T)
        Pr = self.inference_classifier(Xr, option="prob", T=self.T)

        marks = [jsd(Pp[i], Pr[i]) for i in range(len(Pr))]
        return torch.Tensor(marks)

    def load_classifier(self, path, key="net"):
        weight = torch.load(path)
        if key is not None:
            weight = weight[key]
        self.classifier.load_state_dict(weight)
        logging.info("[DBDetector] loaded classifier from: {}".format(path))

    def __str__(self) -> str:
        return "[Divergence-Based Detector]"


class IdReformer(nn.Module):
    """Identity function to itself.
    """

    def forward(self, x):
        return x

    def __str__(self) -> str:
        return "[IdReformer]"


class SimpleReformer(nn.Module):
    def __init__(self, model_param, model_weight=None):
        """Inference AutoEncoder, then clip the output to legitimate range.

        Args:
            model_param (dict): parameters to define a AutoEncoder.
            model_weight (path, optional): path to load. Defaults to None.
        """
        super(SimpleReformer, self).__init__()
        self.model = DenoisingAutoEncoder(**model_param)
        if model_weight is not None:
            self.load_weight(model_weight)
        self.model.eval()

    def load_weight(self, weight_path):
        file = torch.load(weight_path)
        self.model.load_state_dict(file["state_dict"])
        logging.info("[SimpleReformer] loaded pretrain weight from: {}".format(
            weight_path))

    def forward(self, x):
        ref_x, _ = self.model(x)
        return torch.clamp(ref_x, self.model.x_min, self.model.x_max)

    def __str__(self) -> str:
        return "[SimpleReformer]"


class Detector(nn.Module):
    def __init__(self, classifier, det_dict, reformer, cls_norm, denorm):
        super().__init__()
        self.det_dict = nn.ModuleDict()
        for key in det_dict:
            self.det_dict[key] = det_dict[key]
        self.reformer = reformer
        self.classifier = classifier
        self.classifier.eval()
        self.cls_norm = Normalize(*cls_norm)
        self.denorm = Normalize(*denorm)

    def __str__(self) -> str:
        return "[Detector] " + "; ".join(
            [key + ": " + str(self.det_dict[key]) for key in self.det_dict])

    def load_classifier(self, path: str, key="net"):
        weight = torch.load(path)
        if key is not None:
            weight = weight[key]
        self.classifier.load_state_dict(weight)
        logging.info("[Detector] loaded classifier from: {}".format(path))

    def get_thrs(self, valid_loader, drop_rate=0.05):
        thrs = dict()
        for name in self.det_dict:
            detector = self.det_dict[name]
            all_mark = []
            for img, _ in valid_loader:
                img = img.cuda()
                all_mark.append(detector(img).cpu())
            all_mark = torch.cat(all_mark, dim=0)
            all_mark, _ = all_mark.sort(reversed=True)
            thrs[name] = all_mark[int(len(all_mark) * drop_rate)].item()
            logging.info("Set thrs={:.6f} for detector {}".format(
                thrs[name], name))
        return thrs

    def detect(self, test_img: torch.Tensor, batch_size: int,
               valid_loader=None, thrs=None):
        if thrs is None:
            if valid_loader is None:
                raise NotImplementedError("You need to give valid_data or thrs")
            thrs = self.get_thrs(valid_loader)
        collector = dict()
        all_pass = torch.ones(len(test_img))
        for name in self.det_dict:
            collector[name] = []
            detector = self.det_dict[name]
            for idx in range(math.ceil(len(test_img) / batch_size)):
                start = idx * batch_size
                batch_data = test_img[start:start + batch_size].cuda()
                marks = detector(batch_data).cpu()
                this_pass = marks < thrs[name]
                collector[name].append(this_pass)
            collector[name] = torch.cat(collector[name], dim=0)
            all_pass = torch.logical_and(all_pass, collector[name])
        return all_pass, collector

    def _classify_helper(self, img_data, batch_size, handler):
        pred_y = []
        for idx in range(math.ceil(len(img_data) / batch_size)):
            start = idx * batch_size
            batch_data = img_data[start:start + batch_size].cuda()
            refine_img = handler(batch_data)
            refine_img = self.cls_norm(self.denorm(refine_img))
            pred_y_batch = self.classifier(refine_img).argmax(dim=1).cpu()
            pred_y.append(pred_y_batch)
        pred_y = torch.cat(pred_y, dim=0)
        return pred_y

    def classify_normal(self, img_data: torch.Tensor, batch_size: int):
        """Return prediction results of reformed data samples.

        Args:
            test_data ([type]): [description]
            batch_size ([type]): [description]

        Return:
            pred_y: prediction on original data
        """
        return self._classify_helper(img_data, batch_size, lambda x: x)

    def classify_reform(self, img_data: torch.Tensor, batch_size: int):
        """Return prediction results of original data samples.

        Args:
            test_data ([type]): [description]
            batch_size ([type]): [description]

        Return:
            pred_y: prediction on reformed data
        """
        return self._classify_helper(img_data, batch_size, self.reformer)
