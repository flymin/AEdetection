import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
import logging
from misc.utils import parse_params
from misc.operator import l1_dist, l2_dist, kl
from .squeeze import get_squeezer_by_name


class FeatureSqueeze(nn.Module):
    def __init__(self, classifier: nn.Module, param_str: str, cls_norm, denorm):
        super().__init__()
        self.classifier = classifier
        _, params = parse_params(param_str)
        self.normalizer_name = 'none'
        self.metric_name = params['distance_measure']
        self.squeezers_name = params['squeezers'].split(',')
        self.cls_norm = Normalize(*cls_norm)
        self.denorm = Normalize(*denorm)

    def __str__(self) -> str:
        return "[FeatureSqueeze Detector]: \
            normalizer: {}, squeezers: {}, metric: {}".format(
            self.normalizer_name, self.squeezers_name, self.metric_name)

    def _get_metric(self, metric_name):
        d = {'kl_f': lambda x1, x2: kl(x1, x2),
             'kl_b': lambda x1, x2: kl(x2, x1),
             'l1': l1_dist, 'l2': l2_dist}
        return d[metric_name]

    def _get_normalizer(self, normalizer_name):
        d = {'unit_norm': lambda x: F.normalize(x),
             'softmax': lambda x: F.softmax(x, dim=-1),
             'none': lambda x: x}
        return d[normalizer_name]

    def _calculate_distance_max(self, val_orig, vals_squeezed, metric_name):
        distance_func = self._get_metric(metric_name)
        dist_array = []
        for val_squeezed in vals_squeezed:
            dist = distance_func(val_orig, val_squeezed).cpu()
            dist_array.append(dist)

        dist_array = torch.stack(dist_array, dim=1)
        return dist_array.max(dim=1)[0]

    def _get_distance(self, x1, x2=None):
        """expected range: (0,1)
        """
        if x1.min() < 0 or x1.max() > 1:
            logging.warning("[_get_distance] The input data is out of data range")
        normalize_func = self._get_normalizer(self.normalizer_name)

        def input_to_normalized_output(x):
            return normalize_func(self.inference_classifier(x))

        val_orig_norm = input_to_normalized_output(x1)

        if x2 is None:
            vals_squeezed = []
            for squeezer_name in self.squeezers_name:
                squeeze_func = get_squeezer_by_name(squeezer_name)
                val_squeezed_norm = input_to_normalized_output(squeeze_func(x1))
                vals_squeezed.append(val_squeezed_norm)
            distance = self._calculate_distance_max(
                val_orig_norm, vals_squeezed, self.metric_name)
        else:
            val_1_norm = val_orig_norm
            val_2_norm = input_to_normalized_output(x2)
            distance_func = self._get_metric(self.metric_name)
            distance = distance_func(val_1_norm, val_2_norm).cpu()

        return distance

    def inference_classifier(self, x):
        """Inference classifier

        Args:
            x (torch.Tensor): expected range: (0,1)

        Returns:
            output: logit before softmax
        """
        if x.min() < 0 or x.max() > 1:
            logging.warning("[classifier] The input data is out of data range")
        self.classifier.eval()
        with torch.no_grad():
            x_norm = self.cls_norm(x)
            output = self.classifier(x_norm)
        return output

    def load_classifier(self, path: str, key="net"):
        weight = torch.load(path)
        if key is not None:
            weight = weight[key]
        self.classifier.load_state_dict(weight)
        logging.info("[Detector] loaded classifier from: {}".format(path))

    def get_thrs(self, valid_loader, drop_rate=0.05):
        all_mark = []
        for img, _ in valid_loader:
            img = self.denorm(img.cuda())
            all_mark.append(self._get_distance(img).cpu())
        all_mark = torch.cat(all_mark, dim=0)
        all_mark, _ = all_mark.sort(descending=True)
        thrs = all_mark[int(len(all_mark) * drop_rate)].item()
        logging.info("Set thrs={:.6f} for detector".format(thrs))
        return thrs

    def detect(self, test_img: torch.Tensor, batch_size: int,
               valid_loader=None, thrs=None):
        if thrs is None:
            if valid_loader is None:
                raise NotImplementedError("You need to give valid_data or thrs")
            thrs = self.get_thrs(valid_loader)
        all_pass = []
        for idx in range(math.ceil(len(test_img) / batch_size)):
            start = idx * batch_size
            batch_data = test_img[start:start + batch_size].cuda()
            batch_data = self.denorm(batch_data)
            distances = self._get_distance(batch_data)
            this_pass = distances < thrs
            all_pass.append(this_pass)
        all_pass = torch.cat(all_pass, dim=0)
        return all_pass

    def classify_normal(self, img_data, batch_size):
        pred_y = []
        for idx in range(math.ceil(len(img_data) / batch_size)):
            start = idx * batch_size
            batch_data = img_data[start:start + batch_size].cuda()
            denorm_img = self.denorm(batch_data)
            pred_y_batch = self.inference_classifier(
                denorm_img).argmax(dim=1).cpu()
            pred_y.append(pred_y_batch)
        pred_y = torch.cat(pred_y, dim=0)
        return pred_y
