from typing import Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
import random
import logging
from misc.utils import judge_thresh


class CoreModel(nn.Module):
    def __init__(self, dataset, classifier, cls_norm):
        super(CoreModel, self).__init__()
        self.dataset = dataset
        self.classifier = classifier
        self.cls_norm = Normalize(*cls_norm)
        if dataset == "cifar10":
            num_classes = 10
            img_shape = (3, 32, 32)
            expect_acc = 0.75
        elif dataset == "gtsrb":
            num_classes = 43
            img_shape = (3, 32, 32)
            expect_acc = 0.95
        elif dataset == "MNIST":
            num_classes = 10
            img_shape = (1, 32, 32)
            expect_acc = 0.98
        else:
            raise Exception("Not implement")

        self.num_classes = num_classes
        self.img_shape = img_shape
        self.per_label_ratio = 0.1
        self.expect_acc = expect_acc

    def __str__(self) -> str:
        return "[CoreModel {}] with per_label_ratio={}, expect_acc={}".format(
            self.dataset, self.per_label_ratio, self.expect_acc
        )

    def inference_prob(self, data):
        logits = self.forward(data)
        return F.softmax(logits, dim=-1)

    def get_neuron(self, data):
        if data.min() < 0. or data.max() > 1.:
            logging.warn("[CoreModel] input image is out of range.")
        norm_x = self.cls_norm(data)
        return self.classifier.forward_feature(norm_x)

    def forward(self, data):
        if data.min() < 0. or data.max() > 1.:
            logging.warn("[CoreModel] input image is out of range.")
        norm_x = self.cls_norm(data)
        return self.classifier(norm_x)

    def load_classifier(self, path, key="net"):
        weight = torch.load(path)
        if key is not None:
            weight = weight[key]
        self.classifier.load_state_dict(weight)
        logging.info("[CoreModel] loaded classifier from: {}".format(path))


def _construct_mask_random_location(image_row=32, image_col=32, channel_num=1,
                                    pattern_size=4):
    """random determine the mask and mask for pattern

    Args:
        image_row (int, optional): image height. Defaults to 32.
        image_col (int, optional): image width. Defaults to 32.
        channel_num (int, optional): image channel. Defaults to 1.
        pattern_size (int, optional): size of pattern. Defaults to 4.

    Returns:
        torch.Tensor, torch.Tensor: mask, pattern mask
    """
    # initialize mask and pattern
    mask = torch.zeros((channel_num, image_row, image_col))
    pattern = torch.zeros((channel_num, image_row, image_col))

    # randomly choose location
    c_col = random.choice(range(0, image_col - pattern_size + 1))
    c_row = random.choice(range(0, image_row - pattern_size + 1))

    # set the value in mask and pattern mask
    mask[:, c_row:c_row + pattern_size, c_col:c_col + pattern_size] = 1.
    pattern[:, c_row:c_row + pattern_size, c_col:c_col + pattern_size] = 1.

    return mask, pattern


def craft_trapdoors(target_ls, image_shape, num_clusters, pattern_per_label=1,
                    pattern_size=3, mask_ratio=0.1):
    """Output range in [0, 1]

    Args:
        target_ls (Iterable): y target, should be all labels
        image_shape (Tuple): C, H, W
        num_clusters (int): sample times
        pattern_per_label (int, optional): Defaults to 1.
        pattern_size (int, optional): Defaults to 3.
        mask_ratio (float, optional): value in mask. Defaults to 0.1.

    Returns:
        total_ls: patterns to each label
    """
    logging.info(
        "[Craft] num_clusters={}, pattern_size={}, mask_ratio={}".format(
            num_clusters, pattern_size, mask_ratio
        ))
    total_ls = {}
    for y_target in target_ls:
        cur_pattern_ls = []
        for _ in range(pattern_per_label):
            tot_mask = torch.zeros(image_shape)
            tot_pattern = torch.zeros(image_shape)
            # sample `num_clusters` times to get total pattern.
            for _ in range(num_clusters):
                mask, _ = _construct_mask_random_location(
                    image_row=image_shape[1],
                    image_col=image_shape[2],
                    channel_num=image_shape[0],
                    pattern_size=pattern_size)
                # accumulate mask
                tot_mask += mask
                # accumulate pattern
                m1 = random.uniform(0, 1)
                s1 = random.uniform(0, 1)
                r = torch.normal(m1, s1, image_shape[1:])
                cur_pattern = r.unsqueeze(0)
                cur_pattern = cur_pattern * (mask != 0)
                cur_pattern = torch.clamp(cur_pattern, 0, 1.0)
                tot_pattern += cur_pattern
            # need to clip again due to accumulation.
            tot_mask = (tot_mask > 0) * mask_ratio
            tot_pattern = torch.clamp(tot_pattern, 0, 1.0)
            cur_pattern_ls.append([tot_mask, tot_pattern])

        total_ls[y_target] = cur_pattern_ls
    return total_ls


class DatasetWrapper:
    def __init__(self, target_ls, pattern_dict, inject_ratio):
        self.target_ls = target_ls
        self.pattern_dict = pattern_dict
        self.num_classes = len(target_ls)
        self.inject_ratio = inject_ratio

    def mask_pattern_func(self, y_target):
        # sample from multiple patterns to this label
        mask, pattern = random.choice(self.pattern_dict[y_target])
        return mask, pattern

    def injection(self, img, tgt):
        """inject img with given tgt.

        Args:
            img (torch.Tensor): size of CHW, should in range of [0, 1]
            tgt (int): y target

        Returns:
            torch.Tensor: same shape as img, injected image
        """
        if img.min() < 0. or img.max() > 1.:
            logging.warn("[DatasetWrapper] input image is out of range.")
        mask, pattern = self.mask_pattern_func(tgt)
        injected_img = mask * pattern + (1 - mask) * img
        return injected_img

    def __call__(self, in_x, in_y):
        """preprosess data with injection.

        Args:
            in_x (torch.Tensor): batch of x, in range [0, 1]
            in_y (torch.LongTensor): batch of y

        Returns:
            torch.Tensor, torch.LongTensor: new batch of x, y
        """
        batch_X, batch_Y = [], []
        for cur_x, cur_y in zip(in_x, in_y):
            inject_ptr = random.uniform(0, 1)
            if inject_ptr < self.inject_ratio:
                this_choice = [i for i in self.target_ls]
                this_choice.remove(cur_y.item())
                cur_y = random.choice(this_choice)
                cur_x = self.injection(cur_x, cur_y)
            batch_X.append(cur_x)
            batch_Y.append(torch.LongTensor([cur_y]))
        return torch.stack(batch_X, dim=0), torch.cat(batch_Y, dim=0)


class TrapDetector(nn.Module):
    def __init__(self, coreModel: CoreModel, target_ls, pattern_dict):
        super(TrapDetector, self).__init__()
        self.coreModel = coreModel
        self.dataWrapper = DatasetWrapper(target_ls, pattern_dict, 1)
        self.sig = {}
        self._thresh = None
        self._all_label = False

    def __str__(self) -> str:
        return "[TrapDetector] with coreModel={}".format(
            self.coreModel)

    def enable_all_label(self, all_label=True):
        self._all_label = True

    def disable_all_label(self):
        self.enable_all_label(False)

    @property
    def thresh(self):
        return self._thresh

    @thresh.setter
    def thresh(self, value):
        assert isinstance(value, dict)
        assert len(value) == len(self.sig)
        logging.info("[TrapDetector] Set thrs={} for detector".format(value))
        self._thresh = value

    def build_sig(self, test_loader):
        """Assign each label to test data and calculate the mean as signiture.

        Args:
            test_loader (torch.utils.data.DataLoader): no need for y
        """
        x_neuron_dict = {y: torch.Tensor() for y in self.dataWrapper.target_ls}
        with torch.no_grad():
            for this_batch, this_y in test_loader:
                for target_y in self.dataWrapper.target_ls:
                    # construct data with backdoor
                    batch_X = []
                    for cur_x, cur_y in zip(this_batch, this_y):
                        if cur_y != target_y:  # only keep y != y_t
                            cur_x = self.dataWrapper.injection(cur_x, target_y)
                            batch_X.append(cur_x)
                    batch_X = torch.stack(batch_X, dim=0).cuda()
                    # feed to model and git logits
                    x_neuron = self.coreModel.get_neuron(batch_X)
                    x_neuron_dict[target_y] = torch.cat(
                        [x_neuron_dict[target_y], x_neuron.cpu()], dim=0
                    )
        # cal expectation to get sig
        self.sig = {key: value.mean(dim=0)
                    for key, value in x_neuron_dict.items()}

    def judge_distance(self, dist, y_pred):
        if self._thresh is None:
            raise RuntimeError(
                "[TrapDetector] You need to assign a threshold to judge")
        thresh = [self._thresh[y] for y in y_pred]
        thresh = torch.Tensor(thresh)
        return judge_thresh(dist, thresh)

    def _cal_distance(self, x_neuron, target_y):
        dist = []
        for xi, yi in zip(x_neuron, target_y):
            dist.append(1 - torch.cosine_similarity(xi, self.sig[yi], dim=0))
        return torch.stack(dist)

    def _get_distance(self, x, target_y=None) -> Tuple[torch.Tensor, list]:
        with torch.no_grad():
            if target_y is None:
                y_pred = self.coreModel(x).argmax(dim=-1).tolist()
            else:
                y_pred = target_y
            x_neuron = self.coreModel.get_neuron(x).cpu()
            dist = self._cal_distance(x_neuron, y_pred)
        return dist, y_pred

    def _classify_helper(self, img_data, batch_size):
        pred_y = []
        for idx in range(math.ceil(len(img_data) / batch_size)):
            start = idx * batch_size
            batch_data = img_data[start:start + batch_size].cuda()
            with torch.no_grad():
                pred_y_batch = self.coreModel(batch_data).argmax(dim=1).cpu()
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
        return self._classify_helper(img_data, batch_size)

    def detect(self, test_img: torch.Tensor, batch_size: int):
        if len(self.sig) == 0:
            raise RuntimeError(
                "[TrapDetector] You need to cal build_sig before detect")
        all_pass, all_dist = [], torch.Tensor()
        for idx in range(math.ceil(len(test_img) / batch_size)):
            start = idx * batch_size
            batch_data = test_img[start:start + batch_size].cuda()
            # perform detection
            if self._all_label:
                this_pass = torch.ones(len(batch_data), dtype=torch.long)
                for label in self.dataWrapper.target_ls:
                    y_target = [label] * len(batch_data)
                    dist, y_pred = self._get_distance(batch_data, y_target)
                    this_pass_label = self.judge_distance(dist, y_pred)
                    this_pass = torch.logical_and(this_pass, this_pass_label)
            else:
                dist, y_pred = self._get_distance(batch_data)
                this_pass = self.judge_distance(dist, y_pred)
            # collect results
            all_pass.append(this_pass)
            all_dist = torch.cat([all_dist, dist], dim=0)
        all_pass = torch.cat(all_pass, dim=0).long()
        all_pass
        return all_pass, all_dist

    def get_thrs(self, valid_loader, drop_rate=0.05):
        if len(self.sig) == 0:
            logging.warn(
                "[TrapDetector] You need to call build_sig before detect, " +
                "I will do this now.")
            self.build_sig(valid_loader)
        all_dist, thresh = {i: [] for i in self.dataWrapper.target_ls}, {}
        for img, _ in valid_loader:
            img = img.cuda()
            dist, y_pred = self._get_distance(img)
            for y, dis in zip(y_pred, dist):
                all_dist[y].append(dis.cpu())
        for key in all_dist:
            this_dist = torch.stack(all_dist[key], dim=0)
            this_dist, _ = this_dist.sort()
            thrs = this_dist[int(len(this_dist) * drop_rate)].item()
            thresh[key] = thrs
        logging.info("[TrapDetector] get thrs={}".format(thresh))
        return thresh
