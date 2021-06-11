import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
import random
import logging


class CoreModel(nn.Module):
    def __init__(self, dataset, classifier, cls_norm):
        super(CoreModel, self).__init__()
        self.dataset = dataset
        self.classifier = classifier
        self.cls_norm = Normalize(*cls_norm)
        if dataset == "cifar10":
            num_classes = 10
            img_shape = (3, 32, 32)
            per_label_ratio = 0.1
            expect_acc = 0.8  # change this from 0.75 to 0.8
        elif dataset == "gtsrb":
            num_classes = 43
            img_shape = (3, 32, 32)
            per_label_ratio = 0.1
            expect_acc = 0.9
        elif dataset == "MNIST":
            num_classes = 10
            img_shape = (1, 32, 32)
            per_label_ratio = 0.1
            expect_acc = 0.98
        else:
            raise Exception("Not implement")

        self.num_classes = num_classes
        self.img_shape = img_shape
        self.per_label_ratio = per_label_ratio
        self.expect_acc = expect_acc

    def inference_logits(self, data):
        norm_x = self.cls_norm(data)
        return self.classifier(norm_x)

    def forward(self, data):
        logits = self.inference_logits(data)
        return F.softmax(logits, dim=-1)

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
                cur_y = random.choice(self.target_ls)
                cur_x = self.injection(cur_x, cur_y)
            batch_X.append(cur_x)
            batch_Y.append(torch.LongTensor([cur_y]))
        return torch.stack(batch_X, dim=0), torch.cat(batch_Y, dim=0)
