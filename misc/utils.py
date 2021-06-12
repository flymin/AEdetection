import glob
import os
import torch
import logging
import numpy as np
from urllib import parse as urlparse


def isfloat(value):
    try:
        float(value)
        return True
    except:
        return False


def parse_params(params_str):
    if '?' not in params_str:
        return params_str, {}
    subject, params_str = params_str.split('?')
    params = urlparse.parse_qs(params_str)
    params = dict((k, v.lower() if len(v) > 1 else v[0])
                  for k, v in params.items())

    # Data type conversion.
    integer_parameter_names = [
        'batch_size', 'max_iterations', 'num_classes', 'max_iter', 'nb_iter',
        'max_iter_df']
    for k, v in params.items():
        if k in integer_parameter_names:
            params[k] = int(v)
        elif v == 'true':
            params[k] = True
        elif v == 'false':
            params[k] = False
        elif v == 'inf':
            params[k] = np.inf
        elif isfloat(v):
            params[k] = float(v)

    return subject, params


def make_logger(run_name, log_output):
    if log_output is not None:
        if run_name == None:
            run_name = log_output.split('/')[-1].split('.')[0]
        else:
            log_output = os.path.join(log_output, f'{run_name}.log')
    logger = logging.getLogger()
    logger.propagate = False
    log_filepath = log_output if log_output is not None else os.path.join(
        'results/log', f'{run_name}.log')

    log_dir = os.path.dirname(os.path.abspath(log_filepath))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not logger.handlers:  # execute only if logger doesn't already exist
        file_handler = logging.FileHandler(log_filepath, 'a', 'utf-8')
        stream_handler = logging.StreamHandler(os.sys.stdout)

        formatter = logging.Formatter(
            '[%(levelname)s] %(asctime)s > %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')

        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)
    return logger


def save_best(best_prec, prefix, acc, params, epoch, arch, results_dir,
              min_mode=False):
    on_save = acc <= best_prec if min_mode else acc >= best_prec
    if on_save:
        best_prec = acc
        for file in glob.glob(
                results_dir + '/{}_{}E*'.format(arch, prefix)):
            os.remove(file)
        save_name = '/{}_{}E{}V{:.6f}.pth'.format(arch, prefix, epoch, acc)
        torch.save(params, results_dir + save_name)
        logging.info(">>>>>> Best saved to {} <<<<<<".format(save_name))
    else:
        logging.info(">>>>>> Best not change from {} <<<<<<".format(best_prec))
    return best_prec


def judge_thresh(out, thresh, min_distance=False):
    # True for pass, False for reject
    if min_distance:
        return (out < thresh).long()
    else:
        return (out > thresh).long()
