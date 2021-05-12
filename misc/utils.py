import glob
import os
import torch
import logging

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
        torch.save(
            params, results_dir +
            '/{}_{}E{}V{:.6f}.pth'.format(arch, prefix, epoch, acc))
        print(">>>>>> Best saved <<<<<<")
    else:
        print(">>>>>> Best not change from {} <<<<<<".format(best_prec))
    return best_prec