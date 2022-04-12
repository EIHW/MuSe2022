import pathlib
import sys
import torch
import numpy as np
import random
import pandas as pd
import os


class Logger(object):
    def __init__(self, log_file="log_file.log"):
        self.terminal = sys.stdout
        self.file = open(log_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.file.flush()


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def log_results(csv_path, params, val_results, test_results, seeds, best_idx, model_files, metric_name,
                exclude_keys=['result_csv', 'cache', 'save', 'save_path', 'predict', 'eval_model', 'log_file_name']):
    '''
    Logs result of a run into a csv
    :param csv_path: path to the desired csv. Appends, if csv exists, else creates it anew
    :param params: configuration of the run (parsed cli arguments)
    :param val_results: array of validation metric results
    :param test_results: array of test metric results
    :param best_idx: index of the chosen result
    :param model_files: list of saved model files
    :param metric_name: name of the used metric
    :param exclude_keys: keys in params not to consider for logging
    :return: None
    '''
    dct = {k:[v] for k,v in vars(params).items() if not k in exclude_keys}
    dct.update({'best_seed':seeds[best_idx]})
    dct.update({f'best_val_{metric_name}': val_results[best_idx]})
    dct.update({f'test_{metric_name}':test_results[best_idx]})
    dct.update({f'mean_val_{metric_name}': np.mean(np.array(val_results))})
    dct.update({f'std_val_{metric_name}': np.std(np.array(val_results))})
    dct.update({f'mean_test_{metric_name}': np.mean(np.array(test_results))})
    dct.update({f'std_test_{metric_name}': np.std(np.array(test_results))})
    dct.update({'model_file':model_files[best_idx]})
    df = pd.DataFrame(dct)

    # make sure the directory exists
    csv_dir = pathlib.Path(csv_path).parent.resolve()
    os.makedirs(csv_dir, exist_ok=True)

    # write back
    if os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)
        df = pd.concat([old_df, df])
    df.to_csv(csv_path, index=False)