from argparse import ArgumentParser, Namespace
import os
from shutil import rmtree

import pandas as pd

from glob import glob
from tqdm import tqdm

from config import PATH_TO_FEATURES


parser = ArgumentParser()
parser.add_argument('--task', type=str, required=True, choices=['humor', 'reaction', 'stress'])
parser.add_argument('--feature_sets', nargs='+', required=True, help='feature sets to fuse, for early fusion')

def parse_args():
    args = parser.parse_args()
    assert len(args.feature_sets) >= 2, "For early fusion, please give at least 2 different feature sets"
    return args

def get_first_features(feature_names, task):        
        feature_a = tqdm(glob(os.path.join(PATH_TO_FEATURES[task], feature_names[0], '**', '*.csv'), recursive=True))
        return feature_a 
        
        
def combine_feature_sets(feature_names, file, task, feature_idx): 
        file_id = os.path.basename(file) if task in ['reaction', 'stress'] else \
            os.path.sep.join(file.split(os.path.sep)[-2:])
        feature_set_a = pd.read_csv(file)
        feature_set_others = [pd.read_csv(os.path.join(PATH_TO_FEATURES[task], feat_name, file_id))
                                  .iloc[:,feature_idx:] for feat_name in feature_names[1:]]
        df_concat = pd.concat([feature_set_a]+ feature_set_others,1)
        df_concat = df_concat.fillna(0)
        return df_concat,file_id

def main(args, feature_idx=2):
    task = args.task 
    feature_names = args.feature_sets
    print(f'Concatenating features {"+".join(feature_names)} for early fusion')
    target_dir = os.path.join(PATH_TO_FEATURES[task], f'early-fusion-{"+".join(feature_names)}')
    if os.path.exists(target_dir):
        rmtree(target_dir)
    os.makedirs(target_dir)
    print(f'Features stored at: {target_dir}')
    feature_a = get_first_features(feature_names, task)

    for file in tqdm(feature_a):
        df_concat, file_id = combine_feature_sets(feature_names,file, task,feature_idx)
        if task == 'humor':
            os.makedirs(os.path.dirname(os.path.join(target_dir, file_id)), exist_ok=True)
        df_concat.to_csv(os.path.join(target_dir, file_id),index=False)
        
if __name__ == '__main__':
    args = parse_args()
    main(args)

