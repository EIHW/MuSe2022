
from argparse import ArgumentParser, Namespace
import os
from shutil import rmtree

import pandas as pd

import torch
from glob import glob
from tqdm import tqdm

from config import MODEL_FOLDER, device, PATH_TO_FEATURES


parser = ArgumentParser()
parser.add_argument('--task', type=str, required=True, choices=['humor', 'reaction', 'stress'])
parser.add_argument('--model_ids', nargs='+', required=True, help='model ids')
parser.add_argument('--aliases', nargs='+', default=None, help='Preferably shorter aliases for the model ids. '
                                                               'Optional, script will take the feature names by default.')
parser.add_argument('--name', type=str, default=None, help='Optional name for the new "feature set". If not given,'
                                                           'name will be calculated from the aliases.')
parser.add_argument('--checkpoint_seeds', nargs='+', required=True, help='Checkpoints to use, e.g. '
                                                                         '101 if for model that was trained with seed 101 '
                                                                         '(cf. output in the model directory)')

def parse_args():
    args = parser.parse_args()
    assert len(set(args.model_ids)) == len(args.model_ids), "Error, duplicate model file"
    assert len(args.model_ids) >= 2, "For late fusion, please give at least 2 different models"
    if not args.aliases is None:
        assert(len(args.aliases) == len(args.model_ids))
    assert(len(args.checkpoint_seeds) == len(args.model_ids)), "Number of checkpoint_seeds must match number of given model_ids"
    dims = []
    features = []
    aliases = [] if args.aliases is None else args.aliases
    model_configs = []
    for m in args.model_ids:
        split_name = [x.replace("]","").replace("[","") for x in m.split("_")]
        features.append(split_name[1])
        dims.append(split_name[2] if args.task=='stress' else None)
        if args.aliases is None:
            aliases.append(split_name[1])

        model_configs.append({
            'd_rnn': int(split_name[-6]),
            'rnn_n_layers': int(split_name[-5]),
            'rnn_bi': bool(split_name[-4]),
            'd_fc_out': int(split_name[-3]),
            'rnn_dropout': 0.,
            'n_to_1': True,
            'd_in': infer_feature_dim(args.task, split_name[1]),
            'n_targets': 7 if args.task=='reaction' else 1,
            'linear_dropout': 0.,
            'task': args.task
        })
    args.dims = dims
    args.features = features
    args.aliases = aliases
    if args.name is None:
        args.name = "+".join(args.aliases)
    args.model_files = [os.path.join(MODEL_FOLDER, args.task, m, f'model_{args.checkpoint_seeds[i]}.pth')
                           for i,m in enumerate(args.model_ids)]
    args.model_configs = [Namespace(**config) for config in model_configs]
    return args


def infer_feature_dim(task, feature_name, feature_idx=2):
    feature_csvs = glob(os.path.join(PATH_TO_FEATURES[task], feature_name, '**', '*.csv'), recursive=True)
    random_df = pd.read_csv(feature_csvs[0])
    return len(random_df.columns) - feature_idx


def predictions_for_csv(csv, model, feature_idx=2):
    '''
    Extracts model predictions, given a feature csv and a model
    :param csv: feature csv
    :param model: model
    :param feature_idx: position where the features start in the csv
    :return: numpy array (len(csv), pred_dim) where pred_dim == 1 for stress and humor, 7 for reaction
    '''
    model.set_n_to_1(False)
    model.to(device)
    model.eval()
    df = pd.read_csv(csv)
    feature_arr = df.iloc[:,feature_idx:].values
    lens = torch.tensor([feature_arr.shape[0]])
    lens = lens.to(device)
    features = torch.Tensor(feature_arr).unsqueeze(0)
    features = features.to(device)
    predictions = model(features, lens).squeeze(0)
    return predictions.detach().cpu().numpy()


def main(args, feature_idx=2):
    # create a directory
    target_dir = os.path.join(PATH_TO_FEATURES[args.task], f'late-fusion-{args.name}')
    if os.path.exists(target_dir):
        rmtree(target_dir)
    os.makedirs(target_dir)
    # load all models
    num_models = len(args.model_files)

    for i in range(num_models):
        model = torch.load(args.model_files[i])
        # load all csvs
        feature = args.features[i]
        csvs = sorted(glob(os.path.join(PATH_TO_FEATURES[args.task], feature, '**', '*.csv'), recursive=True))
        print(f"Predicting for model {i+1}/{num_models}...")
        for csv in tqdm(csvs):
            predictions = predictions_for_csv(csv, model)
            pred_dim = predictions.shape[1]
            pred_lists = [predictions[:,j].tolist() for j in range(pred_dim)]
            pred_cols = [f'{args.aliases[i]}_{j}' for j in range(pred_dim)]
            csv_last_part = csv.replace(os.path.join(PATH_TO_FEATURES[args.task], feature), "")
            target_csv = target_dir + csv_last_part
            os.makedirs(os.path.dirname(target_csv), exist_ok=True)

            # create a new csv with all meta columns
            feature_df = pd.read_csv(csv)
            meta_df = feature_df.iloc[:,:feature_idx]
            meta_cols = list(feature_df.columns)[:feature_idx]
            # construct new df: meta data + first predictions
            dct = {meta_cols[k]: meta_df.iloc[:,k] for k in range(len(meta_cols))}
            for col,values in zip(pred_cols, pred_lists):
                dct.update({col:values})
            df = pd.DataFrame(dct)
            # hack for alignment, so that the join does not change the length
            if args.task == 'stress':
                if df.timestamp.values[0] == 0:
                    df.timestamp = df.timestamp.values + 500

            if os.path.exists(target_csv):
                df_existing = pd.read_csv(target_csv)
                join_type = 'outer'
                df = pd.merge(df_existing, df, how= join_type, on=meta_cols)

                df.sort_values(by='timestamp', inplace=True)
                pass
                df.fillna(method='ffill', inplace=True)
                df.fillna(method='bfill', inplace=True)

            df.to_csv(target_csv, index=False)

if __name__ == '__main__':
    args = parse_args()
    main(args)


