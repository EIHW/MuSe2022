import numpy as np
import os
import pandas as pd
import torch
from sklearn.metrics import roc_curve, auc
from scipy import stats 

from config import REACTION_LABELS


def calc_ccc(preds, labels):
    '''
    Concordance Correlation Coefficient
    :param preds: 1D np array
    :param labels: 1D np array
    :return:
    '''

    preds_mean, labels_mean = np.mean(preds), np.mean(labels)
    cov_mat = np.cov(preds, labels)
    covariance = cov_mat[0, 1]
    preds_var, labels_var = cov_mat[0, 0], cov_mat[1, 1]

    ccc = 2.0 * covariance / (preds_var + labels_var + (preds_mean - labels_mean) ** 2)
    return ccc

def mean_ccc(preds, labels):
    '''

    :param preds: list of list of lists (num batches, batch_size, num_classes)
    :param labels: same
    :return: scalar
    '''
    preds = np.row_stack([np.array(p) for p in preds])
    labels = np.row_stack([np.array(l) for l in labels])
    num_classes = preds.shape[1]
    class_wise_cccs = np.array([calc_ccc(preds[:,i], labels[:,i]) for i in range(num_classes)])
    mean_ccc = np.mean(class_wise_cccs)
    return mean_ccc

def calc_pearsons(preds,labels):
    r = stats.pearsonr(preds, labels)
    return r[0]

def mean_pearsons(preds,labels):
    preds = np.row_stack([np.array(p) for p in preds])
    labels = np.row_stack([np.array(l) for l in labels])
    num_classes = preds.shape[1]
    class_wise_r = np.array([calc_pearsons(preds[:,i], labels[:,i]) for i in range(num_classes)])
    mean_r = np.mean(class_wise_r)
    return mean_r


    
def calc_auc(preds, labels):
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    fpr, tpr, thresholds = roc_curve(labels, preds)
    return auc(fpr, tpr)

def write_reaction_predictions(full_metas, full_preds, csv_dir, filename):
    meta_arr = np.row_stack(full_metas).squeeze()
    preds_arr = np.row_stack(full_preds)
    pred_df = pd.DataFrame(columns=['File_ID']+REACTION_LABELS)
    pred_df['File_ID'] = meta_arr
    pred_df[REACTION_LABELS] = preds_arr
    pred_df.to_csv(os.path.join(csv_dir, filename), index=False)
    return None


def write_predictions(task, full_metas, full_preds, prediction_path, filename):
    assert prediction_path != ''


    if not os.path.exists(prediction_path):
        os.makedirs(prediction_path)

    if task == 'reaction':
        return write_reaction_predictions(full_metas, full_preds, prediction_path, filename)

    metas_flat = []
    for meta in full_metas:
        metas_flat.extend(meta)
    preds_flat = []
    for pred in full_preds:
        preds_flat.extend(pred if isinstance(pred, list) else (pred.squeeze() if pred.ndim>1 else pred))

    if isinstance(metas_flat[0],list):
        num_meta_cols = len(metas_flat[0])
    else:
        # np array
        num_meta_cols = metas_flat[0].shape[0]
    prediction_df = pd.DataFrame(columns = [f'meta_col_{i}' for i in range(num_meta_cols)])
    for i in range(num_meta_cols):
        prediction_df[f'meta_col_{i}'] = [m[i] for m in metas_flat]
    prediction_df['prediction'] = preds_flat
    #prediction_df['label'] = labels_flat
    prediction_df.to_csv(os.path.join(prediction_path, filename), index=False)


def evaluate(task, model, data_loader, loss_fn, eval_fn, use_gpu=False, predict=False, prediction_path=None, filename=None):
    losses, sizes = 0, 0
    full_preds = []
    full_labels = []
    if predict:
        full_metas = []
    else:
        full_metas = None

    model.eval()
    with torch.no_grad():
        for batch, batch_data in enumerate(data_loader, 1):
            features, feature_lens, labels, metas = batch_data
            if predict is not True:
                if torch.any(torch.isnan(labels)):
                    print('No labels available, no evaluation')
                    return np.nan, np.nan

            batch_size = features.size(0) if task!='stress' else 1

            if use_gpu:
                model.cuda()
                features = features.cuda()
                feature_lens = feature_lens.cuda()
                labels = labels.cuda()

            preds = model(features, feature_lens)

            # only relevant for stress
            feature_lens = feature_lens.detach().cpu().tolist()
            cutoff = feature_lens[0] if task=='stress' else batch_size
            if predict:
                full_metas.append(metas.tolist()[:cutoff])

            loss = loss_fn(preds.squeeze(-1), labels.squeeze(-1), feature_lens)

            losses += loss.item() * batch_size
            sizes += batch_size

            full_labels.append(labels.cpu().detach().squeeze().numpy().tolist()[:cutoff])
            full_preds.append(preds.cpu().detach().squeeze().numpy().tolist()[:cutoff])

        if predict:
            write_predictions(task, full_metas, full_preds, prediction_path, filename)
            return
        else:
            if task == 'stress':
                full_preds = flatten_stress_for_ccc(full_preds)
                full_labels = flatten_stress_for_ccc(full_labels)
            score = eval_fn(full_preds, full_labels)
            total_loss = losses / sizes
            return total_loss, score

def flatten_stress_for_ccc(lst):
    '''
    Brings full_preds and full_labels of stress into the right format for the CCC function
    :param lst: list of lists of different lengths
    :return: flattened numpy array
    '''
    return np.concatenate([np.array(l) for l in lst])
