import os
import numpy as np
import pandas as pd
import pickle
from glob import glob
from tqdm import tqdm 
from sklearn.preprocessing import StandardScaler

from config import PATH_TO_FEATURES, PATH_TO_LABELS, PARTITION_FILES

################# GLOBAL UTILITY METHODS #############################################

def get_data_partition(partition_file):
    '''
    Reads mappings from subject ids to their partition and vice versa
    :param partition_file: path to the partition file (csv with two columns: id, partition)
    :return: dicts subject2partition, partition2subject
    '''
    subject2partition, partition2subject = {}, {}
    if not os.path.exists(partition_file):
        print(os.path.abspath(partition_file))
    df = pd.read_csv(partition_file)

    for row in df.values:
        subject, partition = str(row[0]), row[-1]
        subject2partition[subject] = partition
        if partition not in partition2subject:
            partition2subject[partition] = []
        if subject not in partition2subject[partition]:
            partition2subject[partition].append(subject)

    return subject2partition, partition2subject


def get_all_training_csvs(task, feature):
    '''
    Loads a list of all feature csvs that are used for training a certain task
    :param task: humor, stress etc.
    :param feature: name of the feature folder (e.g. 'egemaps')
    :return: list of csvs
    '''
    _, partition_to_subject = get_data_partition(PARTITION_FILES[task])
    feature_dir = os.path.join(PATH_TO_FEATURES[task], feature)
    csvs = []
    for subject in tqdm(partition_to_subject['train']):
        if task == 'stress':
            csvs.append(os.path.join(feature_dir, f'{subject}.csv'))
        elif task == 'reaction':
            subject = subject[1:-1] 
            csvs.append(os.path.join(feature_dir, f'{subject}.csv'))
        elif task == 'humor':
            csvs.extend(sorted(glob(os.path.join(feature_dir, subject, "*.csv"))))

    return csvs


def fit_normalizer(task, feature, feature_idx=2):
    '''
    Fits a sklearn StandardScaler based on training data
    :param task: task
    :param feature: feature
    :param feature_idx: index in the feature csv where the features start
    (typically 2, features starting after segment_id, timestamp)
    :return: fitted sklearn.preprocessing.StandardScaler
    '''
    # load training subjects
    training_csvs = get_all_training_csvs(task, feature)
    if task == 'reaction': 
        print('Concatenating csvs') 
        df = pd.concat([pd.read_csv(training_csv) for training_csv in tqdm(training_csvs)]) 
    else:
        df = pd.concat([pd.read_csv(training_csv) for training_csv in training_csvs])
    values = df.iloc[:,feature_idx:].values
    print(f'Scaling values') 
    normalizer = StandardScaler().fit(values)
    return normalizer


################# TASK-SPECIFIC LOADER METHODS FOR SINGLE SUBJECTS #############################################

# --------------------------------------  humor ---------------------------------------------------------------#

def load_humor_subject(feature, subject_id, normalizer):
    '''
    Loads data for a single subject for the humor task
    :param feature: feature name
    :param subject_id: subject name
    :param normalizer: fitted StandardScaler, can be None if no normalization is desired
    :return: features, labels, metas.
        features is a list of ndarrays of shape (seg_len, feature_dim)
        labels is a ndarray of shape (len(features), 1) (label for each element in the features list)
        metas is a ndarray of shape (len(features), 1, 1+len(label columns)) (segment_id, seq_start, seq_end, segment_id)
    '''
    # parse labels
    label_path = PATH_TO_LABELS['humor']
    label_files = sorted(glob(os.path.join(label_path, subject_id + '/*.csv')))
    assert len(label_files) > 0, f'Error: no available humor label files for coach "{subject_id}": "{label_files}".'
    label_df = pd.concat([pd.read_csv(label_file) for label_file in label_files])

    # idx of the data frame (column) where features start
    feature_idx = 2
    feature_path = PATH_TO_FEATURES['humor']

    feature_files = sorted(glob(os.path.join(feature_path, feature, subject_id + '/*.csv')))
    assert len(feature_files) > 0, f'Error: no available "{feature}" feature files for coach "{subject_id}": "{feature_files}".'
    feature_df = pd.concat([pd.read_csv(feature_file) for feature_file in feature_files])
    if not (normalizer is None):
        feature_values = feature_df.iloc[:,feature_idx:].values
        feature_df.iloc[:,feature_idx:] = normalizer.transform(feature_values)
    feature_dim = len(feature_df.columns) - feature_idx

    # load features for each label
    features = []
    for _,y in label_df.iterrows():
        start = y['timestamp_start']
        end = y['timestamp_end']
        segment_id = y['segment_id']
        segment_features = feature_df[feature_df.segment_id == segment_id]
        label_features = segment_features[(segment_features.timestamp >= start) &
                                              (segment_features.timestamp < end)].iloc[:,feature_idx:].values
        # imputation?
        if label_features.shape[0] == 0:
            label_features = np.zeros((1, feature_dim))
        features.append(label_features)

    # store
    # expand for compatibility with the dataset class
    labels = np.expand_dims(label_df.iloc[:,-1].values, -1)
    metas = np.expand_dims(label_df.iloc[:,:-1].values, 1)

    return features, labels, metas


# --------------------------------------  stress ---------------------------------------------------------------#

def segment_stress(sample, win_len, hop_len):
    segmented_sample = []
    assert hop_len <= win_len and win_len >= 10

    for s_idx in range(0, len(sample), hop_len):
        e_idx = min(s_idx + win_len, len(sample))
        segment = sample.iloc[s_idx:e_idx]
        segmented_sample.append(segment)
        if e_idx == len(sample):
            break

    return segmented_sample


def load_stress_subject(feature, subject_id, partition, emo_dim, normalizer, apply_segmentation=True,
                win_len=200, hop_len=100):
    '''
    Loads data for a single subject for the stress task
    :param feature: feature name
    :param subject_id: subject name
    :param normalizer: fitted StandardScaler, can be None if no normalization is desired
    :param apply_segmentation: apply segmentation method?
    :param win_len: window length for segmentation
    :param hop_len: hop length for segmentation
    :return: features, labels, metas.
            features is a list of ndarrays of shape (seq_len, feature_dim)
            labels is a list of ndarrays corresponding to features, each shaped (seq_len, 1) accordingly
            metas is a list of ndarrays corresponding to features, each shaped (seq_len, 3) accordingly
                (subject_id, timestamp, segment_id)
    '''
    # this will contain timestamp, segment_id, features f1...fn, label
    sample_data = []

    feature_idx = 2

    feature_path = PATH_TO_FEATURES['stress']

    feature_file = os.path.join(feature_path, feature, subject_id + '.csv')
    assert os.path.exists(
            feature_file), f'Error: no available "{feature}" feature file for video "{subject_id}": "{feature_file}".'
    feature_data = pd.read_csv(feature_file)
    feature_dim = feature_data.shape[1] - feature_idx

    feature_values = feature_data.iloc[:,-feature_dim:].values
    if not(normalizer is None):
        feature_data.iloc[:,-feature_dim:] = normalizer.transform(feature_values)
    feature_data.iloc[:,-feature_dim:] = np.nan_to_num(feature_values)
    sample_data.append(feature_data)

    # parse labels
    label_path = PATH_TO_LABELS['stress']
    label_file = os.path.join(label_path, emo_dim, subject_id + '.csv')
    assert os.path.exists(
        label_file), f'Error: no available "{emo_dim}" label file for video "{subject_id}": "{label_file}".'
    df = pd.read_csv(label_file)
    # timestamp from label file are the relevant ones
    timestamps = df.timestamp.values

    label_data = pd.DataFrame(data=df['value'].values, columns=[emo_dim])
    sample_data.append(label_data)

    # concat
    sample_data = pd.concat(sample_data, axis=1)
    if partition != 'test':
        sample_data = sample_data.dropna()
    else:
        sample_data = sample_data.fillna(0)
    sample_data['timestamp'] = timestamps

    if apply_segmentation:
        samples = segment_stress(sample_data, win_len, hop_len)
    else:
        samples = [sample_data]

    # store
    features = []
    labels = []
    metas = []
    for i, segment in enumerate(samples):  # each segment has columns: timestamp, segment_id, features, labels
        n_emo_dims = 1
        if len(segment.iloc[:, feature_idx:-n_emo_dims].values) > 0:  # check if there are features
            meta = np.column_stack((np.array([int(subject_id)] * len(segment)),
                                    segment.iloc[:, :feature_idx].values))  # video_id, timestamp, segment_id
            metas.append(meta)
            labels.append(segment.iloc[:, -n_emo_dims:].values)
            features.append(segment.iloc[:, feature_idx:-n_emo_dims].values)

    return features, labels, metas


# --------------------------------------  reaction ---------------------------------------------------------------#

def load_reaction_subject(feature, subject_id, normalizer):
    '''
    Loads data for a single subject for the reaction task
    :param feature: feature name
    :param subject_id: subject name/ID
    :param normalizer: fitted StandardScaler, can be None if no normalization is desired. It is created in the load_data
    method, so no need to take care of that. It just needs to be called in the load_reaction_subject method somewhere
    to normalize the features
    :return: features, labels, metas.
        Assuming every subject consists of n segments of lengths l_1,...,l_n:
            features is a list (length n) of ndarrays of shape (l_i, feature_dim)  - each item corresponding to a segment
            labels is a ndarray of shape (n, num_classes) (labels for each element in the features list, assuming every segment
                has num_classes labels)
            metas is a ndarray of shape (n, 1, x) where x is the number of columns needed to describe the segment
                Typically something like (subject_id, segment_id, seq_start, seq_end) or the like
                They are only used to write the predictions: a prediction line consists of all the meta data associated
                    with one data point + the predicted label(s)
    '''
    # parse labels
    label_path = PATH_TO_LABELS['reaction']
    label_df = pd.read_csv(os.path.join(label_path, 'labels.csv'))
    labels = label_df[label_df.File_ID==subject_id].iloc[:,1:].values
    assert labels.shape == (1,7), f"Malformed label file for ID {subject_id}"

    feature_path = os.path.join(PATH_TO_FEATURES['reaction'], feature)
    feature_df = pd.read_csv(os.path.join(feature_path, f'{subject_id.replace("[","").replace("]","")}.csv')) 

    feature_idx = 2
    features = feature_df.iloc[:,feature_idx:].values
    if not (normalizer is None):
        features = normalizer.transform(features)
    features = [features]

    metas = np.array([subject_id]).reshape((1,1,1))

    return features, labels, metas


################# LOAD DATASETS USING THE SPECIFIC METHODS ABOVE #############################################

def load_data(task, paths, feature, emo_dim, normalize=True, win_len=200, hop_len=100, save=False,
              segment_train=True):
    '''
    Loads the complete data sets
    :param task: task
    :param paths: dict for paths to data and partition file
    :param feature: feature to load
    :param emo_dim: emotion dimension to load labels for
    :param normalize: whether normalization is desired
    :param win_len: window length for segmentation (ignored for humor - and reaction?)
    :param hop_len: hop length for segmentation (ignored for humor - and reaction?)
    :param save: whether to cache the loaded data as .pickle
    :param segment_train: whether to do segmentation on the training data
    :return: dict with keys 'train', 'devel' and 'test', each in turn a dict with keys:
        feature: list of ndarrays shaped (seq_length, features)
        labels: corresponding list of ndarrays shaped (seq_length, 1) for n-to-n tasks like stress, (1,) for n-to-1
            task humor, (7,) for n-to-7 task reaction
        meta: corresponding list of ndarrays shaped (seq_length, metadata_dim) where seq_length=1 for n-to-1/n-to-7
    '''

    data_file_name = f'data_{task}_{feature}_{emo_dim+"_" if len(emo_dim)>0 else ""}_{"norm_" if normalize else ""}{win_len}_' \
        f'{hop_len}{"_seg" if segment_train else ""}.pkl'
    data_file = os.path.join(paths['data'], data_file_name)

    if os.path.exists(data_file):  # check if file of preprocessed data exists
        print(f'Find cached data "{os.path.basename(data_file)}".')
        data = pickle.load(open(data_file, 'rb'))
        return data

    print('Constructing data from scratch ...')
    data = {'train': {'feature': [], 'label': [], 'meta': []},
            'devel': {'feature': [], 'label': [], 'meta': []},
            'test': {'feature': [], 'label': [], 'meta': []}}
    subject2partition, partition2subject = get_data_partition(paths['partition'])
    print('Normalising data') if normalize else None 
    normalizer = fit_normalizer(task=task, feature=feature) if normalize else None

    for partition, subject_ids in partition2subject.items():
        print(f'Setting up {partition} Partition') 

        apply_segmentation = segment_train and partition=='train'

        for subject_id in tqdm(subject_ids):
            if task == 'stress':
                features, labels, metas = load_stress_subject(feature=feature, subject_id=subject_id,
                                                              partition=partition, emo_dim=emo_dim,
                                                              normalizer=normalizer,
                                                              apply_segmentation=apply_segmentation, win_len=win_len,
                                                              hop_len=hop_len)
            elif task == 'humor':
                features, labels, metas = load_humor_subject(feature=feature, subject_id=subject_id,
                                                             normalizer=normalizer)
            elif task == 'reaction':
                features, labels, metas = load_reaction_subject(feature=feature, subject_id=subject_id,
                                                                normalizer=normalizer)

            data[partition]['feature'].extend(features)
            data[partition]['label'].extend(labels)
            data[partition]['meta'].extend(metas)

    if save:  # save loaded and preprocessed data
        print('Saving data...')
        pickle.dump(data, open(data_file, 'wb'))

    return data
