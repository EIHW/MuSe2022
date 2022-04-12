import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset
import numpy as np


class MuSeDataset(Dataset):
    def __init__(self, data, partition):
        super(MuSeDataset, self).__init__()
        self.partition = partition
        features, labels = data[partition]['feature'], data[partition]['label']
        metas = data[partition]['meta']
        self.feature_dim = features[0].shape[-1]
        self.n_samples = len(features)

        feature_lens = []
        label_lens = []
        for feature in features:
            feature_lens.append(len(feature))
        for label in labels:
            if label.ndim == 1:
                label_lens.append(1)
            else:
                label_lens.append(label.shape[0])
        max_feature_len = np.max(np.array(feature_lens))
        max_label_len = np.max(np.array(label_lens))
        if max_label_len > 1:
            assert(max_feature_len==max_label_len)

        self.feature_lens = torch.tensor(feature_lens)

        features = [np.pad(f, pad_width=((0, max_feature_len-f.shape[0]),(0,0))) for f in features]
        self.features = [torch.tensor(f, dtype=torch.float) for f in features]
        # if n-to-n task like stress
        if max_label_len > 1:
            labels = [np.pad(l, pad_width=((0, max_label_len-l.shape[0]),(0,0))) for l in labels]
        self.labels = [torch.tensor(l, dtype=torch.float) for l in labels]
        self.metas = [np.pad(meta, pad_width=((0,max_label_len-meta.shape[0]),(0,0)), mode='empty') for meta in metas]

        self.metas = [m.astype(np.object).tolist() for m in self.metas]
        pass

    def get_feature_dim(self):
        return self.feature_dim

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        '''

        :param idx:
        :return: feature, feature_len, label, meta with
            feature: tensor of shape seq_len, feature_dim
            feature_len: int tensor, length of the feature tensor before padding
            label: tensor of corresponding label(s) (shape 1 for n-to-1, else (seq_len,1))
            meta: list of lists containing corresponding meta data
        '''
        feature = self.features[idx]
        feature_len = self.feature_lens[idx]
        label = self.labels[idx]
        meta = self.metas[idx]

        sample = feature, feature_len, label, meta
        return sample


def custom_collate_fn(data):
    '''
    Custom collate function to ensure that the meta data are not treated with standard collate, but kept as ndarrays
    :param data:
    :return:
    '''
    tensors = [d[:3] for d in data]
    np_arrs = [d[3] for d in data]
    coll_features, coll_feature_lens, coll_labels = default_collate(tensors)
    np_arrs_coll = np.row_stack(np_arrs)
    return coll_features, coll_feature_lens, coll_labels, np_arrs_coll
