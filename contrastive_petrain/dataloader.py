import torch
import random
from pathlib import Path
from tqdm import tqdm
import argparse
import datetime
import pickle
import sys
import time
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from torchnet import meter
import warnings
import h5py
import csv
import os
import math
from re import L
import subprocess
import pickle
from collections import defaultdict


def ensure_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False



def get_label_id_dict(names, labels):
    id_label = {}
    label_id = {}
    for name, label in zip(names, labels):
        id_label[name] = int(label)
        if int(label) not in label_id:
            label_id[int(label)] = set()
        label_id[int(label)].add(name)
    return id_label, label_id



def parse_fasta(dir, number=None):
    names = []
    sequences = []
    labels = []
    if number is None:
        number = -1
    with open(dir, 'r') as f:
        data = f.readlines()
        for i in range(0, len(data[:number]), 2):
            line = data[i]
            if line.startswith('>'):
                label = int(line.split('|')[-1])
                labels.append(label)
                names.append(line.strip()[1:])
                sequences.append(data[i + 1].strip())
    return np.array(names), np.array(sequences), np.array(labels)



def get_pretrained_features(names, sequences, pre_dict_path, theta=50):
    features = {}
    with h5py.File(pre_dict_path, "r") as h5fi:
        for name, seq in zip(names, sequences):
            pre_features_ref = h5fi[name][:]
            sequence_len = len(seq)
            feature_length = min(pre_features_ref.shape[0], theta, sequence_len)
            feature = np.zeros((theta, pre_features_ref.shape[-1]), dtype=np.float32)
            if feature_length > 0:
                feature[:feature_length, :] = pre_features_ref[:feature_length, :]
            features[name] = feature
    return features



def data_pre(txt_path, theta=40):
    txt_path = Path(txt_path).expanduser().resolve()
    h5_path = txt_path.with_suffix('.h5')

    if not txt_path.exists():
        raise FileNotFoundError(f'txt文件不存在: {txt_path}')
    if txt_path.suffix.lower() != '.txt':
        raise ValueError(f'输入文件必须是 .txt 后缀: {txt_path}')
    if not h5_path.exists():
        raise FileNotFoundError(f'对应的h5文件不存在: {h5_path}')

    names, sequences, labels = parse_fasta(str(txt_path), number=None)
    pre_feas = get_pretrained_features(names, sequences, str(h5_path), theta=theta)
    return pre_feas, labels, names, sequences



def feature_dict_to_tensor_dict(feature_dict):
    return {k: torch.from_numpy(v).float() if isinstance(v, np.ndarray) else v.float() for k, v in feature_dict.items()}


class Triplet_dataset_with_mine_EC(torch.utils.data.Dataset):
    def __init__(self, id_label, label_id, negative, pre_feas_dict):
        self.id_label = id_label
        self.label_id = label_id
        self.negative = negative
        self.pre_feas_dict = feature_dict_to_tensor_dict(pre_feas_dict)
        self.sample_ids = list(id_label.keys())

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        anchor_id = self.sample_ids[index]
        pos_candidates = list(self.label_id[self.id_label[anchor_id]])
        pos_candidates = [sid for sid in pos_candidates if sid != anchor_id]
        positive_id = np.random.choice(pos_candidates) if pos_candidates else anchor_id

        if anchor_id in self.negative and self.negative[anchor_id]['negative']:
            neg_candidates = self.negative[anchor_id]['negative']
            weights = self.negative[anchor_id]['weights']
            negative_id = np.random.choice(neg_candidates, p=weights)
        else:
            diff_ids = [sid for sid in self.sample_ids if self.id_label[sid] != self.id_label[anchor_id]]
            negative_id = np.random.choice(diff_ids) if diff_ids else anchor_id

        return (
            self.pre_feas_dict[anchor_id],
            self.pre_feas_dict[positive_id],
            self.pre_feas_dict[negative_id],
        )


class TripletDatasetWithValidationNegatives(torch.utils.data.Dataset):
    def __init__(self, train_id_label, val_id_label, train_label_id, val_label_id, negative_dict,
                 pre_feas_train, pre_feas_valid, knn=10):
        self.train_id_label = train_id_label
        self.val_id_label = val_id_label
        self.train_label_id = train_label_id
        self.val_label_id = val_label_id
        self.pre_feas_valid = feature_dict_to_tensor_dict(pre_feas_valid)
        self.pre_feas_train = feature_dict_to_tensor_dict(pre_feas_train)
        self.knn = knn
        self.negative_dict = negative_dict
        self.val_ids = list(val_id_label.keys())
        self.train_ids = list(train_id_label.keys())

    def __len__(self):
        return len(self.val_ids)

    def __getitem__(self, index):
        anchor_id = self.val_ids[index]
        anchor_feat = self.pre_feas_valid[anchor_id]

        pos_candidates = list(self.train_label_id[self.val_id_label[anchor_id]])
        pos_candidates = [sid for sid in pos_candidates if sid != anchor_id]
        positive_id = np.random.choice(pos_candidates) if pos_candidates else anchor_id
        positive_feat = self.pre_feas_train[positive_id]

        if anchor_id in self.negative_dict and self.negative_dict[anchor_id]['negative']:
            neg_candidates = self.negative_dict[anchor_id]['negative']
            weights = self.negative_dict[anchor_id]['weights']
            negative_id = np.random.choice(neg_candidates, p=weights)
        else:
            diff_ids = [sid for sid in self.train_ids if self.train_id_label[sid] != self.val_id_label[anchor_id]]
            negative_id = np.random.choice(diff_ids) if diff_ids else anchor_id
        negative_feat = self.pre_feas_train[negative_id]

        return anchor_feat, positive_feat, negative_feat


class TripletDatasetCross(Dataset):
    """
    提速版：
    1. 特征只在初始化时转一次 tensor
    2. 预构建 label -> pool_ids，避免 __getitem__ 里反复全池扫描
    3. 预构建不同标签候选，降低 Python 开销
    """
    def __init__(self, anchor_ids, id_label_anchor, anchor_feas,
                 pool_ids, id_label_pool, pool_feas, neg_dict):
        self.anchor_ids = list(anchor_ids)
        self.id_label_a = id_label_anchor
        self.pool_ids = list(pool_ids)
        self.id_label_p = id_label_pool
        self.neg_dict = neg_dict

        self.anchor_feas = feature_dict_to_tensor_dict(anchor_feas)
        self.pool_feas = feature_dict_to_tensor_dict(pool_feas)

        self.pool_by_label = defaultdict(list)
        for pid in self.pool_ids:
            self.pool_by_label[self.id_label_p[pid]].append(pid)

        all_pool_by_label = {label: ids[:] for label, ids in self.pool_by_label.items()}
        self.diff_pool_by_label = {}
        for aid in self.anchor_ids:
            a_label = self.id_label_a[aid]
            diff = []
            for label, ids in all_pool_by_label.items():
                if label != a_label:
                    diff.extend(ids)
            self.diff_pool_by_label[aid] = diff

    def __len__(self):
        return len(self.anchor_ids)

    def __getitem__(self, idx):
        aid = self.anchor_ids[idx]
        a_label = self.id_label_a[aid]
        a_feas = self.anchor_feas[aid]

        same_label_pool = self.pool_by_label.get(a_label, [])
        if same_label_pool:
            pos_id = random.choice(same_label_pool)
            if pos_id == aid and len(same_label_pool) > 1:
                while pos_id == aid:
                    pos_id = random.choice(same_label_pool)
        else:
            pos_id = aid
        p_feas = self.pool_feas[pos_id] if pos_id in self.pool_feas else self.anchor_feas[aid]

        if aid in self.neg_dict and self.neg_dict[aid]['negative']:
            neg_cands = self.neg_dict[aid]['negative']
            weights = self.neg_dict[aid]['weights']
            neg_id = np.random.choice(neg_cands, p=weights)
        else:
            diff = self.diff_pool_by_label[aid]
            neg_id = random.choice(diff) if diff else aid
        n_feas = self.pool_feas[neg_id] if neg_id in self.pool_feas else self.anchor_feas[aid]

        return a_feas, p_feas, n_feas
