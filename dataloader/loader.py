"""
VOCASET loader (and dispatch to the TFHP loader for --dataset TFHP).

    <dataset_root>/wav/<seq>.wav, <dataset_root>/vertices_npy/<seq>.npy (60 fps, sub-sampled to 30 fps),
    <dataset_root>/templates.pkl {subject: (5023, 3) neutral mesh}

Speaker splits come from --train/val/test_subjects; train uses sentences 1-40, val/test sentences 21-40.
Each item is (audio input_values, vertices (T, 3V), template (3V), one-hot, file name).
"""
import os
import pickle
from collections import defaultdict

import librosa
import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm
from transformers import Wav2Vec2Processor


def _resolve_dataset_root(args):
    root = getattr(args, "dataset_root", None)
    if not root:
        raise ValueError(f"dataset_root is not set for dataset '{args.dataset}'")
    return root


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, subjects_dict, data_type="train"):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        file_name = self.data[index]["name"]
        audio = self.data[index]["audio"]
        vertice = self.data[index]["vertice"]
        template = self.data[index]["template"]
        if self.data_type == "train":
            subject = "_".join(file_name.split("_")[:-1])
            one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
        else:
            one_hot = self.one_hot_labels
        return torch.FloatTensor(audio), torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(one_hot), file_name

    def __len__(self):
        return self.len


def read_data(args):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    dataset_root = _resolve_dataset_root(args)
    audio_path = os.path.join(dataset_root, args.wav_path)
    vertices_path = os.path.join(dataset_root, args.vertices_path)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    template_file = os.path.join(dataset_root, args.template_file)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')

    for r, ds, fs in os.walk(audio_path):
        for f in tqdm(fs):
            if f.endswith("wav"):
                wav_path = os.path.join(r, f)
                speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                input_values = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
                key = f.replace("wav", "npy")
                data[key]["audio"] = input_values
                subject_id = "_".join(key.split("_")[:-1])
                temp = templates[subject_id]
                data[key]["name"] = f
                data[key]["template"] = temp.reshape((-1))
                vertice_path = os.path.join(vertices_path, f.replace("wav", "npy"))
                if not os.path.exists(vertice_path):
                    del data[key]
                else:
                    data[key]["vertice"] = np.load(vertice_path, allow_pickle=True)[::2, :]   # 60 -> 30 fps

    subjects_dict = {}
    subjects_dict["train"] = [i for i in args.train_subjects.split(" ")]
    subjects_dict["val"] = [i for i in args.val_subjects.split(" ")]
    subjects_dict["test"] = [i for i in args.test_subjects.split(" ")]

    splits = {'train': range(1, 41), 'val': range(21, 41), 'test': range(21, 41)}

    for k, v in data.items():
        subject_id = "_".join(k.split("_")[:-1])
        sentence_id = int(k.split(".")[0][-2:])
        if subject_id in subjects_dict["train"] and sentence_id in splits['train']:
            train_data.append(v)
        if subject_id in subjects_dict["val"] and sentence_id in splits['val']:
            valid_data.append(v)
        if subject_id in subjects_dict["test"] and sentence_id in splits['test']:
            test_data.append(v)

    print(len(train_data), len(valid_data), len(test_data))
    return train_data, valid_data, test_data, subjects_dict


def get_dataloaders(args):
    # TFHP: lazy split loader in tfhp_loader.py (head-pose-free wo_vert, 100-frame windows, train hop 50)
    if getattr(args, "dataset", None) == "TFHP":
        from .tfhp_loader import get_dataloaders as _tfhp_get_loaders
        args.dataset_root = _resolve_dataset_root(args)
        if not hasattr(args, "tfhp_use_headpose"):
            args.tfhp_use_headpose = False
        if not hasattr(args, "tfhp_window_frames"):
            args.tfhp_window_frames = 100
        if not hasattr(args, "tfhp_overlap_frames"):
            args.tfhp_overlap_frames = 50
        if not hasattr(args, "tfhp_coef_fps"):
            args.tfhp_coef_fps = 25
        if not hasattr(args, "num_workers"):
            args.num_workers = 0
        return _tfhp_get_loaders(args)

    if args.dataset != "vocaset":
        raise ValueError(f"Unsupported dataset: {args.dataset} (vocaset | TFHP)")

    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args)
    # VOCASET sequences have different lengths: train batch size > 1 needs equal lengths (val/test always 1)
    _bs = getattr(args, "batch_size", 1)
    train_data = Dataset(train_data, subjects_dict, "train")
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=_bs, shuffle=True)
    valid_data = Dataset(valid_data, subjects_dict, "val")
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False)
    test_data = Dataset(test_data, subjects_dict, "test")
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    return dataset
