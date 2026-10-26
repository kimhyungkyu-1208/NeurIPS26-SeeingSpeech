"""
TFHP loader (head-pose-free variant, 25 fps): lazy, split-file based, fixed-length windows.

    <dataset_root>/{train,val,test}.txt                          <person>/<seq> per line (speaker splits)
    <dataset_root>/TFHP_wo_Head_v03/<person>_<seq>_<clip>/{audio.wav, wo_vert.npy (T, 5023, 3)}
    <dataset_root>/TFHP_templates_v03.pkl                        {<person>_<seq>: (5023, 3) neutral mesh}

The clips of one <person>_<seq> are concatenated in name order and cut into windows of `window_frames`
frames: hop = window_frames - overlap_frames for train, non-overlapping windows for val/test (a trailing
remainder shorter than a window is dropped). Only the part of each clip covered by a window is read.
Each item is (audio input_values, vertices (T, 3V), template (3V), one-hot, name), as in loader.py.
"""
import os
import pickle

import numpy as np
import soundfile as sf
import torch
from torch.utils import data
from transformers import Wav2Vec2Processor


class TFHPDataset(data.Dataset):
    def __init__(
        self,
        data_list,
        subjects_dict,
        templates,
        split_name="train",
        use_head_pose=False,
        wav2vec_model_name="facebook/wav2vec2-base-960h",
        sr=16000,
        coef_fps=25,
    ):
        self.data = data_list
        self.len = len(self.data)

        self.subjects_dict = subjects_dict
        self.templates = templates
        self.split_name = split_name
        self.use_head_pose = use_head_pose

        self.sr = sr
        self.coef_fps = coef_fps
        self.audio_unit = float(self.sr) / float(self.coef_fps)  # audio samples per mesh frame

        # one-hot over the training speakers
        self.train_subjects = subjects_dict["train"]
        self.one_hot_labels = np.eye(len(self.train_subjects), dtype=np.float32)

        self.processor = Wav2Vec2Processor.from_pretrained(wav2vec_model_name)

        # subject_id -> clip list (shared by all windows of that subject)
        self.subject2clips = {}
        for it in self.data:
            sid = it["subject_id"]
            if "clip_infos" in it:
                if sid not in self.subject2clips:
                    self.subject2clips[sid] = it["clip_infos"]

    def __len__(self):
        return self.len

    def _load_verts_slice(self, vert_path, f0, f1):
        """vert_path: .npy (T,V,3); returns frames [f0, f1) of the clip."""
        verts_mmap = np.load(vert_path, mmap_mode="r")
        verts = verts_mmap[f0:f1]
        return np.asarray(verts, dtype=np.float32)

    def _load_audio_slice(self, audio_path, f0, f1):
        """audio_path: wav; returns the samples of mesh frames [f0, f1) of the clip."""
        s0 = int(round(f0 * self.audio_unit))
        s1 = int(round(f1 * self.audio_unit))
        num = max(s1 - s0, 0)

        if num <= 0:
            return np.zeros((0,), dtype=np.float32)

        with sf.SoundFile(audio_path, "r") as f:
            if f.samplerate != self.sr:
                raise ValueError(f"Invalid sampling rate: {f.samplerate} (expected {self.sr}) in {audio_path}")
            f.seek(s0)
            audio = f.read(num, dtype="float32")

        if audio.ndim == 2:   # stereo -> mono
            audio = audio.mean(axis=1)

        return audio.astype(np.float32)

    def __getitem__(self, index):
        item = self.data[index]

        subject_id  = item["subject_id"]
        start_frame = item["start_frame"]
        end_frame   = item["end_frame"]
        file_name   = item["name"]

        if subject_id in self.subject2clips:
            clip_infos = self.subject2clips[subject_id]
        else:
            clip_infos = item["clip_infos"]

        # read only the clips that overlap the window
        audio_parts = []
        vert_parts = []

        for c in clip_infos:
            g0 = c["frame_start"]   # global frame range of this clip
            g1 = c["frame_end"]     # exclusive

            if end_frame <= g0 or start_frame >= g1:
                continue

            seg0 = max(start_frame, g0)
            seg1 = min(end_frame, g1)
            local0 = seg0 - g0
            local1 = seg1 - g0

            vert_parts.append(self._load_verts_slice(c["vert_path"], local0, local1))
            audio_parts.append(self._load_audio_slice(c["audio_path"], local0, local1))

        if len(vert_parts) == 0:
            raise RuntimeError(f"[EMPTY WINDOW] {subject_id} {start_frame}-{end_frame}")

        verts_win = np.concatenate(vert_parts, axis=0)     # (W,V,3)
        audio_win = np.concatenate(audio_parts, axis=0)    # (W*audio_unit,)

        if verts_win.shape[0] != (end_frame - start_frame):
            raise RuntimeError(
                f"[FRAME MISMATCH] {file_name}: got {verts_win.shape[0]} frames, "
                f"expected {end_frame-start_frame}"
            )

        T, V, C = verts_win.shape
        verts_flat = verts_win.reshape(T, V * C)

        input_values = np.squeeze(
            self.processor(audio_win, sampling_rate=self.sr).input_values
        )

        if subject_id not in self.templates:
            raise KeyError(f"subject_id {subject_id} not in templates")
        template = np.asarray(self.templates[subject_id], dtype=np.float32).reshape(-1)

        if self.split_name == "train":
            subj_idx = self.train_subjects.index(subject_id)
            one_hot = self.one_hot_labels[subj_idx]
        else:
            one_hot = self.one_hot_labels

        return (
            torch.FloatTensor(input_values),
            torch.FloatTensor(verts_flat),
            torch.FloatTensor(template),
            torch.FloatTensor(one_hot),
            file_name,
        )


def read_tfhp_windows(
    dataset_root,
    use_head_pose=False,
    train_txt=None,
    val_txt=None,
    test_txt=None,
    min_valid_len: int = 0,
    window_frames: int = 100,
    overlap_frames: int = 50,
):
    """Collect the clips of every split entry and build the window list (see the module docstring)."""
    print("Loading TFHP data (lazy, fixed-length windows)...")
    print(f"[TFHP] min_valid_len (T threshold) = {min_valid_len}")
    print(f"[TFHP] window_frames = {window_frames}, overlap_frames = {overlap_frames}")

    assert window_frames > 0
    assert 0 <= overlap_frames < window_frames

    if train_txt is None:
        train_txt = os.path.join(dataset_root, "train.txt")
    if val_txt is None:
        val_txt = os.path.join(dataset_root, "val.txt")
    if test_txt is None:
        test_txt = os.path.join(dataset_root, "test.txt")

    clip_root = os.path.join(dataset_root, "TFHP_w_Head_v03" if use_head_pose else "TFHP_wo_Head_v03")
    template_pkl_path = os.path.join(dataset_root, "TFHP_templates_v03.pkl")
    assert os.path.exists(template_pkl_path), f"Template pkl not found: {template_pkl_path}"

    with open(template_pkl_path, "rb") as f:
        templates = pickle.load(f, encoding="latin1")

    def _collect_clips_for_subject(subject_id):
        """subject_id = '<person>_<seq>' -> clip infos with cumulative frame offsets."""
        prefix = subject_id + "_"
        folders = sorted(x for x in os.listdir(clip_root) if x.startswith(prefix))

        clip_infos = []
        total_frames = 0

        for clip_folder in folders:
            clip_dir = os.path.join(clip_root, clip_folder)
            audio_path = os.path.join(clip_dir, "audio.wav")
            vert_path  = os.path.join(clip_dir, "w_vert.npy" if use_head_pose else "wo_vert.npy")

            if not (os.path.exists(audio_path) and os.path.exists(vert_path)):
                continue

            try:
                verts_hdr = np.load(vert_path, mmap_mode="r")
                if verts_hdr.ndim != 3 or verts_hdr.shape[-1] != 3:
                    continue
                T = int(verts_hdr.shape[0])
            except Exception:
                continue

            if min_valid_len > 0 and T <= min_valid_len:
                continue

            clip_infos.append({
                "clip_dir": clip_dir,
                "audio_path": audio_path,
                "vert_path": vert_path,
                "T": T,
                "frame_start": total_frames,
                "frame_end": total_frames + T,
            })
            total_frames += T

        return clip_infos, total_frames

    def load_split_from_txt(txt_path, split_tag="train"):
        assert os.path.exists(txt_path), f"Split file not found: {txt_path}"

        data_list = []
        subject_ids = []

        with open(txt_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        # overlapping windows for train, non-overlapping for val/test
        hop_local = window_frames - overlap_frames if split_tag == "train" else window_frames

        for line in lines:
            entry = line.split()[0]
            parts = entry.split("/")
            if len(parts) != 2:
                continue

            person_id, seq_id = parts[0], parts[1]
            subject_id = f"{person_id}_{seq_id}"

            clip_infos, total_frames = _collect_clips_for_subject(subject_id)
            if len(clip_infos) == 0:
                continue

            if total_frames < window_frames:
                continue

            win_idx = 0
            for start in range(0, total_frames - window_frames + 1, hop_local):
                end = start + window_frames

                # train: <subject>_train_win<i>_<start>_<end> ; val/test: <subject>_<i>  (prediction file name)
                if split_tag == "train":
                    sample_name = f"{subject_id}_{split_tag}_win{win_idx:05d}_{start:06d}_{end:06d}"
                else:
                    sample_name = f"{subject_id}_{win_idx:03d}"

                data_list.append({
                    "subject_id": subject_id,
                    "clip_infos": clip_infos,
                    "start_frame": start,
                    "end_frame": end,
                    "name": sample_name,
                })
                subject_ids.append(subject_id)
                win_idx += 1

        split_name = os.path.basename(txt_path)
        print(f"[{split_name}] #window_samples = {len(data_list)} (hop={hop_local})")
        return data_list, subject_ids

    train_data, train_subjects = load_split_from_txt(train_txt, split_tag="train")
    val_data,   val_subjects   = load_split_from_txt(val_txt,   split_tag="val")
    test_data,  test_subjects  = load_split_from_txt(test_txt,  split_tag="test")

    subjects_dict = {
        "train": sorted(list(set(train_subjects))),
        "val":   sorted(list(set(val_subjects))),
        "test":  sorted(list(set(test_subjects))),
    }

    print(f"[TFHP] #train_windows = {len(train_data)}")
    print(f"[TFHP] #val_windows   = {len(val_data)}")
    print(f"[TFHP] #test_windows  = {len(test_data)}")
    print(f"[TFHP] #train_subjects = {len(subjects_dict['train'])}")
    print(f"[TFHP] clip_root = {clip_root}")
    print(f"[TFHP] template_pkl = {template_pkl_path}")

    return train_data, val_data, test_data, subjects_dict, templates


def get_dataloaders(args):
    use_head_pose = getattr(args, "tfhp_use_headpose", False)
    coef_fps = int(getattr(args, "tfhp_coef_fps", 25))

    train_meta, val_meta, test_meta, subjects_dict, templates = read_tfhp_windows(
        dataset_root=args.dataset_root,
        use_head_pose=use_head_pose,
        train_txt=getattr(args, "tfhp_train_txt", None),
        val_txt=getattr(args, "tfhp_val_txt", None),
        test_txt=getattr(args, "tfhp_test_txt", None),
        min_valid_len=int(getattr(args, "tfhp_min_len", 0)),
        window_frames=int(getattr(args, "tfhp_window_frames", 100)),
        overlap_frames=int(getattr(args, "tfhp_overlap_frames", 50)),
    )

    def _dataset(meta, split):
        return TFHPDataset(data_list=meta, subjects_dict=subjects_dict, templates=templates,
                           split_name=split, use_head_pose=use_head_pose, coef_fps=coef_fps)

    train_set, val_set, test_set = _dataset(train_meta, "train"), _dataset(val_meta, "val"), _dataset(test_meta, "test")

    _nw = getattr(args, "num_workers", 4)
    _train_kwargs = dict(
        batch_size=getattr(args, "batch_size", 1),
        shuffle=True,
        num_workers=_nw,
        pin_memory=True,
    )
    if _nw > 0:
        _train_kwargs.update(persistent_workers=True, prefetch_factor=2)
    return {
        "train": data.DataLoader(train_set, **_train_kwargs),
        "valid": data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0),
        "test":  data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0),
    }
