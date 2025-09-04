import os
import json
import pickle
import random
import yaml
import numpy as np
import torch
import torchaudio.functional as F
from torch.utils.data import Dataset
from scipy.io import wavfile
from transformers import RobertaTokenizer

from pianobind.preprocess.midi.utils import find_closest_bar
from pianobind.preprocess.midi.model import CP


class PIAST_YT_Dataset(Dataset):
    def __init__(self, config, tokenizer=None, dataset_type="train"):
        super().__init__()
        self.config = config or {}
        self._dataset_type = dataset_type
        self._dataset_name = "piast_yt"
        self.use_midi = config.model in ["midi_text", "trimodal"]

        self._set_config()
        self._set_paths()
        self.tokenizer = tokenizer or self._build_tokenizer()

        self._load_metadata()
        if self.use_midi:
            self._load_midi_metadata()

    def _set_config(self):
        self.max_seq_length = self.config.text.max_seq_length
        self.sample_rate = self.config.audio.sr
        self.num_samples = self.sample_rate * self.config.audio.crop_length
        self.random_crop = self.config.audio.random_crop

    def _set_paths(self):
        base_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "datasets", "piast_yt")
        self._data_dir = base_path
        if self._dataset_type == "test":
            self.dataset_json = os.path.join(base_path, "..", "test_dataset_tag.json")
        else:
            self.dataset_json = os.path.join(base_path, f"dataset_{self._dataset_type}.json")
        self.audio_dir = os.path.join(base_path, "audio")

    def _build_tokenizer(self):
        tk_type = self.config.text.tokenizer.lower()
        if tk_type == "robertatokenizer":
            return RobertaTokenizer.from_pretrained("roberta-base")
        else:
            raise ValueError(f"Unsupported tokenizer type: {self.config.text.tokenizer}")

    def _load_metadata(self):
        with open(self.dataset_json) as f:
            self.samples = json.load(f)
        self.audio_ids = [s["audio_id"] for s in self.samples]
        self.captions = [s["caption"].strip() for s in self.samples]
        self.audio_paths = [os.path.join(self.audio_dir, s["audio_path"]) for s in self.samples]

    def _load_midi_metadata(self):
        self.midi_dir = os.path.join(self.config.midi.dir, "token_cp", "pkls")
        segment_info_path = os.path.join(self.config.midi.dir, "token_cp", "midi_segment_info.json")
        args_path = os.path.join(self.config.midi.dir, "token_cp", "args.yaml")

        with open(segment_info_path) as f:
            midi_segment_info = json.load(f)
        with open(args_path, 'r') as f:
            args = yaml.safe_load(f)
        self.cp = CP(dict=args['dict'])

        self.midi_paths, self.midi_bar_start_times = [], []
        valid_indices = []

        for idx, s in enumerate(self.samples):
            pkl_path = os.path.join(self.midi_dir, s["audio_path"].replace(".npy", ".pkl"))
            if os.path.exists(pkl_path):
                valid_indices.append(idx)
                self.midi_paths.append(pkl_path)
                midi_key = os.path.join(self.config.midi.dir, 'midi', os.path.basename(pkl_path).replace(".pkl", ".mid"))
                self.midi_bar_start_times.append(midi_segment_info[midi_key]["bar_start_times_sec"])

        self.audio_ids = [self.audio_ids[i] for i in valid_indices]
        self.captions = [self.captions[i] for i in valid_indices]
        self.audio_paths = [self.audio_paths[i] for i in valid_indices]

    def get_raw_caption(self, idx):
        return self.captions[idx]

    def _remove_silence(self, waveform, sample_rate):
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        vad_audio = F.vad(waveform, sample_rate)
        return waveform if vad_audio.shape[1] == 0 else vad_audio.squeeze(0)

    def _crop_audio(self, mmapped_array):
        audio_length = mmapped_array.shape[1] if mmapped_array.ndim == 2 else mmapped_array.shape[0]
        if audio_length <= self.num_samples:
            start = 0
        else:
            start = np.random.randint(0, audio_length - self.num_samples) if self._dataset_type == "train" and self.random_crop else (audio_length - self.num_samples) // 2
        end = start + self.num_samples
        audio = mmapped_array[:, start:end].mean(axis=0) if mmapped_array.ndim == 2 else mmapped_array[start:end]
        return audio.astype("float32"), start

    def get_audio(self, idx):
        mmapped_array = np.load(self.audio_paths[idx], mmap_mode="r")
        cropped_audio, start_index = self._crop_audio(mmapped_array)
        audio = torch.tensor(cropped_audio.copy(), dtype=torch.float)
        audio = self._remove_silence(audio, self.sample_rate)
        if audio.numel() == 0:
            audio = torch.zeros(self.num_samples, dtype=torch.float)
        if audio.dim() == 2:
            audio = audio.squeeze(0)
        return torch.cat((audio, torch.zeros(self.num_samples - len(audio)))) if len(audio) < self.num_samples else audio[:self.num_samples], start_index

    def get_midi(self, idx, start_index):
        midi_data = pickle.load(open(self.midi_paths[idx], 'rb'))
        tokens = midi_data['words']
        bar_start_token_idxs = midi_data['bar_start_token_idxs']
        start_sec = start_index / self.sample_rate
        end_sec = (start_index + self.num_samples) / self.sample_rate
        start_bar, _ = find_closest_bar(self.midi_bar_start_times[idx], start_sec)
        end_bar, _ = find_closest_bar(self.midi_bar_start_times[idx], end_sec)
        end_bar = min(end_bar + 1, len(bar_start_token_idxs) - 1)
        cropped = torch.tensor(tokens[bar_start_token_idxs[start_bar]:bar_start_token_idxs[end_bar]], dtype=torch.long)
        tgt_len = self.config.midi.tgt_len
        pad = torch.tensor([self.cp.pad_word] * max(tgt_len - len(cropped), 0), dtype=torch.long)
        return torch.cat((cropped, pad))[:tgt_len].clone()

    def get_input_ids(self, idx):
        return self.tokenizer.encode(
            self.get_raw_caption(idx), max_length=self.max_seq_length, truncation=True
        )

    def get_text_input(self, idx):
        input_ids = self.get_input_ids(idx)
        pad_len = self.max_seq_length - len(input_ids)
        input_ids += [0] * pad_len
        attention_mask = [1] * (len(input_ids) - pad_len) + [0] * pad_len
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.zeros(self.max_seq_length, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )

    def __getitem__(self, idx):
        idx_tensor = torch.tensor(idx)
        audio_id = torch.tensor(self.audio_ids[idx], dtype=torch.long)
        audio, start_index = self.get_audio(idx)
        text_ids, type_ids, mask = self.get_text_input(idx)
        midi = self.get_midi(idx, start_index) if self.use_midi else torch.empty(0, dtype=torch.long)

        if self.config.debug_input:
            os.makedirs("debug_input", exist_ok=True)
            wavfile.write(f"debug_input/output_{idx}.wav", self.sample_rate, (audio.numpy() * 32767).astype(np.int16))
            decoded_event_str = self.cp.token_to_event_str(midi)
            event_seq = self.cp.event_str_to_events(decoded_event_str)
            self.cp.events2midi(event_seq, f"debug_input/output_{idx}.mid")

        return audio_id, audio.clone(), midi.clone(), text_ids.clone(), type_ids.clone(), mask.clone(), idx_tensor

    def __len__(self):
        return len(self.audio_paths)

    @classmethod
    def config_path(cls):
        return "configs/datasets/piast_yt.yaml"
