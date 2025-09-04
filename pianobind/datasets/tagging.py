import os
import numpy as np
import pickle
import yaml
from pathlib import Path
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

import json

from pianobind.preprocess.midi.utils import find_closest_bar
from pianobind.preprocess.midi.model import CP

class TestDataset(Dataset):
    """Dataset for classification task: PIAST-AT with tag-format text"""
    
    def __init__(self, config, json_path, npy_dir, subset="test"):

        assert subset in ["train", "val", "test"], "Subset must be one of 'train', 'val', or 'test'."

        self.config = config
        self.json_path = json_path
        with open(json_path, 'r') as file:
            self.sample = json.load(file)
        self.npy_dir = npy_dir
        self.audio_paths = [str(Path(self.npy_dir) / item['audio_path']) for item in self.sample]

        self.subset = subset
            
        self.sample_rate = 16000
        self.num_samples = self.sample_rate * 20 
        self.random_crop = True if subset == "train" else False
        self._dataset_type = subset
        if self.config.midi.use_midi:
            self._setup_midi_data()
            
    def _setup_midi_data(self):
        self.midi_paths = []
        self.midi_dir = os.path.join(self.config.midi.dir, "token_cp", "pkls")
        valid_indices = []
        for idx, i in enumerate(self.sample):
            midi_path = os.path.join(self.midi_dir, i["audio_path"].replace(".npy", ".pkl") )
            if os.path.exists(midi_path):    
                valid_indices.append(idx)
                self.midi_paths.append(midi_path)
                        
        self.audio_paths = [self.audio_paths[idx] for idx in valid_indices]
        self.sample = [self.sample[idx] for idx in valid_indices]
        with open(os.path.join(self.config.midi.dir, "token_cp", "midi_segment_info.json")) as f2:
            midi_segment_info = json.load(f2)
        
        segment_keys = [os.path.join(self.config.midi.dir, 'midi', midi_path.split('/')[-1].replace(".pkl", ".mid")) 
                        for midi_path in self.midi_paths]
        self.midi_bar_start_times = [ midi_segment_info[segment_key]["bar_start_times_sec"] for segment_key in segment_keys]
        
        with open(os.path.join(self.config.midi.dir, "token_cp", "args.yaml"), 'r') as yaml_file:
            args = yaml.safe_load(yaml_file)
            
        self.cp = CP(dict=args['dict'])
        
    def __len__(self):
        return len(self.sample)
    
    def _crop_audio(self, mmapped_array):
        if np.shape(mmapped_array)[0] == 2:
            audio_length = np.shape(mmapped_array)[1]
        else:
            audio_length = np.shape(mmapped_array)[0]

        if audio_length <= self.num_samples:
            start_index = 0
            end_index = None
        else:
            if self._dataset_type == "train" and self.random_crop:
                start_index = np.random.randint(0, audio_length - self.num_samples)
            else:
                start_index = (audio_length - self.num_samples) // 2
            end_index = start_index + self.num_samples

        if np.shape(mmapped_array)[0] == 2:
            audio = (
                mmapped_array[:, start_index:end_index].astype("float32").mean(axis=0)
            )
        else:
            audio = mmapped_array[start_index:end_index].astype("float32")
        return audio, start_index
    
    def get_audio(self, idx):
        try:
            mmapped_array = np.load(self.audio_paths[idx], mmap_mode="r")
        except:
            mmapped_array = np.load(self.audio_paths[idx], mmap_mode="r+")
        
        cropped_audio, start_index = self._crop_audio(mmapped_array)
        audio = torch.tensor(cropped_audio, dtype=torch.float)

        if len(audio) < self.num_samples:
            zeros_needed = torch.zeros(self.num_samples - len(audio))
            audio = torch.cat((audio, zeros_needed), dim=0)

        return audio, start_index
    
    def get_midi(self, idx, start_index=None):
        if start_index == None:
            NotImplementedError('not implemented yet.')
        else:
            midi_data = pickle.load(open(self.midi_paths[idx], 'rb'))
            tokens = midi_data['words']
            bar_start_token_idxs = midi_data['bar_start_token_idxs']
            start_sec = start_index / self.config.audio.sr
            bar_start_times_sec = self.midi_bar_start_times[idx]
            start_bar, _ = find_closest_bar(bar_start_times_sec, start_sec)
            end_bar, _ = find_closest_bar(bar_start_times_sec, (start_index + self.num_samples) / self.config.audio.sr)
    
            end_bar = end_bar + 1 if end_bar + 1 < len(bar_start_token_idxs) else end_bar
            start_token_idxs, end_token_idxs = bar_start_token_idxs[start_bar], bar_start_token_idxs[end_bar]
            
            cropped_tokens = torch.tensor(tokens[start_token_idxs:end_token_idxs], dtype=torch.long)
            if len(cropped_tokens) < self.config.midi.tgt_len:
                pad_needed = torch.tensor(
                    [self.cp.pad_word] * (self.config.midi.tgt_len - len(cropped_tokens)),
                    dtype=torch.long
                )
                cropped_tokens = torch.cat((cropped_tokens, pad_needed), dim=0)
            elif len(cropped_tokens) > self.config.midi.tgt_len:
                cropped_tokens = cropped_tokens[:self.config.midi.tgt_len]

            for i, etype in enumerate(self.cp.word2event):
                assert torch.max(cropped_tokens[:, i]).item() <= max(self.cp.word2event[etype].keys())
            
            
        return cropped_tokens  

    def __getitem__(self, idx):
        item = self.sample[idx]
        input_audio, start_index = self.get_audio(idx)
        caption = item['caption']
        
        if self.config.midi.use_midi:
            input_midi = self.get_midi(idx, start_index)
        else:
            input_midi = torch.tensor([])
            
        data_int16 = np.int16(input_audio.cpu().numpy() * 32767)  
        if input_audio is None or input_midi is None:
            print(f"Index {idx} has NoneType in audio or midi.")
    
        # debug
        # from scipy.io import wavfile
        # debug_dir = 'debug_input_eval'
        # os.makedirs(debug_dir, exist_ok=True)
        # wavfile.write(f'{debug_dir}/output_{idx}.wav', self.sample_rate, data_int16)

        # decoded_event_str = self.cp.token_to_event_str(input_midi)
        # event_seq = self.cp.event_str_to_events(decoded_event_str)
        # self.cp.events2midi(event_seq, f'{debug_dir}/output_{idx}.mid')

        
        return input_audio, input_midi, caption