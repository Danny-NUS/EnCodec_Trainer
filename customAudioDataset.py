import os
import pandas as pd
import torch
import torchaudio
import random
import json
import torchaudio.transforms as T



class CustomAudioDataset(torch.utils.data.Dataset):
    def __init__(self, meta_path, transform=None, tensor_cut=0, fixed_length=None):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
        self.meta_data = meta_data
        self.transform = transform
        self.fixed_length = fixed_length
        self.tensor_cut = tensor_cut

    def __len__(self):
        if self.fixed_length:
            return self.fixed_length
        return len(self.meta_data)

    def __getitem__(self, idx):
        audio_path = self.meta_data[idx]
        waveform, sample_rate = torchaudio.load(audio_path)

        if self.transform:
            waveform = self.transform(waveform)

        # All resample to 24000Hz
        self.transform = T.Resample(orig_freq=sample_rate, new_freq=24000)
        if sample_rate != 24000:
            waveform = self.transform(waveform)

        if self.tensor_cut > 0:
            if waveform.size()[1] > self.tensor_cut:
                start = random.randint(0, waveform.size()[1]-self.tensor_cut-1)
                waveform = waveform[:, start:start+self.tensor_cut]
        return waveform, sample_rate

