import numpy as np

import torch
from torch import nn
from transformers import RobertaModel

from pianobind.utils.losses import clip_loss
from pianobind.modules.midi_backbones import load_MidiBERT_backbone


class PianoBind_MIDIText(nn.Module):
    def __init__(self, config):
        super().__init__()
        midi_config = config.midi
        text_config = config.text

        projection_dim = config.projection_dim
        midi_dim = midi_config.hidden_size
        text_dim = text_config.hidden_size

        self.temperature = config.temperature
        self.midi_backbone = load_MidiBERT_backbone(midi_config)
        
        pretrained_model = config.text.pretrained
        self.textual_head = RobertaModel.from_pretrained(pretrained_model)
        
        self.midi_projection = nn.Linear(midi_dim, projection_dim, bias=False)
        self.text_projection = nn.Linear(text_dim, projection_dim, bias=False)

        if self.temperature is None:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def encode_midi(self, midi):
        midi_features = self.midi_backbone(midi)
        midi_features = self.midi_projection(midi_features)
        return midi_features

    def encode_text(self, text, text_mask=None):
        outputs = self.textual_head(input_ids=text, attention_mask=text_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1) 
        text_features = self.text_projection(pooled_output)
        return text_features

    def forward(
        self,
        midi,
        text,
        text_mask=None,
        return_loss=True,
    ):
        midi_features = self.encode_midi(midi)
        text_features = self.encode_text(text, text_mask)

        midi_features = midi_features / midi_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if self.temperature is None:
            logit_scale = self.logit_scale.exp()
        else:
            logit_scale = 1.0 / self.temperature
        logits_per_midi = logit_scale * midi_features @ text_features.t()
        logits_per_text = logits_per_midi.t()

        if return_loss:
            loss = clip_loss(logits_per_text)
            return loss
        else:
            return logits_per_midi, logits_per_text

    @classmethod
    def config_path(cls):
        return "configs/models/midi_text.yaml"