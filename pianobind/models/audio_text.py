import numpy as np

import torch
from torch import nn
from transformers import RobertaModel

from pianobind.utils.losses import clip_loss
from pianobind.modules.audio_backbones import ModifiedResNet



class PianoBind_AudioText(nn.Module):
    def __init__(self, config):
        super().__init__()
        audio_config = config.audio
        text_config = config.text

        projection_dim = config.projection_dim
        audio_dim = audio_config.hidden_size
        text_dim = text_config.hidden_size

        self.temperature = config.temperature
        self.audio_backbone = ModifiedResNet(audio_config)
            
        pretrained_model = config.text.pretrained
        self.textual_head = RobertaModel.from_pretrained(pretrained_model)
        
        self.audio_projection = nn.Linear(audio_dim, projection_dim, bias=False)
        self.text_projection = nn.Linear(text_dim, projection_dim, bias=False)

        if self.temperature is None:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_audio(self, audio):
        audio_features = self.audio_backbone(audio)
        audio_features = self.audio_projection(audio_features)
        return audio_features

    def encode_text(self, text, text_mask=None):
        outputs = self.textual_head(input_ids=text, attention_mask=text_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1) 
        text_features = self.text_projection(pooled_output)
        return text_features

    def forward(
        self,
        audio,
        text,
        text_mask=None,
        return_loss=True,
    ):
        audio_features = self.encode_audio(audio)
        text_features = self.encode_text(text, text_mask)

        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if self.temperature is None:
            logit_scale = self.logit_scale.exp()
        else:
            logit_scale = 1.0 / self.temperature
            
        logits_per_audio = logit_scale * audio_features @ text_features.t()
        logits_per_text = logits_per_audio.t()

        if return_loss:
            loss = clip_loss(logits_per_text)
            return loss
        else:
            return logits_per_audio, logits_per_text

    @classmethod
    def config_path(cls):
        return "configs/models/audio_text.yaml"