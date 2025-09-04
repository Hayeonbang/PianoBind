import math
import numpy as np
import random

import torch
import torch.nn as nn
from transformers import BertModel
import yaml
import argparse
import pickle
import copy
from transformers import BertConfig

class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

def get_MidiBERT_configs(config_file='./MidiBERT_utils/yamls/get_pretrained_embedding_config.yaml'):
    # Load the yaml configuration file

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
        
    args = Config(config)

    return args

class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super().__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# BERT model: similar approach to "felix"
class MidiBert(nn.Module):
    def __init__(self, bertConfig, e2w, w2e):
        super().__init__()
        
        self.bert = BertModel(bertConfig)
        bertConfig.d_model = bertConfig.hidden_size
        self.hidden_size = bertConfig.hidden_size
        self.bertConfig = bertConfig

        # token types: [Bar, Position, Pitch, Duration]
        self.n_tokens = []      # [3,18,88,66]
        self.classes = ['Bar', 'Position', 'Pitch', 'Duration']
        for key in self.classes:
            self.n_tokens.append(len(e2w[key]))
        self.emb_sizes = [256, 256, 256, 256]
        self.e2w = e2w
        self.w2e = w2e

        # for deciding whether the current input_ids is a <PAD> token
        self.bar_pad_word = self.e2w['Bar']['Bar <PAD>']        
        self.mask_word_np = np.array([self.e2w[etype]['%s <MASK>' % etype] for etype in self.classes], dtype=np.longlong)
        self.pad_word_np = np.array([self.e2w[etype]['%s <PAD>' % etype] for etype in self.classes], dtype=np.longlong)
        
        # word_emb: embeddings to change token ids into embeddings
        self.word_emb = []
        for i, key in enumerate(self.classes):
            self.word_emb.append(Embeddings(self.n_tokens[i], self.emb_sizes[i]))
        self.word_emb = nn.ModuleList(self.word_emb)

        # linear layer to merge embeddings from different token types 
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), bertConfig.d_model)


    def forward(self, input_ids, attn_mask=None, output_hidden_states=True):
        # convert input_ids into embeddings and merge them through linear layer
        embs = []
        for i, key in enumerate(self.classes):
            embs.append(self.word_emb[i](input_ids[..., i]))
        embs = torch.cat([*embs], dim=-1)
        emb_linear = self.in_linear(embs)

        # feed to bert 
        y = self.bert(inputs_embeds=emb_linear, attention_mask=attn_mask, output_hidden_states=output_hidden_states)
        #y = y.last_hidden_state         # (batch_size, seq_len, 768)
        return y
    
    def get_rand_tok(self):
        c1,c2,c3,c4 = self.n_tokens[0], self.n_tokens[1], self.n_tokens[2], self.n_tokens[3]
        return np.array([random.choice(range(c1)),random.choice(range(c2)),random.choice(range(c3)),random.choice(range(c4))])


class MidiBertLM(nn.Module):
    def __init__(self, midibert: MidiBert):
        super().__init__()
        
        self.midibert = midibert
        self.mask_lm = MLM(self.midibert.e2w, self.midibert.n_tokens, self.midibert.hidden_size)

    def forward(self, x, attn, get_embedding=False):
        x = self.midibert(x, attn)
        # Last hidden states
        if get_embedding:
            x_last = x.last_hidden_state
            return self.mask_lm(x), x_last
        return self.mask_lm(x)

class MidiBERTLM_encoder(nn.Module):
    def __init__(self, midibert: MidiBert, max_seq_len, get_embedding=True, device='cuda:0'):
        super().__init__()
        self.device = device
        self.midibert = midibert.to(self.device)        # save this for ckpt
        self.model = MidiBertLM(midibert).to(self.device)
        self.max_seq_len = max_seq_len
        self.get_embedding = get_embedding
    
    def forward(self, data):
        batch = data.shape[0]
        data = data.to(self.device)  # (batch, seq_len, 4)
        input_ids = copy.deepcopy(data)
        var_len = min(data.shape[1], self.max_seq_len)
        loss_mask = torch.zeros(batch, var_len) # for variable length

        loss_mask = loss_mask.to(self.device)      

        # avoid attend to pad word
        attn_mask = (input_ids[:, :, 0] != self.midibert.bar_pad_word).float().to(self.device)   # (batch, seq_len)
        
        y, output = self.model.forward(input_ids, attn_mask, self.get_embedding)
        # last_embedding = output[:,0,:] # sos token
        last_embedding = output.mean(dim=1)
        return last_embedding
            

class MLM(nn.Module):
    def __init__(self, e2w, n_tokens, hidden_size):
        super().__init__()
        
        # proj: project embeddings to logits for prediction
        self.proj = []
        for i, etype in enumerate(e2w):
            self.proj.append(nn.Linear(hidden_size, n_tokens[i]))
        self.proj = nn.ModuleList(self.proj)

        self.e2w = e2w
    
    def forward(self, y):
        # feed to bert 
        y = y.hidden_states[-1]
        
        # convert embeddings back to logits for prediction
        ys = []
        for i, etype in enumerate(self.e2w):
            ys.append(self.proj[i](y))           # (batch_size, seq_len, dict_size)
        return ys
    
def load_MidiBERT_backbone(args):
    print("Loading Dictionary")
    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)
    
    configuration = BertConfig(max_position_embeddings=args.max_seq_len,
                            position_embedding_type='relative_key_query',
                            hidden_size=args.hs) 
    midi_bert = MidiBert(bertConfig=configuration, e2w=e2w, w2e=w2e)
    if args.ckpt is not None:
        best_mdl = args.ckpt
        print("   Loading pre-trained model from", best_mdl.split('/')[-1])
        checkpoint = torch.load(best_mdl, map_location='cpu')
        midi_bert.load_state_dict(checkpoint['state_dict'], strict=False)
    
    midibert_encoder = MidiBERTLM_encoder(midibert=midi_bert, max_seq_len=args.max_seq_len, get_embedding=True)
    return midibert_encoder
