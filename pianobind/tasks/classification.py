import os
import numpy as np
from sklearn import metrics
import json 
import torch
import random
from torch.utils.data import DataLoader
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer

from pianobind.tasks.retrieval import Retrieval
from pianobind.datasets.tagging import TestDataset
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score


TAGNAMES = ['Emotional','Relaxing/Calm', 'Bright', 'Happy', 'Upbeat/Energetic', 'Cute', 'Playful', 'Dreamy',
        'Mysterious', 'Sad', 'Dark', 'Tense', 'Epic', 'Intense/Grand',
        'Passionate', 'Powerful', 'Difficult/Advanced', 
        'Easy', 'Speedy', 'Laid-back', 'Jazz',
        'New-age', 'Pop-Piano Cover', 'Classical', 
        'Swing', 'Funk', 'Latin', 'Blues', 'Ragtime','Ballad', 'Bossa Nova']

tags = [tag.lower() for tag in TAGNAMES]

def _l2(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

@torch.no_grad()
def encode_labels(label_str, all_tags, device=None):
    labels = [s.strip() for s in label_str.lower().split(",") if s.strip()]
    vec = torch.tensor([1 if tag in labels else 0 for tag in all_tags], dtype=torch.float)
    return vec.to(device or "cpu")

def prepare_labels(labels, prompt=None, model_name=None, use_dropout=False, device=None, max_length=77):
    valid = ["audio_text","midi_text","trimodal","PianoBind_AudioText","PianoBind_MIDIText","PianoBind_Trimodal"]
    if model_name not in valid:
        raise ValueError(f"Unsupported model_name: {model_name}")
    tok = RobertaTokenizer.from_pretrained("roberta-base")

    texts = []
    for s in labels:
        tags_ = [t.strip() for t in s.split(",") if t.strip()]
        if use_dropout:
            import random, numpy as np
            k = np.random.randint(1, len(tags_)+1)
            tags_ = random.sample(tags_, k)
        text = ", ".join(tags_) if prompt is None else f"A {', '.join(tags_)} track"
        texts.append(text)

    out = tok(texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    return out["input_ids"].to(device or "cpu"), out["attention_mask"].to(device or "cpu")


def get_metrics(predictions, ground_truth, tags):
    predictions = predictions.detach().cpu().numpy()
    ground_truth = ground_truth.detach().cpu().numpy()

    results = {}
    tag_results = {}
    
    roc_auc = roc_auc_score(ground_truth, predictions, average=None)
    pr_auc = average_precision_score(ground_truth, predictions, average=None)
    for i, tag in enumerate(tags):
        tag_results[tag] = {
            'ROC-AUC': roc_auc[i],
            'PR-AUC': pr_auc[i]
        }
        
    results["ROC-AUC-macro"] = np.mean(roc_auc)
    results["PR-AUC-macro"] = np.mean(pr_auc)    
    results['tagwise'] = tag_results

    
    return results

@torch.no_grad()
def compute_muscall_similarity_score_trimodal(model, data_loader, text_ids, text_mask, device, model_name):
    N = len(data_loader.dataset)
    A = torch.zeros(N, 512, device=device); M = torch.zeros(N, 512, device=device)
    G = torch.zeros(N, len(tags), device=device)
    T = _l2(model.encode_text(text_ids, text_mask))  # [T,512]

    off = 0
    for batch in data_loader:
        in_a, in_m, labels_tuple = tuple(t.to(device) if torch.is_tensor(t) else t for t in batch)
        b = in_a.size(0)
        y = encode_labels(labels_tuple[0], tags, device=device)
        a = _l2(model.encode_audio(in_a)); m = _l2(model.encode_midi(in_m))
        A[off:off+b] = a; M[off:off+b] = m; G[off:off+b] = y; off += b

    logits_audio = torch.sigmoid(model.logit_scale.exp() * (A @ T.t()))
    logits_midi  = torch.sigmoid(model.logit_scale.exp() * (M @ T.t()))
    logits_avg   = torch.sigmoid(model.logit_scale.exp() * (_l2(0.5*(A+M)) @ T.t())) 
    return logits_audio, logits_midi, logits_avg, G



@torch.no_grad()
def compute_muscall_similarity_score_midi(model, data_loader, text_ids, text_mask, device, model_name):
    N = len(data_loader.dataset)
    M = torch.zeros(N, 512, device=device); G = torch.zeros(N, len(tags), device=device)
    T = _l2(model.encode_text(text_ids, text_mask))
    off = 0
    for batch in data_loader:
        _, in_m, labels_tuple = tuple(t.to(device) if torch.is_tensor(t) else t for t in batch)
        b = in_m.size(0)
        y = encode_labels(labels_tuple[0], tags, device=device)
        m = _l2(model.encode_midi(in_m))
        M[off:off+b] = m; G[off:off+b] = y; off += b
    logits_midi = torch.sigmoid(model.logit_scale.exp() * (M @ T.t()))
    return logits_midi, G

@torch.no_grad()
def compute_muscall_similarity_score(model, data_loader, text_ids, text_mask, device, model_name):
    N = len(data_loader.dataset)
    A = torch.zeros(N, 512, device=device); G = torch.zeros(N, len(tags), device=device)
    T = _l2(model.encode_text(text_ids, text_mask))
    off = 0
    for batch in data_loader:
        in_a, _, labels_tuple = tuple(t.to(device) if torch.is_tensor(t) else t for t in batch)
        b = in_a.size(0)
        y = encode_labels(labels_tuple[0], tags, device=device)
        a = _l2(model.encode_audio(in_a))
        A[off:off+b] = a; G[off:off+b] = y; off += b
    logits_audio = torch.sigmoid(model.logit_scale.exp() * (A @ T.t()))
    return logits_audio, G



def random_predictions(ground_truth, seed=None):
    np.random.seed(seed)
    random_preds = np.random.rand(*ground_truth.shape)
    return torch.tensor(random_preds, dtype=torch.float)

def evaluate_random_predictions(ground_truth, tags):
    random_preds = random_predictions(ground_truth)
    metrics_random = get_metrics(random_preds, ground_truth, tags)
    return metrics_random

def get_top5_tags(score_matrix, tags):
    top5_scores, top5_indices = torch.topk(score_matrix, 5, dim=1)  
    top5_tags_with_scores = [
        {tags[idx]: float(score) for idx, score in zip(indices, scores)}
        for indices, scores in zip(top5_indices.cpu().numpy(), top5_scores.cpu().numpy())
    ]
    return top5_tags_with_scores

class Tagging(Retrieval):
    def __init__(self, pretrain_config, json_path, npy_dir):
        self.json_path = json_path
        self.npy_dir = npy_dir

        super().__init__(pretrain_config)

    def load_dataset(self):
        # self.train_dataset = TestDataset(self.json_path, self.npy_dir)
        self.test_dataset = TestDataset(self.muscall_config.dataset_config, self.json_path, self.npy_dir)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=1)

    def evaluate(self):
        text_prompts = prepare_labels(tags, model_name=self.muscall_config.model_config.model_name)
        text_prompts_dropout = prepare_labels(tags, model_name=self.muscall_config.model_config.model_name, use_dropout=True)

        with torch.no_grad():
            name = self.muscall_config.model_config.model_name
            text_ids, attn = prepare_labels(tags, model_name=name, device=self.device)
            text_ids_do, attn_do = prepare_labels(tags, model_name=name, device=self.device, use_dropout=True)

            if name in ["trimodal","PianoBind_Trimodal"]:
                score_matrix_audio, score_matrix_midi, score_matrix_avg, ground_truth = compute_muscall_similarity_score_trimodal(
                        self.model, self.test_loader, text_ids, attn, self.device, name
                    )
                top5_tags_audio = get_top5_tags(score_matrix_audio, tags)
                top5_tags_midi = get_top5_tags(score_matrix_midi, tags)
                top5_tags_avg = get_top5_tags(score_matrix_avg, tags)
                
                actual_results = {}
                for i, (top_tags_audio, top_tags_midi, top_tags_avg, actual_labels) in enumerate(
                    zip(top5_tags_audio, top5_tags_midi, top5_tags_avg, ground_truth)
                ):
                    actual_tags = [tags[j] for j in range(len(tags)) if actual_labels[j] == 1]
                    actual_results[i] = {
                        'actual_tags': actual_tags,
                        'top5_predicted_tags_audio': top_tags_audio,
                        'top5_predicted_tags_midi': top_tags_midi,
                        'top5_predicted_tags_avg': top_tags_avg,
                        'tag_scores_audio': {tags[j]: float(score_matrix_audio[i][j]) for j in range(len(tags))},
                        'tag_scores_midi': {tags[j]: float(score_matrix_midi[i][j]) for j in range(len(tags))},
                        'tag_scores_avg': {tags[j]: float(score_matrix_avg[i][j]) for j in range(len(tags))},
                    }
                
                model_metrics_audio = get_metrics(score_matrix_audio.cpu(), ground_truth.cpu(), tags)
                model_metrics_midi = get_metrics(score_matrix_midi.cpu(), ground_truth.cpu(), tags)
                model_metrics_avg = get_metrics(score_matrix_avg.cpu(), ground_truth.cpu(), tags)
                
                results = {
                    "Model Metrics - Audio": model_metrics_audio,
                    "Model Metrics - Midi": model_metrics_midi,
                    "Model Metrics - Average": model_metrics_avg,
                    "Actual Results": actual_results
                }
                
            elif name in ["midi_text","PianoBind_MIDIText"]:
                score_matrix, ground_truth = compute_muscall_similarity_score_midi(
                    self.model, self.test_loader, text_ids, attn, self.device, name
                )
                top5_tags = get_top5_tags(score_matrix, tags)
                actual_results = {}
                for i, (top_tags, actual_labels) in enumerate(zip(top5_tags, ground_truth)):
                    actual_tags = [tags[j] for j in range(len(tags)) if actual_labels[j] == 1]
                    actual_results[i] = {
                        'actual_tags': actual_tags,
                        'top5_predicted_tags': top_tags,
                        'tag_scores': {tags[j]: float(score_matrix[i][j]) for j in range(len(tags))}
                    }
                model_metrics = get_metrics(score_matrix.cpu(), ground_truth.cpu(), tags)
                results = {
                    "Model Metrics": model_metrics,
                    "Actual Results": actual_results
                }

            
            #Audio-Text
            elif name in ["audio_text","PianoBind_AudioText"]:
                score_matrix, ground_truth = compute_muscall_similarity_score(
                    self.model, self.test_loader, text_ids, attn, self.device, name
                )
                top5_tags = get_top5_tags(score_matrix, tags)
                actual_results = {}
                for i, (top_tags, actual_labels) in enumerate(zip(top5_tags, ground_truth)):
                    actual_tags = [tags[j] for j in range(len(tags)) if actual_labels[j] == 1]
                    #print(f"Sample {i}: \n Actual Tags: {actual_tags}, \n Top-5 Predicted Tags: {top_tags}")
                    actual_results[i] = {
                        'actual_tags': actual_tags,
                        'top5_predicted_tags': top_tags
                    }                
                model_metrics = get_metrics(score_matrix.cpu(), ground_truth.cpu(), tags)
                random_metrics = evaluate_random_predictions(ground_truth.cpu(), tags)
                
                score_matrix_dropout, _ = compute_muscall_similarity_score(
                    self.model, self.test_loader, text_ids_do, attn_do, self.device, name
                )
                model_metrics_dropout = get_metrics(score_matrix_dropout.cpu(), ground_truth.cpu(), tags)
            
                results = {
                    "Model Metrics": model_metrics,
                    "Dropout Model Metrics": model_metrics_dropout,
                    "Random Prediction Metrics": random_metrics,
                    "Actual Results": actual_results
                }
        

        with open(f'./save/experiments/{self.muscall_config.env.experiment_id}/tagging_results.json', 'w') as file:
            json.dump(results, file, indent=4)