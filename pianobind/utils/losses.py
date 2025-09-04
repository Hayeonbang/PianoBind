import torch
import torch.nn.functional as F

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(len(logits), device=logits.device)
    return F.cross_entropy(logits, labels)

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    return (contrastive_loss(similarity) + contrastive_loss(similarity.T)) / 2.0
