import torch

# -----------------------------
# Feature Extraction
# -----------------------------

@torch.no_grad()
def get_muscall_features_audio_midi_text(model, data_loader, device, model_name):
    dataset_size = len(data_loader.dataset)
    audio_all = torch.zeros(dataset_size, 512).to(device)
    midi_all = torch.zeros(dataset_size, 512).to(device)
    text_all = torch.zeros(dataset_size, 512).to(device)

    count = 0
    for batch in data_loader:
        batch = tuple(t.to(device) for t in batch)
        _, input_audio, input_midi, text_ids, _, attn_mask, _ = batch

        audio = model.encode_audio(input_audio)
        midi = model.encode_midi(input_midi)
        text = model.encode_text(text_ids, attn_mask)

        audio = audio / audio.norm(dim=-1, keepdim=True)
        midi = midi / midi.norm(dim=-1, keepdim=True)
        text = text / text.norm(dim=-1, keepdim=True)

        bsz = input_audio.size(0)
        audio_all[count:count+bsz] = audio
        midi_all[count:count+bsz] = midi
        text_all[count:count+bsz] = text
        count += bsz

    return audio_all, midi_all, text_all


@torch.no_grad()
def get_muscall_features_midi(model, data_loader, device, model_name):
    dataset_size = len(data_loader.dataset)
    midi_all = torch.zeros(dataset_size, 512).to(device)
    text_all = torch.zeros(dataset_size, 512).to(device)

    count = 0
    for batch in data_loader:
        batch = tuple(t.to(device) for t in batch)
        _, _, input_midi, text_ids, _, attn_mask, _ = batch

        midi = model.encode_midi(input_midi)
        text = model.encode_text(text_ids, attn_mask)

        midi = midi / midi.norm(dim=-1, keepdim=True)
        text = text / text.norm(dim=-1, keepdim=True)

        bsz = input_midi.size(0)
        midi_all[count:count+bsz] = midi
        text_all[count:count+bsz] = text
        count += bsz

    return midi_all, text_all


@torch.no_grad()
def get_muscall_features(model, data_loader, device, model_name):
    dataset_size = len(data_loader.dataset)
    audio_all = torch.zeros(dataset_size, 512).to(device)
    text_all = torch.zeros(dataset_size, 512).to(device)

    count = 0
    for batch in data_loader:
        batch = tuple(t.to(device) for t in batch)
        _, input_audio, _, text_ids, _, attn_mask, _ = batch

        audio = model.encode_audio(input_audio)
        text = model.encode_text(text_ids, attn_mask)

        audio = audio / audio.norm(dim=-1, keepdim=True)
        text = text / text.norm(dim=-1, keepdim=True)

        bsz = input_audio.size(0)
        audio_all[count:count+bsz] = audio
        text_all[count:count+bsz] = text
        count += bsz

    return audio_all, text_all

# -----------------------------
# Similarity + Retrieval
# -----------------------------

def compute_sim_score(audio_features=None, midi_features=None, text_features=None):
    if audio_features is not None and text_features is not None:
        return (audio_features @ text_features.t()).t()
    elif midi_features is not None and text_features is not None:
        return (midi_features @ text_features.t()).t()
    else:
        raise ValueError("Either audio_features or midi_features must be provided.")

def get_ranking(score_matrix, device):
    sorted_scores, retrieved = torch.sort(score_matrix, dim=1, descending=True)
    num_queries = score_matrix.size(0)
    gt = torch.arange(num_queries).unsqueeze(1).expand(-1, score_matrix.size(1)).to(device)
    return retrieved, gt

def compute_metrics(retrieved_indices, gt_indices):
    num_queries = retrieved_indices.size(0)
    matches = (retrieved_indices == gt_indices)
    ranks = torch.where(matches)[1] + 1

    return {
        "R@1": 100 * matches[:, :1].sum().item() / num_queries,
        "R@5": 100 * matches[:, :5].sum().item() / num_queries,
        "R@10": 100 * matches[:, :10].sum().item() / num_queries,
        "Median Rank": ranks.median().item() if ranks.numel() > 0 else -1
    }

def compute_random_metrics(num_queries, num_items):
    random_indices = torch.randint(0, num_items, (num_queries, num_items))
    gt = torch.arange(num_queries).unsqueeze(1).expand(-1, num_items)
    return compute_metrics(random_indices, gt)

# -----------------------------
# Results Formatting
# -----------------------------

def print_top_results(retrieved, all_dataset_metadata, top_k=5): # Renamed `dataset` to `all_dataset_metadata` for clarity
    results = {}
    for query_idx in range(len(retrieved)):
        # Directly use the metadata from the JSON for the query
        query_info = all_dataset_metadata[query_idx] 

        query_caption = query_info['caption']
        query_audio_path = query_info['audio_path']
        results[query_caption] = {
            "audio_path": query_audio_path,
            "retrieved_tracks": []
        }

        top_k_indices = retrieved[query_idx][:top_k].tolist()
        for rank, idx in enumerate(top_k_indices, start=1):
            # Directly use the metadata from the JSON for the retrieved item
            top_info = all_dataset_metadata[idx] 
            top_caption = top_info['caption']
            top_audio_path = top_info['audio_path']
            results[query_caption]["retrieved_tracks"].append({
                "rank": rank,
                "caption": top_caption,
                "audio_path": top_audio_path
            })
    return results


def calculate_actual_ranks(retrieved_indices, gt_indices):
    ranks = {}
    for i, gt in enumerate(gt_indices):
        idx = (retrieved_indices[i] == gt[0]).nonzero(as_tuple=True)[0]
        ranks[i] = (idx.item() + 1) if idx.numel() > 0 else -1
    return ranks

def tensor_to_list(x):
    return x.tolist() if isinstance(x, torch.Tensor) else x

# -----------------------------
# High-level Retrieval Wrapper
# -----------------------------

@torch.no_grad()
def run_retrieval(model, data_loader, device, model_name, use_midi=False):
    if model_name == "trimodal":
        audio, midi, text = get_muscall_features_audio_midi_text(model, data_loader, device, model_name)

        score_audio = compute_sim_score(audio_features=audio, text_features=text)
        score_midi = compute_sim_score(midi_features=midi, text_features=text)

        ret_audio, gt_audio = get_ranking(score_audio, device)
        ret_midi, gt_midi = get_ranking(score_midi, device)

        metrics_audio = compute_metrics(ret_audio, gt_audio)
        metrics_midi = compute_metrics(ret_midi, gt_midi)

        return metrics_audio, metrics_midi

    elif model_name == "audio_text":
        audio, text = get_muscall_features(model, data_loader, device, model_name)
        score = compute_sim_score(audio_features=audio, text_features=text)

    elif model_name == "midi_text":
        midi, text = get_muscall_features_midi(model, data_loader, device, model_name)
        score = compute_sim_score(midi_features=midi, text_features=text)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    ret, gt = get_ranking(score, device)
    return compute_metrics(ret, gt)
