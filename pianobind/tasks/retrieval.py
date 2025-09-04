import os
import json
import torch
from torch.utils.data import DataLoader, Subset

from pianobind.models.audio_text import PianoBind_AudioText
from pianobind.models.midi_text import PianoBind_MIDIText
from pianobind.models.trimodal import PianoBind_Trimodal

from pianobind.datasets.piast_at import PIAST_AT_Dataset
from pianobind.datasets.piast_yt import PIAST_YT_Dataset

from pianobind.utils.evaluation_utils import (get_muscall_features_audio_midi_text, get_muscall_features_midi, 
                                              get_muscall_features, compute_sim_score, get_ranking,
                                              compute_metrics, compute_random_metrics, print_top_results,
                                              calculate_actual_ranks, tensor_to_list,)


@torch.no_grad()
def get_muscall_features_audio_midi_text(model, data_loader, device, model_name):
    dataset_size = data_loader.dataset.__len__()
    
    all_audio_features = torch.zeros(dataset_size, 512).to(device)
    all_midi_features = torch.zeros(dataset_size, 512).to(device)
    all_text_features = torch.zeros(dataset_size, 512).to(device)

    samples_in_previous_batch = 0
    for i, batch in enumerate(data_loader):

        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
        _, input_audio, input_midi, text_input_ids, text_input_type_ids, text_attention_mask, _ = batch
    
        
        audio_features = model.encode_audio(input_audio)
        midi_features = model.encode_midi(input_midi)
        text_features = model.encode_text(text_input_ids, text_attention_mask)
        
        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
        midi_features = midi_features / midi_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        samples_in_current_batch = input_audio.size(0)
        start_index = i * samples_in_previous_batch
        end_index = start_index + samples_in_current_batch
        samples_in_previous_batch = samples_in_current_batch

        all_audio_features[start_index:end_index] = audio_features
        all_midi_features[start_index:end_index] = midi_features
        all_text_features[start_index:end_index] = text_features
        


    return all_audio_features, all_midi_features, all_text_features

@torch.no_grad()
def get_muscall_features_midi(model, data_loader, device, model_name):
    dataset_size = data_loader.dataset.__len__()

    all_midi_features = torch.zeros(dataset_size, 512).to(device)
    all_text_features = torch.zeros(dataset_size, 512).to(device)

    samples_in_previous_batch = 0
    for i, batch in enumerate(data_loader):
        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
        _, input_audio, input_midi, text_input_ids, text_input_type_ids, text_attention_mask, _ = batch

        midi_features = model.encode_midi(input_midi)
        

        if model_name == "audio_text":
            text_features = model.encode_text(text_input_ids, text_attention_mask)
        elif model_name == "midi_text":
            text_features = model.encode_text(text_input_ids, text_attention_mask)
        
        midi_features = midi_features / midi_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        samples_in_current_batch = input_midi.size(0)
        start_index = i * samples_in_previous_batch
        end_index = start_index + samples_in_current_batch
        samples_in_previous_batch = samples_in_current_batch

        all_midi_features[start_index:end_index] = midi_features
        all_text_features[start_index:end_index] = text_features

    return all_midi_features, all_text_features

@torch.no_grad()
def get_muscall_features(model, data_loader, device, model_name):
    dataset_size = data_loader.dataset.__len__()

    all_audio_features = torch.zeros(dataset_size, 512).to(device)
    all_text_features = torch.zeros(dataset_size, 512).to(device)

    samples_in_previous_batch = 0
    for i, batch in enumerate(data_loader):
        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
        _, input_audio, _, text_input_ids, text_input_type_ids, text_attention_mask, _ = batch

        audio_features = model.encode_audio(input_audio)
        
        text_features = model.encode_text(text_input_ids, text_attention_mask)

        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        samples_in_current_batch = input_audio.size(0)
        start_index = i * samples_in_previous_batch
        end_index = start_index + samples_in_current_batch
        samples_in_previous_batch = samples_in_current_batch

        all_audio_features[start_index:end_index] = audio_features
        all_text_features[start_index:end_index] = text_features

    return all_audio_features, all_text_features

def compute_sim_score(audio_features=None, midi_features=None, text_features=None):        
    # audio-text
    if audio_features is not None and text_features is not None and midi_features is None:
        logits_per_audio = audio_features @ text_features.t()
        logits_per_text = logits_per_audio.t()
    else:
        logits_per_audio = None

    # midi-text
    if midi_features is not None and text_features is not None and audio_features is None:
        logits_per_midi = midi_features @ text_features.t()
        logits_per_text = logits_per_midi.t()
    else:
        logits_per_midi = None

    return logits_per_text


# --- get_ranking: score_matrix -> retrieved_indices, gt_indices
def get_ranking(score_matrix: torch.Tensor, device=None):
    # score_matrix: [num_queries, num_items]
    device = device or score_matrix.device
    retrieved_indices = torch.argsort(score_matrix, dim=1, descending=True)
    num_queries, num_items = score_matrix.shape
    gt_indices = torch.arange(num_queries, device=device).unsqueeze(1).expand(num_queries, num_items)
    return retrieved_indices, gt_indices


def compute_metrics(retrieved_indices: torch.Tensor, gt_indices: torch.Tensor):
    bm = retrieved_indices.eq(gt_indices)  

    Q, N = bm.shape

    # ---- Recalls ----
    r1  = 100.0 * bm[:, :1].any(dim=1).float().mean()
    r5  = 100.0 * bm[:, :5].any(dim=1).float().mean()
    r10 = 100.0 * bm[:, :10].any(dim=1).float().mean()

    ranks = torch.full((Q,), N + 1, device=bm.device, dtype=torch.float32)

    rows, cols = torch.where(bm)     
    if rows.numel() > 0:
        first_pos = torch.zeros(Q, dtype=torch.bool, device=bm.device)
        for r, c in zip(rows.tolist(), cols.tolist()):
            if not first_pos[r]:
                ranks[r] = float(c + 1)  # 1-based
                first_pos[r] = True

    medr = ranks.median()

    return {"R@1": r1, "R@5": r5, "R@10": r10, "MedR": medr}



def run_retrieval(model, data_loader, device, model_name, use_midi):
    """Wrapper function to run all steps for text-audio/audio-text retrieval"""
    
    if model_name == "trimodal":
        audio_features, midi_features, text_features = get_muscall_features_audio_midi_text(
            model, data_loader, device, model_name)
        
        score_matrix_audio = compute_sim_score(audio_features=audio_features, text_features=text_features)
        score_matrix_midi = compute_sim_score(midi_features=midi_features, text_features=text_features)
        
        retrieved_indices_audio, gt_indices_audio = get_ranking(score_matrix_audio, device)
        retrieval_metrics_audio = compute_metrics(retrieved_indices_audio, gt_indices_audio)
        
        retrieved_indices_midi, gt_indices_midi = get_ranking(score_matrix_midi, device)
        retrieval_metrics_midi = compute_metrics(retrieved_indices_midi, gt_indices_midi)

        return retrieval_metrics_audio, retrieval_metrics_midi

    elif model_name == "midi_text":
        midi_features, text_features = get_muscall_features_midi(
            model, data_loader, device, model_name)
        score_matrix = compute_sim_score(midi_features=midi_features, text_features=text_features)
    elif model_name == "audio_text" :
        audio_features, text_features = get_muscall_features(
            model, data_loader, device, model_name)
        score_matrix = compute_sim_score(audio_features=audio_features, text_features=text_features)
    
    retrieved_indices, gt_indices = get_ranking(score_matrix, device)
    retrieval_metrics = compute_metrics(retrieved_indices, gt_indices)

    return retrieval_metrics



def compute_random_metrics(num_queries, num_items):
    retrieved_indices_random = torch.randint(0, num_items, (num_queries, num_items))

    gt_indices = torch.zeros((num_queries, num_items, 1))
    for i in range(num_queries):
        gt_indices[i] = torch.full((num_queries, 1), i)
    gt_indices = gt_indices.squeeze(-1)

    return compute_metrics(retrieved_indices_random, gt_indices)


class Retrieval:
    def __init__(self, muscall_config, test_set_size=0):
        super().__init__()
        self.muscall_config = muscall_config
        self.device = torch.device(self.muscall_config.training.device)
        self.path_to_model = os.path.join(
            self.muscall_config.env.experiments_dir,
            self.muscall_config.env.experiment_id,
            "best_model.pth.tar",
        )
        print("path to model", self.path_to_model)
        self.model_name = self.muscall_config.model_config.model_name
        self.test_set_size = test_set_size

        self.load_dataset()
        self.build_model()
        
        
    def load_dataset(self):
        dataset_name = self.muscall_config.dataset_config.dataset_name
        
        if dataset_name == "piast_at":
            dataset = PIAST_AT_Dataset(self.muscall_config.dataset_config, dataset_type="test")
        elif dataset_name == "piast_yt":
            dataset = PIAST_YT_Dataset(self.muscall_config.dataset_config, dataset_type="test")
        elif dataset_name == "piast_joint":
            dataset = PIAST_AT_Dataset(self.muscall_config.dataset_config, dataset_type="test")
        else:
            raise ValueError("{} dataset is not supported.".format(dataset_name))
        
        indices = torch.randperm(len(dataset))[:self.test_set_size]
        random_dataset = Subset(dataset, indices)
        self.batch_size = 1
        self.data_loader = DataLoader(
            dataset=random_dataset,
            batch_size=self.batch_size,
            drop_last=False,
        )

    def build_model(self):
        if self.model_name == "audio_text":
            self.model = PianoBind_AudioText(self.muscall_config.model_config)
        elif self.model_name == "midi_text":
            self.model = PianoBind_MIDIText(self.muscall_config.model_config)
        elif self.model_name == "trimodal":
            self.model = PianoBind_Trimodal(self.muscall_config.model_config)
        
        self.checkpoint = torch.load(self.path_to_model)
        self.model.load_state_dict(self.checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()


    def print_top_results(self, retrieved_indices, test_dataset, top_k=5):
        results = {}
        for query_idx in range(min(len(test_dataset), self.test_set_size)):
            query_data = test_dataset[query_idx]
            query_caption = query_data['caption']
            query_audio_path = query_data['audio_path']
            results[query_caption] = {
                "audio_path": query_audio_path,
                "retrieved_tracks": []
            }

            top_k_indices = retrieved_indices[query_idx][:top_k].tolist()
            for rank, idx in enumerate(top_k_indices, start=1):
                top_data = test_dataset[idx]
                top_caption = top_data['caption']
                top_audio_path = top_data['audio_path']
                results[query_caption]["retrieved_tracks"].append({
                    "rank": rank,
                    "caption": top_caption,
                    "audio_path": top_audio_path
                })
        return results

    
    def calculate_actual_ranks(self, retrieved_indices, gt_indices):
        actual_ranks = {}
        for query_idx, gt_idx in enumerate(gt_indices):
            retrieved_idx = (retrieved_indices[query_idx] == gt_idx).nonzero(as_tuple=True)[0]
            if retrieved_idx.size(0) > 0:
                rank = retrieved_idx.item() + 1
            else:
                rank = -1 
            actual_ranks[query_idx] = rank
        return actual_ranks        
            
    def tensor_to_list(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.tolist() 
        else:
            return tensor
        
    def evaluate(self):
        with open('./data/datasets/test_dataset_tag.json', 'r') as file:
            test_dataset = json.load(file)
        

        if self.model_name == "trimodal":
            audio_features, midi_features, text_features = get_muscall_features_audio_midi_text(
                self.model, self.data_loader, self.device, self.model_name
            )

            # ---------------------------
            # (A) Audio-text Retrieval 
            # ---------------------------
            score_matrix_audio = audio_features @ text_features.t()
            retrieved_indices_audio, gt_indices_audio = get_ranking(score_matrix_audio, self.device)
            retrieval_metrics_audio = compute_metrics(retrieved_indices_audio, gt_indices_audio)
            actual_ranks_audio = self.calculate_actual_ranks(retrieved_indices_audio, gt_indices_audio)
            print("Retrieval_metrics (Audio->Text): ", retrieval_metrics_audio)
            

            # ---------------------------
            # (B) MIDI-text Retrieval 
            # ---------------------------
            score_matrix_midi = midi_features @ text_features.t()
            retrieved_indices_midi, gt_indices_midi = get_ranking(score_matrix_midi, self.device)
            retrieval_metrics_midi = compute_metrics(retrieved_indices_midi, gt_indices_midi)
            actual_ranks_midi = self.calculate_actual_ranks(retrieved_indices_midi, gt_indices_midi)
            print("Retrieval_metrics (MIDI->Text): ", retrieval_metrics_midi)

            # ---------------------------
            # (C1) Average
            # ---------------------------
            combined_features_avg = (audio_features + midi_features) / 2.0
            combined_features_avg = combined_features_avg / combined_features_avg.norm(dim=-1, keepdim=True)
            score_matrix_avg = combined_features_avg @ text_features.t()

            retrieved_indices_avg, gt_indices_avg = get_ranking(score_matrix_avg, self.device)
            retrieval_metrics_avg = compute_metrics(retrieved_indices_avg, gt_indices_avg)
            actual_ranks_avg = self.calculate_actual_ranks(retrieved_indices_avg, gt_indices_avg)
            print("Retrieval_metrics (Audio+MIDI - Average): ", retrieval_metrics_avg)




            audio_results = self.print_top_results(retrieved_indices_audio, test_dataset)
            midi_results = self.print_top_results(retrieved_indices_midi, test_dataset)
            avg_results = self.print_top_results(retrieved_indices_avg, test_dataset)

            final_results = {
                "retrieval_metrics_audio": {k: self.tensor_to_list(v) for k, v in retrieval_metrics_audio.items()},
                "retrieval_metrics_midi": {k: self.tensor_to_list(v) for k, v in retrieval_metrics_midi.items()},
                "retrieval_metrics_avg": {k: self.tensor_to_list(v) for k, v in retrieval_metrics_avg.items()},
                "audio_results": audio_results,
                "midi_results": midi_results,
                "avg_results": avg_results,
                "actual_ranks_audio": actual_ranks_audio,
                "actual_ranks_midi": actual_ranks_midi,
                "actual_ranks_avg": actual_ranks_avg,
            }

            save_path = f"./save/experiments/{self.muscall_config.env.experiment_id}/retrieval_results.json"
            with open(save_path, "w") as f:
                json.dump(final_results, f, indent=4)

            return (
                retrieval_metrics_audio,
                retrieval_metrics_midi,
                retrieval_metrics_avg,
            )





        else:
            # midi-text or audio-text
            if self.model_name in ["midi_text", "muscall_midi_cliptext"]:
                midi_features, text_features = get_muscall_features_midi(
                    self.model, self.data_loader, self.device, self.model_name)
                score_matrix = compute_sim_score(midi_features=midi_features, text_features=text_features)
            elif self.model_name in ["audio_text", "muscall_cliptext"]:
                audio_features, text_features = get_muscall_features(
                    self.model, self.data_loader, self.device, self.model_name)
                score_matrix = compute_sim_score(audio_features=audio_features, text_features=text_features)

            retrieved_indices, gt_indices = get_ranking(score_matrix, self.device)
            retrieval_metrics = compute_metrics(retrieved_indices, gt_indices)
            actual_ranks = self.calculate_actual_ranks(retrieved_indices, gt_indices)
            print('Retrieval_metrics: ', retrieval_metrics)

            results = self.print_top_results(retrieved_indices, test_dataset)

            final_results = {
                'retrieval_metrics': {k: self.tensor_to_list(v) for k, v in retrieval_metrics.items()},
                'results': results,
                'actual_ranks': actual_ranks
            }

            with open(f'./save/experiments/{self.muscall_config.env.experiment_id}/retrieval_results.json', 'w') as f:
                json.dump(final_results, f, indent=4)

            return retrieval_metrics
