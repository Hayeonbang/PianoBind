import torch
from torch.utils.data import Dataset, Sampler
import random
import numpy as np
from typing import List, Iterator, Optional, Sized

class MultiSourceDataset(Dataset):
    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self.dataset_sizes = [len(dataset) for dataset in datasets]
        self.cumulative_sizes = np.cumsum(self.dataset_sizes)
        
    def __len__(self):
        return sum(self.dataset_sizes)
    
    def __getitem__(self, idx):
        dataset_idx, sample_idx = self.get_dataset_and_index(idx)
        
        sample = self.datasets[dataset_idx][sample_idx]
        
        if isinstance(sample, dict):
            if dataset_idx == 0:
                sample['dataset_source'] = 'piast_yt'
            else:
                sample['dataset_source'] = 'piast_at'
            
            sample['dataset_idx'] = dataset_idx
            sample['sample_idx'] = sample_idx
            sample['global_idx'] = idx
        
        return sample
    
    def get_dataset_and_index(self, idx):
        if idx < self.cumulative_sizes[0]:
            dataset_idx = 0
            sample_idx = idx
        else:
            dataset_idx = 1
            sample_idx = idx - self.cumulative_sizes[0]

        
        return dataset_idx, sample_idx
    
    def get_raw_caption(self, idx):

        dataset_idx, sample_idx = self.get_dataset_and_index(idx)
        return self.datasets[dataset_idx].get_raw_caption(sample_idx)


class RatioSampler(Sampler):
    def __init__(self, 
                 dataset: MultiSourceDataset, 
                 ratios: List[float], 
                 batch_size: int,
                 shuffle: bool = True, 
                 seed: Optional[int] = None):
        """
        Args:
            dataset: MultiSourceDataset to sample from
            ratios: Sampling ratios for each dataset (must sum to 1.0)
            batch_size: Batch size
            shuffle: Whether to shuffle
            seed: Random seed
        """
        self.dataset = dataset
        self.ratios = ratios
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        
        if abs(sum(ratios) - 1.0) > 1e-5:
            print(f"Warning: The sum of ratios is not 1.0. Current sum: {sum(ratios)}")
            self.ratios = [r / sum(ratios) for r in ratios]
            print(f"Normalized ratios: {self.ratios}")
        
        if len(ratios) != len(dataset.datasets):
            raise ValueError(f"The number of ratios ({len(ratios)}) does not match the number of datasets ({len(dataset.datasets)})")
        
        self.total_samples = (len(dataset) // batch_size) * batch_size
        
        self.samples_per_dataset = [int(ratio * self.total_samples) for ratio in self.ratios]
        
        diff = self.total_samples - sum(self.samples_per_dataset)
        self.samples_per_dataset[0] += diff
        
    def __iter__(self) -> Iterator[int]:
        all_batches = []
        
        dataset_indices = []
        for dataset_idx in range(len(self.dataset.datasets)):
            if dataset_idx == 0:
                start_idx = 0
            else:
                start_idx = self.dataset.cumulative_sizes[dataset_idx - 1]
            
            end_idx = self.dataset.cumulative_sizes[dataset_idx]
            indices = list(range(start_idx, end_idx))
            
            if self.shuffle:
                if self.seed is not None:
                    rng = random.Random(self.seed + dataset_idx)
                    rng.shuffle(indices)
                else:
                    random.shuffle(indices)
            
            dataset_indices.append(indices)
        
        num_batches = self.total_samples // self.batch_size
        
        for batch_idx in range(num_batches):
            batch = []
            
            for dataset_idx, ratio in enumerate(self.ratios):
                samples_in_batch = int(ratio * self.batch_size)
                if dataset_idx == 0:  
                    samples_in_batch += self.batch_size - sum([int(r * self.batch_size) for r in self.ratios])
                
                for i in range(samples_in_batch):
                    global_idx = batch_idx * samples_in_batch + i
                    idx = global_idx % len(dataset_indices[dataset_idx])
                    batch.append(dataset_indices[dataset_idx][idx])
            
            if self.shuffle:
                if self.seed is not None:
                    rng = random.Random(self.seed + batch_idx + 100)
                    rng.shuffle(batch)
                else:
                    random.shuffle(batch)
            
            all_batches.append(batch)
        
        if self.shuffle:
            if self.seed is not None:
                rng = random.Random(self.seed + 200)
                rng.shuffle(all_batches)
            else:
                random.shuffle(all_batches)
        
        all_indices = []
        for batch in all_batches:
            all_indices.extend(batch)
        
        return iter(all_indices)
    
    def __len__(self) -> int:
        return self.total_samples 