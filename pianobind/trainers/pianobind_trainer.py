import os
import time
import numpy as np
from itertools import islice

import torch

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset


from pianobind.datasets.piast_at import PIAST_AT_Dataset
from pianobind.datasets.piast_yt import PIAST_YT_Dataset
from pianobind.datasets.multi_source_dataset import MultiSourceDataset, RatioSampler
from pianobind.trainers.base_trainer import BaseTrainer

from pianobind.utils.evaluation_utils import run_retrieval
from pianobind.utils.audio_utils import get_transform_chain
from pianobind.models.build_model import create_model, freeze_modules


class PianoBindTrainer(BaseTrainer):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.batch_size = self.config.training.dataloader.batch_size
        self.accumulation_steps = getattr(self.config.training, "accumulation_steps", 1) 

        self.load()

        self.scaler = torch.cuda.amp.GradScaler()
    
    def load_dataset(self):
        self.logger.write("Loading dataset")
        self.config.dataset_config.model = self.config.model_config.model_name     
           
        dataset_name = self.config.dataset_config.dataset_name
        if dataset_name == "piast_at":
            self.train_dataset = PIAST_AT_Dataset(self.config.dataset_config)
            self.val_dataset = PIAST_AT_Dataset(self.config.dataset_config, dataset_type="val")
        elif dataset_name == "piast_yt":
            self.train_dataset = PIAST_YT_Dataset(self.config.dataset_config)
            self.val_dataset = PIAST_YT_Dataset(self.config.dataset_config, dataset_type="val")
        elif dataset_name == "piast_joint":
            piast_yt_dataset = PIAST_YT_Dataset(self.config.dataset_config)
            piast_at_dataset = PIAST_AT_Dataset(self.config.dataset_config)

            self.logger.write(f"YT dataset size: {len(piast_yt_dataset)}")
            self.logger.write(f"AT dataset size: {len(piast_at_dataset)}")

            sampling_probs = self.config.dataset_config.sampling_prob
            self.logger.write(f"Sampling Ratio: {sampling_probs}")

            self.train_dataset = MultiSourceDataset([piast_yt_dataset, piast_at_dataset])

            self.logger.write(f"Cumulative sizes: {self.train_dataset.cumulative_sizes}")

            ratio_sampler = RatioSampler(
                dataset=self.train_dataset,
                ratios=sampling_probs,
                batch_size=self.batch_size,
                shuffle=True,
                seed=getattr(self.config, 'seed', None)
            )

            self.val_dataset = PIAST_AT_Dataset(self.config.dataset_config, dataset_type="val")
        else:
            raise ValueError(f"{dataset_name} dataset is not supported.")


        if dataset_name == "piast_joint":
            self.logger.write("Using Joint Dataset")
            self.train_loader = DataLoader(
                dataset=self.train_dataset,
                sampler=ratio_sampler,  
                batch_size=self.batch_size,  
                num_workers=self.config.training.dataloader.num_workers,
                pin_memory=True,
                drop_last=True
            )

        else:
            self.logger.write("Using Single Dataset")
            self.train_loader = DataLoader(
                dataset=self.train_dataset,
                **self.config.training.dataloader,
                drop_last=True
            )

        self.val_loader = DataLoader(
            dataset=self.val_dataset,
            **self.config.training.dataloader,
            drop_last=True
        )

        self.logger.write(f"Number of training samples: {len(self.train_dataset)}")

    def build_optimizer(self):
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        opt_cfg = self.config.training.optimizer
        self.optimizer = getattr(torch.optim, opt_cfg.name)(params, **opt_cfg.args)

        steps_per_epoch = len(self.train_loader.dataset) // self.batch_size
        total_steps = steps_per_epoch * self.config.training.epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=int(total_steps * 0.1))

    def build_model(self):

        self.logger.write("Building model")
        self.model = create_model(self.config.model_config)
        freeze_modules(self.model, self.config.freeze, self.logger)

        self.print_parameters()
        self.model.to(self.device)
        
    def get_retrieval_metrics(self, trimodal: bool = False):
        indices = torch.randperm(len(self.val_dataset))[:1000]
        random_val_subset = Subset(self.val_dataset, indices)
        val_subset_loader = DataLoader(
            random_val_subset,
            batch_size=self.batch_size,
        )

        retrieval_metrics = run_retrieval(
            model=self.model,
            data_loader=val_subset_loader,
            device=self.device,
            model_name=self.config.model_config.model_name,
            use_midi=self.config.dataset_config.midi.use_midi,
        )

        if trimodal:
            audio_metrics = retrieval_metrics[0]
            midi_metrics = retrieval_metrics[1]
            return audio_metrics["Median Rank"].item(), midi_metrics["Median Rank"].item()
        else:
            median_rank = retrieval_metrics["Median Rank"]
            return median_rank.item() if isinstance(median_rank, torch.Tensor) else median_rank

        
    def _log_and_save(self, epoch, train_loss, val_loss, med_rank, audio_med_rank=None, midi_med_rank=None):
        epoch_time = time.time() - self.epoch_start_time
        lr = self.scheduler.get_last_lr()[0]

        checkpoint = {
            "epoch": epoch + 1,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        model_name = self.config.model_config.model_name.lower()

        if model_name == "trimodal":
            self.logger.update_training_log_audio_midi(
                epoch + 1,
                train_loss,
                val_loss,
                epoch_time,
                lr,
                audio_med_rank,
                midi_med_rank,
                med_rank
            )
            if self.audio_frozen:
                is_best = midi_med_rank < self.best_midi_rank
                if is_best:
                    self.best_midi_rank = midi_med_rank
            elif self.midi_frozen:
                is_best = audio_med_rank < self.best_audio_rank
                if is_best:
                    self.best_audio_rank = audio_med_rank
            else:
                is_best = med_rank < self.best_med_rank
                if is_best:
                    self.best_med_rank = med_rank
        else:
            self.logger.update_training_log(
                epoch + 1,
                train_loss,
                val_loss,
                epoch_time,
                lr,
                med_rank
            )
            is_best = med_rank < self.best_med_rank
            if is_best:
                self.best_med_rank = med_rank

        self.logger.save_checkpoint(state=checkpoint, is_best=is_best)
            
    def train(self):
        self.best_med_rank = float("inf")
        self.best_audio_rank = float("inf")
        self.best_midi_rank = float("inf")

        if os.path.exists(self.logger.checkpoint_path):
            self.logger.write(
                "Resumed training experiment with id {}".format(
                    self.logger.experiment_id
                )
            )
            self.load_ckp(self.logger.checkpoint_path)
        else:
            self.logger.write(
                "Started training experiment with id {}".format(
                    self.logger.experiment_id
                )
            )
            self.start_epoch = 0
        
        
        audio_frozen = False
        midi_frozen = False

            
        self.audio_frozen = audio_frozen
        self.midi_frozen = midi_frozen   

        if hasattr(self.model, 'audio_backbone'):
            audio_frozen = not any(p.requires_grad for p in self.model.audio_backbone.parameters())
        if hasattr(self.model, 'midi_backbone'):
            midi_frozen = not any(p.requires_grad for p in self.model.midi_backbone.parameters())

        for epoch in range(self.start_epoch, self.config.training.epochs):
            self.epoch_start_time = time.time()

            train_loss = self.train_epoch(self.train_loader, is_training=True)
            val_loss = self.train_epoch_val(self.val_loader)

            model_name = self.config.model_config.model_name.lower()
            if model_name == "trimodal":
                audio_med_rank, midi_med_rank = self.get_retrieval_metrics(trimodal=True)
                med_rank = (audio_med_rank + midi_med_rank) / 2
                self._log_and_save(epoch, train_loss, val_loss, med_rank, audio_med_rank, midi_med_rank)
                
            else:
                med_rank = self.get_retrieval_metrics()  
                self._log_and_save(epoch, train_loss, val_loss, med_rank)


    def load_ckp(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["state_dict"], strict=False)
        
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        except ValueError as e:
            self.build_optimizer()
        
        self.start_epoch = checkpoint["epoch"]

    
    def train_epoch(self, data_loader, is_training):
        running_loss = 0.0
        n_batches = 0

        if is_training:
            self.model.train()
        else:
            self.model.eval()

        for batch_idx, batch in enumerate(data_loader):
            if len(batch[0]) < self.batch_size:
                continue
            
            try:
                batch = tuple(t.to(device=self.device, non_blocking=True) for t in batch)
                audio_id, input_audio, input_midi, text_input_ids, text_input_type_ids, text_attention_mask, data_idx = batch

                original_audio = None
                audio_data_config = self.config.dataset_config.audio
                if is_training and audio_data_config.augment:
                    original_audio = input_audio
                    augment_chain = get_transform_chain(
                        p_polarity=0,
                        p_gain=0,
                        p_noise=audio_data_config.p_noise,
                        p_pitch_shift=audio_data_config.p_pitch_shift,
                        sample_rate=audio_data_config.sr,
                    )
                    input_audio = augment_chain(input_audio.unsqueeze(1), audio_data_config.sr).squeeze(1)

                if batch_idx % 10 == 0:
                    self.logger.write(f"Batch {batch_idx}/{len(data_loader)}")

                with torch.amp.autocast('cuda', enabled=self.config.training.amp):
                    if self.config.model_config.model_name == "audio_text":
                        loss = self.model(
                            input_audio,
                            text_input_ids,
                            text_mask=text_attention_mask,
                            return_loss=True
                        )
                    elif self.config.model_config.model_name == "midi_text": 
                        loss = self.model(
                            input_midi, 
                            text_input_ids,
                            text_mask=text_attention_mask,
                            return_loss=True
                        )
                    elif self.config.model_config.model_name == "trimodal":
                        loss = self.model(
                            input_audio,
                            input_midi,
                            text_input_ids,
                            original_audio=original_audio,
                            return_loss=True
                        )

                # Backpropagation
                if is_training:
                    if self.config.training.amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()

                    # clamp temperature scaling if over log(100)
                    if hasattr(self.model, 'logit_scale'):
                        if self.model.logit_scale.item() > np.log(100):
                            self.model.logit_scale.data = torch.clamp(
                                self.model.logit_scale.data, max=np.log(100)
                            )

                    self.scheduler.step()
                    self.optimizer.zero_grad()

                running_loss += loss.item()
                n_batches += 1
                
                if batch_idx % 10 == 0:
                    self.logger.write(f"Loss: {loss.item():.4f}")
                
            except Exception as e:
                import traceback
                self.logger.write(traceback.format_exc())
                continue

        if n_batches == 0:
            return 0.0
        
        return running_loss / n_batches

    def train_epoch_val(self, data_loader):
        with torch.no_grad():
            prev_amp_setting = self.config.training.amp
            self.config.training.amp = False  
            loss = self.train_epoch(data_loader, is_training=False)
            self.config.training.amp = prev_amp_setting  
        return loss


