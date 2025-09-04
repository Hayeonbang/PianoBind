import os
import time
import torch
from pianobind.trainers.pianobind_trainer import PianoBindTrainer
from pianobind.modules.audio_backbones import ModifiedResNet
from pianobind.modules.midi_backbones import MidiBERTLM_encoder

class PianoBindFinetuner(PianoBindTrainer):
    def __init__(self, config, logger):
        config.env.experiment_id = f"{logger.get_timestamp()}"
        super().__init__(config, logger)
        self.pretrained_experiment_id = config.finetuning.pretrained_experiment_id
        self.init_finetuning()

    def init_finetuning(self):
        self.load_pretrained_model()
        self.freeze_layers()

    def load_pretrained_model(self):
        pretrained_path = f"./save/experiments/{self.pretrained_experiment_id}/best_model.pth.tar"
        try:
            checkpoint = torch.load(pretrained_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            self.logger.write(f"Loaded pretrained model from {pretrained_path}")
        except FileNotFoundError:
            self.logger.write(f"Pretrained model not found at {pretrained_path}. Starting from scratch.")
        except Exception as e:
            self.logger.write(f"Error loading pretrained model: {str(e)}")
            raise

    def freeze_layers(self):
        self.freeze_audio_layers()
        self.freeze_midi_layers()
        self.freeze_text_layers()

        def count_trainable_layers(layer_blocks):
            return sum(1 for layer in layer_blocks if any(p.requires_grad for p in layer.parameters()))

        if hasattr(self.model, 'audio_backbone') and isinstance(self.model.audio_backbone, ModifiedResNet):
            audio_layers = [self.model.audio_backbone.layer1, self.model.audio_backbone.layer2,
                            self.model.audio_backbone.layer3, self.model.audio_backbone.layer4]
            unfrozen_audio_layers = count_trainable_layers(audio_layers)
            self.logger.write(f"Trainable audio layers: {unfrozen_audio_layers}/{len(audio_layers)}")

        if hasattr(self.model, 'midi_backbone') and isinstance(self.model.midi_backbone, MidiBERTLM_encoder):
            midi_layers = self.model.midi_backbone.model.midibert.bert.encoder.layer
            unfrozen_midi_layers = count_trainable_layers(midi_layers)
            self.logger.write(f"Trainable MIDI layers: {unfrozen_midi_layers}/{len(midi_layers)}")

        if hasattr(self.model, 'textual_head'):
            text_layers = self.model.textual_head.encoder.layer
            unfrozen_text_layers = count_trainable_layers(text_layers)
            self.logger.write(f"Trainable text layers: {unfrozen_text_layers}/{len(text_layers)}")

        total_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.write(f"# of trainable parameters: {total_trainable:,}")


    def freeze_audio_layers(self):
        trainable_audio = self.config.finetuning.get('trainable_audio_layers', None)

        if hasattr(self.model, 'audio_backbone') and isinstance(self.model.audio_backbone, ModifiedResNet):
            audio_layers = [
                self.model.audio_backbone.layer1,
                self.model.audio_backbone.layer2,
                self.model.audio_backbone.layer3,
                self.model.audio_backbone.layer4,
            ]

            if trainable_audio is None or trainable_audio >= len(audio_layers):
                return
            else:
                for param in self.model.audio_backbone.parameters():
                    param.requires_grad = False

                for i, layer in enumerate(reversed(audio_layers)):
                    if i < trainable_audio:
                        for param in layer.parameters():
                            param.requires_grad = True
        else:
            self.logger.write("Warning: Unexpected audio backbone structure. Skipping audio layer freezing.")

    def freeze_midi_layers(self):
        trainable_midi = self.config.finetuning.get('trainable_midi_layers', None)
        if hasattr(self.model, 'midi_backbone') and isinstance(self.model.midi_backbone, MidiBERTLM_encoder):
            midi_layers = self.model.midi_backbone.model.midibert.bert.encoder.layer

            if trainable_midi is None or trainable_midi >= len(midi_layers):
                return
            else:
                for param in self.model.midi_backbone.parameters():
                    param.requires_grad = False

                for i, layer in enumerate(reversed(midi_layers)):
                    if i < trainable_midi:
                        for param in layer.parameters():
                            param.requires_grad = True

        else:
            self.logger.write("Warning: Unexpected midi backbone structure. Skipping midi layer freezing.")

    def freeze_text_layers(self):
        trainable_text = self.config.finetuning.get('trainable_text_layers', None)
        total_text_layers = len(self.model.textual_head.encoder.layer)

        if trainable_text is None or trainable_text >= total_text_layers:
            return
        else:
            for param in self.model.textual_head.parameters():
                param.requires_grad = False
            for i, layer in enumerate(self.model.textual_head.encoder.layer):
                if i >= total_text_layers - trainable_text:
                    for param in layer.parameters():
                        param.requires_grad = True


    def build_optimizer(self):
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer_config = self.config.training.optimizer  
        self.optimizer = getattr(torch.optim, optimizer_config.name)(
            params, **optimizer_config.args
        )

        num_train_optimization_steps = (
            int(len(self.train_loader.dataset) / self.batch_size)
            * self.config.training.epochs  
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_train_optimization_steps * 0.1
        )

    def train(self):
        best_med_rank = float("inf")
        
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
       

        for epoch in range(self.start_epoch, self.config.training.epochs):
            epoch_start_time = time.time()

            train_loss = self.train_epoch(self.train_loader, is_training=True)
            val_loss = self.train_epoch_val(self.val_loader)

            if self.config.model_config.model_name in ["audio_text", "midi_text"]:
                track_retrieval_metrics = True
                if track_retrieval_metrics:
                    med_rank = self.get_retrieval_metrics()  

                epoch_time = time.time() - epoch_start_time
                self.logger.update_training_log(
                    epoch + 1,
                    train_loss,
                    val_loss,
                    epoch_time,
                    self.scheduler.get_last_lr()[0],
                    med_rank,
                )

                checkpoint = {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }

                is_best = med_rank < best_med_rank 
                if is_best:
                    best_med_rank = med_rank
                self.logger.save_checkpoint(state=checkpoint, is_best=is_best)
                
            elif "audio_midi_roberta" in self.config.model_config.model_name:
                track_retrieval_metrics = True
                if track_retrieval_metrics:
                    audio_med_rank, midi_med_rank = self.get_retrieval_metrics_audio_midi()
                    med_rank = (audio_med_rank + midi_med_rank) / 2
                    
                epoch_time = time.time() - epoch_start_time
                self.logger.update_training_log_audio_midi(
                    epoch + 1,
                    train_loss,
                    val_loss,
                    epoch_time,
                    self.scheduler.get_last_lr()[0],
                    audio_med_rank,
                    midi_med_rank,
                    med_rank
                )       

                checkpoint = {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }

                is_best = med_rank < best_med_rank
                if is_best:
                    best_med_rank = med_rank

                self.logger.save_checkpoint(state=checkpoint, is_best=is_best)