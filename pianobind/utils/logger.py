import os
import torch
import time
from omegaconf import OmegaConf


class Logger:
    def __init__(self, config):
        self.config = config
        self.experiment_id = self._get_experiment_id()
        self.experiment_dir = os.path.join(self.config.env.experiments_dir, self.experiment_id)
        self.checkpoint_path = os.path.join(self.experiment_dir, 'checkpoint.pth.tar')
        self.log_filename = os.path.join(self.experiment_dir, "train_log.tsv")

        self._init_log_file()

    def _get_experiment_id(self):
        if "experiment_id" in self.config.env and self.config.env.experiment_id is not None:
            return self.config.env.experiment_id
        experiment_id = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
        OmegaConf.update(self.config, "env.experiment_id", experiment_id)
        return experiment_id

    def _init_log_file(self):
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        with open(self.log_filename, 'a') as f:
            if self.config.model_config.model_name == "tri_modal":
                f.write('Epoch\ttrain_loss\tval_loss\taudio_rank\tmidi_rank\tavg_rank\tepoch_time\tlearing_rate\ttime_stamp\n')
            else:
                f.write('Epoch\ttrain_loss\tval_loss\trank\tepoch_time\tlearing_rate\ttime_stamp\n')


    def _save_config(self):
        config_path = os.path.join(self.experiment_dir, 'config.yaml')
        if not os.path.exists(config_path):
            try:
                OmegaConf.save(self.config, config_path)
                print(f"✅ Config saved successfully to {config_path}")
            except Exception as e:
                print(f"❌ Error saving config: {str(e)}")
        else:
            print(f"⚠️ Config file already exists at {config_path}")

    def write(self, text):
        print(text)

    def update_training_log(self, epoch, train_loss, val_loss, epoch_time, learning_rate,
                            rank=None, audio_rank=None, midi_rank=None, avg_rank=None):
        time_stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

        if self.config.model_config.model_name == "tri_modal":
            self.write(
                f"Epoch {epoch}, train loss {train_loss:.4f}, val loss {val_loss:.4f}, "
                f"audio_r10 {audio_rank}, midi_r10 {midi_rank}, avg_r10 {avg_rank}, "
                f"epoch-time {epoch_time}s, lr {learning_rate:.5f}, time-stamp {time_stamp}"
            )
            with open(self.log_filename, 'a') as f:
                f.write(f"{epoch}\t{train_loss}\t{val_loss}\t{audio_rank}\t{midi_rank}\t{avg_rank}\t"
                        f"{epoch_time}\t{learning_rate}\t{time_stamp}\n")
        else:
            self.write(
                f"Epoch {epoch}, train loss {train_loss:.4f}, val loss {val_loss:.4f}, "
                f"rank {rank}, epoch-time {epoch_time}s, lr {learning_rate:.5f}, time-stamp {time_stamp}"
            )
            with open(self.log_filename, 'a') as f:
                f.write(f"{epoch}\t{train_loss}\t{val_loss}\t{rank}\t"
                        f"{epoch_time}\t{learning_rate}\t{time_stamp}\n")
    def get_timestamp(self):
        return str(time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime()))
    
    def save_checkpoint(self, state, is_best=False):
        torch.save(state, self.checkpoint_path)
        if is_best:
            self.write("Saving best model so far")
            best_model_path = os.path.join(self.experiment_dir, 'best_model.pth.tar')
            torch.save(state, best_model_path)

    def save_config(self):
        self._save_config()
