import os
from collections import namedtuple

import torch
import yaml
from common.config import Config, get_function, instantiate
from dataloader.collate_fn import dynamic_padding
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from trainer.utils import (compute_avg_metrics, get_nvidia_smi_output,
                           get_tqdm, save_curves, save_metrics_to_csv)

yaml.add_representer(Config, lambda dumper, data: data.yaml_repr(dumper))

class EHRTrainer():
    def __init__(self, 
        model: torch.nn.Module,
        train_dataset: Dataset = None,
        test_dataset: Dataset = None,
        val_dataset: Dataset = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler.StepLR = None,
        metrics: dict = {},
        args: dict = {},
        sampler: callable = None,
        cfg = None,
        logger = None,
        run = None,
        accumulate_logits: bool = False,
        run_folder: str = None,
        last_epoch: int = None,
    ):
        
        self._initialize_basic_attributes(model, train_dataset, test_dataset, val_dataset, optimizer, scheduler, metrics, sampler, cfg, run, accumulate_logits, last_epoch)
        self._set_default_args(args)
        self.logger = logger
        self.run_folder = run_folder or os.path.join(self.cfg.paths.output_path, self.cfg.paths.run_name)
        self._log_basic_info()
        
        self.log("Initialize metrics")
        self.metrics = {k: instantiate(v) for k, v in metrics.items()} if metrics else {}
        
        self._initialize_mixed_precision()
        self._initialize_early_stopping()
        
    def _initialize_basic_attributes(self, model, train_dataset, test_dataset, val_dataset, optimizer, scheduler, metrics, sampler, cfg, run, accumulate_logits, last_epoch):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = {k: instantiate(v) for k, v in metrics.items()} if metrics else {}
        self.sampler = sampler
        self.cfg = cfg
        self.run = run
        self.accumulate_logits = accumulate_logits
        self.continue_epoch = last_epoch + 1 if last_epoch is not None else 0

    def _log_basic_info(self):
        self.log(f"Run on {self.device}")
        self.log(f"Run folder: {self.run_folder}")
        self.log("Send model to device")
        self.log("Initialize metrics")
        if torch.cuda.is_available():
            self.log(f"Memory on GPU: {torch.cuda.get_device_properties(0).total_memory/1e9} GB")
        self.log(f"PyTorch version: {torch.__version__}")
        self.log(f"CUDA version: {torch.version.cuda}")
    
    def _set_default_args(self, args):
        default_args = {
            'save_every_k_steps': float('inf'),
            'collate_fn': get_function(dynamic_padding) if isinstance(dynamic_padding, str) else dynamic_padding
        }
        self.args = {**default_args, **args}
        if not (self.args['effective_batch_size'] % self.args['batch_size'] == 0):
            raise ValueError('effective_batch_size must be a multiple of batch_size')

    def _initialize_mixed_precision(self):
        if self.cfg.trainer_args.get('mixed_precision', False):
            self.scaler = GradScaler()
            #raise ValueError("Mixed precision produces unstable results (nan loss). Please use full precision.")
        else:
            self.scaler = None

    def _initialize_early_stopping(self):
        self.best_val_loss = float('inf') # Best observed validation loss
        early_stopping = self.cfg.trainer_args.get('early_stopping', False)
        self.early_stopping = True if early_stopping else False
        self.early_stopping_patience = early_stopping if early_stopping else 1000 # Set patience parameter, for example, to 10 epochs.
        self.early_stopping_counter = 0  # Counter to keep track of epochs since last best val loss
        self.stop_training = False
        # Get the metric to use for early stopping from the config
        self.stopping_metric = self.cfg.trainer_args.get('stopping_metric', 'val_loss')
        self.log(f"Early stopping: {self.early_stopping} with patience {self.early_stopping_patience} and metric {self.stopping_metric}")

    def train(self, **kwargs):
        self.log(f"Torch version {torch.__version__}")
        self._update_attributes(**kwargs)

        self.accumulation_steps: int = self.args['effective_batch_size'] // self.args['batch_size']
        dataloader = self.setup_training()
        self.log(f'Test validation before starting training')
        self.validate(epoch=0)
        for epoch in range(self.continue_epoch, self.args['epochs']):
            self._train_epoch(epoch, dataloader)
            if self.stop_training:
                break
    
    def _train_epoch(self, epoch: int, dataloader: DataLoader)->None:
        train_loop = get_tqdm(dataloader)
        train_loop.set_description(f'Train {epoch}')
        epoch_loss = []
        step_loss = 0
        for i, batch in enumerate(train_loop):
            step_loss += self._train_step(batch).item()
            if (i+1) % self.accumulation_steps == 0:
                self._clip_gradients()
                self._update_and_log(step_loss, train_loop, epoch_loss)
                step_loss = 0
            if i%100==0:
                self.run_log_gpu()
        self.validate_and_log(epoch, epoch_loss, train_loop)
        torch.cuda.empty_cache()
        del train_loop
        del epoch_loss

    def _clip_gradients(self):
        # First, unscale the gradients
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        
        # Then clip them if needed
        if self.cfg.trainer_args.get('gradient_clip', False):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.trainer_args.gradient_clip.get('max_norm', 1.0))

    def _train_step(self, batch: dict):
        self.optimizer.zero_grad()
        if self.scaler is not None:
            with autocast():                
                self.batch_to_device(batch)
                outputs = self.model(batch)
                unscaled_loss = outputs.loss  # This is the original, unscaled loss value
                scaled_loss = self.scaler.scale(unscaled_loss)  # Scale the loss for backward
            scaled_loss.backward()
        else:
            self.batch_to_device(batch)
            outputs = self.model(batch)
            unscaled_loss = outputs.loss
            unscaled_loss.backward()

        return unscaled_loss

    def _update_and_log(self, step_loss, train_loop, epoch_loss):
        """Updates the model and logs the loss"""
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()
        train_loop.set_postfix(loss=step_loss / self.accumulation_steps)
        epoch_loss.append(step_loss / self.accumulation_steps)

        if self.args['info']:
            for param_group in self.optimizer.param_groups:
                current_lr = param_group['lr']
                self.run_log('Learning Rate', current_lr)
                break
        self.run_log('Train loss', step_loss / self.accumulation_steps)

    def validate_and_log(self, epoch: int, epoch_loss: float, train_loop: DataLoader)-> None:
        val_loss, metrics = self.validate(epoch)
        if self._should_stop_early(val_loss, epoch, epoch_loss, metrics):
            return 
        self._save_checkpoint_conditionally(epoch, epoch_loss, val_loss, metrics)
        self._self_log_results(epoch, val_loss, metrics, epoch_loss, len(train_loop))

    def _save_checkpoint_conditionally(self, epoch: int, epoch_loss: float, val_loss: float, metrics: dict) -> None:
        should_save = (
            (epoch % self.args.get('checkpoint_frequency', 1) == 0) or 
            (epoch == self.args['epochs'] - 1)
        )
        should_save = should_save and (not self.early_stopping)
        if should_save:
            self._save_checkpoint(epoch, train_loss=epoch_loss, val_loss=val_loss, metrics=metrics, final_step_loss=epoch_loss[-1])

    def _self_log_results(self, epoch: int, val_loss: float, metrics: dict, epoch_loss: float, len_train_loop: int)->None:
        for k, v in metrics.items():
            self.run_log(name = k, value = v)
        self.run_log(name='Val loss', value=val_loss)
        self.log(f'Epoch {epoch} train loss: {sum(epoch_loss) / (len_train_loop / self.accumulation_steps)}')
        self.log(f'Epoch {epoch} val loss: {val_loss}')
        self.log(f'Epoch {epoch} metrics: {metrics}\n')

    def _should_stop_early(self, val_loss: float, epoch: int, epoch_loss: float, metrics: dict) -> bool:
        if not self.early_stopping:
            return False
        # Get the current value of the metric
        current_metric_value = metrics.get(self.stopping_metric, val_loss)
        self._initialize_best_metric_value(current_metric_value)
        if current_metric_value < self.best_metric_value:
            self.best_metric_value = current_metric_value
            self.early_stopping_counter = 0
            self._save_checkpoint(epoch, train_loss=epoch_loss, val_loss=val_loss, metrics=metrics, final_step_loss=epoch_loss[-1])
            return False
        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.early_stopping_patience:
                self.log("Early stopping triggered!")
                self.stop_training = True
                return True
        return False

    def _initialize_best_metric_value(self, current_metric_value: float) -> None:
        if not hasattr(self, 'best_metric_value'):
            self.best_metric_value = current_metric_value
    
    def _initialize_best_metric_value(self, current_metric_value: float) -> None:
        if not hasattr(self, 'best_metric_value'):
            self.best_metric_value = current_metric_value

    def setup_training(self) -> DataLoader:
        """Sets up the training dataloader and returns it"""
        self.log(get_nvidia_smi_output())
        self.model.train()
        self.save_setup()
        dataloader = DataLoader(self.train_dataset, batch_size=self.args['batch_size'], sampler=self.sampler,
                                shuffle=self.args.get('shuffle', True), collate_fn=self.args['collate_fn'])
        return dataloader

    def validate(self, epoch: int)->tuple:
        """Returns the validation loss and metrics"""
        if self.val_dataset is None:
            self.log('No validation dataset provided')
            return None, None
        
        self.model.eval()
        dataloader = self.get_val_dataloader()
        val_loop = get_tqdm(dataloader)
        val_loop.set_description('Validation')
        val_loss = 0
        
        metric_values = {name: [] for name in self.metrics}
        logits_list = [] if self.accumulate_logits else None
        targets_list = [] if self.accumulate_logits else None

        with torch.no_grad():
            for batch in val_loop:
                self.batch_to_device(batch)
                outputs = self.model(batch)
                val_loss += outputs.loss.item()

                if self.accumulate_logits:
                    logits_list.append(outputs.logits.cpu())
                    targets_list.append(batch['target'].cpu())
                else:
                    for name, func in self.metrics.items():
                        metric_values[name].append(func(outputs, batch))

        if self.accumulate_logits:
            metric_values = self.process_binary_classification_results(logits_list, targets_list, epoch)
        else:
            metric_values = compute_avg_metrics(metric_values)
        
        self.model.train()
        
        return val_loss / len(val_loop), metric_values

    def process_binary_classification_results(self, logits:list, targets:list, epoch:int)->dict:
        """Process results specifically for binary classification."""
        targets = torch.cat(targets)
        logits = torch.cat(logits)
        batch = {'target': targets}
        outputs = namedtuple('Outputs', ['logits'])(logits)
        acc_metrics = {}
        for name, func in self.metrics.items():
            v = func(outputs, batch)
            self.log(f"{name}: {v}")
            acc_metrics[name] = v
        save_curves(self.run_folder, logits, targets, epoch)
        save_metrics_to_csv(self.run_folder, acc_metrics, epoch)
        return acc_metrics

    def get_val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.args.get('val_batch_size', self.args['batch_size']), 
            shuffle=self.args.get('shuffle', True), 
            collate_fn=self.args['collate_fn']
        )

    def batch_to_device(self, batch: dict) -> None:
        """Moves a batch to the device in-place"""
        for key, value in batch.items():
            batch[key] = value.to(self.device)

    def log(self, message: str) -> None:
        """Logs a message to the logger and stdout"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def run_log_gpu(self):
        """Logs the GPU memory usage to the run"""
        memory_allocated = torch.cuda.memory_allocated(device=self.device)/1e9
        max_memory_reserved = torch.cuda.max_memory_reserved(device=self.device)/1e9
        memory_cached = torch.cuda.memory_reserved(device=self.device)/1e9
        self.run_log(name="GPU Memory Allocated in GB", value=memory_allocated)
        self.run_log(name="GPU Max Memory Allocated in GB", value=max_memory_reserved)
        self.run_log(name="GPU Memory Cached in GB", value=memory_cached)

    def run_log(self, name, value):
        if self.run is not None:
            self.run.log_metric(name=name, value=value)
        else:
            self.log(f"{name}: {value}")

    def save_setup(self)->None:
        """Saves the config and model config"""
        self.model.config.save_pretrained(self.run_folder)  
        with open(os.path.join(self.run_folder, 'pretrain_config.yaml'), 'w') as file:
            yaml.dump(self.cfg.to_dict(), file)
        self.log(f'Saved config to {self.run_folder}')  

    def _save_checkpoint(self, epoch:int, **kwargs)->None:
        """Saves a checkpoint. Model with optimizer and scheduler if available."""
        # Model/training specific
        id=f'epoch{epoch}_end'
        os.makedirs(os.path.join(self.run_folder, 'checkpoints'), exist_ok=True)
        checkpoint_name = os.path.join(self.run_folder, 'checkpoints', f'checkpoint_{id}.pt')

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            **kwargs
        }, checkpoint_name)

    def _update_attributes(self, **kwargs):
        for key, value in kwargs.items():
            if key == 'args':
                self.args = {**self.args, **value}
            else:
                setattr(self, key, value)