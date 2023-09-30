import os
from collections import namedtuple

import numpy as np
import torch
import yaml
from common.config import Config, get_function, instantiate
from common.logger import TqdmToLogger
from dataloader.collate_fn import dynamic_padding
from sklearn.metrics import precision_recall_curve, roc_curve
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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
        sampler = None,
        cfg = None,
        logger = None,
        run = None,
        accumulate_logits = False
    ):
        self.logger = logger
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.log(f"Run on {self.device}")
        self.run_folder = os.path.join(cfg.paths.output_path, cfg.paths.run_name)
        self.log(f"Run folder: {self.run_folder}")
        self.log("Send model to device")
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.log("Initialize metrics")
        
        self.metrics = {k: instantiate(v) for k, v in metrics.items()} if metrics else {}
        self.sampler = sampler
        self.cfg = cfg
        self.run = run
        self.accumulate_logits = accumulate_logits
        
        if self.cfg.trainer_args.get('mixed_precision', False):
            raise ValueError("Mixed precision produces unstable results (nan loss). Please use full precision.")
            self.scaler = GradScaler()
        else:
            self.scaler = None
        default_args = {
            'save_every_k_steps': float('inf'),
            'collate_fn': dynamic_padding
        }
        if isinstance(default_args['collate_fn'] ,str):
            default_args['collate_fn'] = get_function(default_args['collate_fn'])

        self.args = {**default_args, **args}
        if torch.cuda.is_available():
            self.log(f"Memory on GPU: {torch.cuda.get_device_properties(0).total_memory/1e9} GB")
        self.log(f"PyTorch version: {torch.__version__}" )
        self.log(f"CUDA version: {torch.version.cuda}")

        if not (self.args['effective_batch_size'] % self.args['batch_size'] == 0):
            raise ValueError('effective_batch_size must be a multiple of batch_size')

    def update_attributes(self, **kwargs):
        for key, value in kwargs.items():
            if key == 'args':
                self.args = {**self.args, **value}
            else:
                setattr(self, key, value)

    def validate_training(self):
        assert self.model is not None, 'No model provided'
        assert self.train_dataset is not None, 'No training dataset provided'
        assert self.optimizer is not None, 'No optimizer provided'

    def train(self, **kwargs):
        self.log(f"Torch version {torch.__version__}")
        self.update_attributes(**kwargs)
        self.validate_training()

        self.accumulation_steps: int = self.args['effective_batch_size'] // self.args['batch_size']
        dataloader = self.setup_training()
        self.log(f'Test validation before starting training')
        self.validate(epoch=0)
        for epoch in range(self.args['epochs']):
            self.train_epoch(epoch, dataloader)

    def train_epoch(self, epoch: int, dataloader: DataLoader):
        train_loop = self.get_tqdm(dataloader)
        train_loop.set_description(f'Train {epoch}')
        epoch_loss = []
        step_loss = 0
        for i, batch in enumerate(train_loop):
            step_loss += self.train_step(batch).item()
            if (i+1) % self.accumulation_steps == 0:
                self.clip_gradients()
                self.update_and_log(i, step_loss, train_loop, epoch_loss)
                step_loss = 0
            if ((i+1) / self.accumulation_steps) % self.args['save_every_k_steps'] == 0:
                self.save_checkpoint(id=f'epoch{epoch}_step{(i+1) // self.accumulation_steps}', train_loss=step_loss / self.accumulation_steps)
            if i%100==0:
                self.run_log_gpu()
        self.validate_and_log(epoch, epoch_loss, train_loop)
        torch.cuda.empty_cache()
        del train_loop
        del epoch_loss

    def clip_gradients(self):
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        if self.cfg.trainer_args.get('gradient_clip', False):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.trainer_args.gradient_clip.get('max_norm', 1.0))

    def train_step(self, batch: dict):
        self.optimizer.zero_grad()
        if self.scaler is not None:
            with autocast():
                outputs = self.forward_pass(batch)
                loss = outputs.loss
                loss = self.scaler.scale(loss)
        else:
            outputs = self.forward_pass(batch)
            loss = outputs.loss

        self.backward_pass(loss)
        return loss

    def update_and_log(self, i, step_loss, train_loop, epoch_loss):
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
            #self.log(f'Train loss {(i+1) // self.accumulation_steps}: {step_loss / self.accumulation_steps}')
            for param_group in self.optimizer.param_groups:
                current_lr = param_group['lr']
                self.run_log('Learning Rate', current_lr)
                break
        self.run_log('Train loss', step_loss / self.accumulation_steps)

    def validate_and_log(self, epoch, epoch_loss, train_loop):
        val_loss, metrics = self.validate(epoch)
        self.run_log(name='Val loss', value=val_loss)
        for k, v in metrics.items():
            self.run_log(name = k, value = v)
        self.save_checkpoint(id=f'epoch{epoch}_end', train_loss=epoch_loss, val_loss=val_loss, metrics=metrics, final_step_loss=epoch_loss[-1])
        self.log(f'Epoch {epoch} train loss: {sum(epoch_loss) / (len(train_loop) / self.accumulation_steps)}')
        self.log(f'Epoch {epoch} val loss: {val_loss}')
        self.log(f'Epoch {epoch} metrics: {metrics}\n')

    def setup_training(self) -> DataLoader:
        self.model.train()
        self.save_setup()
        dataloader = DataLoader(self.train_dataset, batch_size=self.args['batch_size'], sampler=self.sampler,
                                shuffle=self.args.get('shuffle', True), collate_fn=self.args['collate_fn'])
        return dataloader

    def forward_pass(self, batch: dict):
        self.to_device(batch)
        return self.model(batch)

    def backward_pass(self, loss):
        loss.backward()

    def validate(self, epoch):
        """Returns the validation loss and metrics"""
        if self.val_dataset is None:
            self.log('No validation dataset provided')
            return None, None
        
        self.model.eval()
        dataloader = self.get_val_dataloader()
        val_loop = self.get_tqdm(dataloader)
        val_loop.set_description('Validation')
        val_loss = 0
        
        metric_values = {name: [] for name in self.metrics}
        logits_list = [] if self.accumulate_logits else None
        targets_list = [] if self.accumulate_logits else None

        with torch.no_grad():
            for batch in val_loop:
                outputs = self.forward_pass(batch)
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
            metric_values = self._compute_avg_metrics(metric_values)
        
        self.model.train()
        
        return val_loss / len(val_loop), metric_values


    def process_binary_classification_results(self, logits_list, targets_list, epoch):
        """Process results specifically for binary classification."""
        targets = torch.cat(targets_list)
        logits = torch.cat(logits_list)
        batch = {'target': targets}
        outputs = namedtuple('Outputs', ['logits'])(logits)
        acc_metrics = {}
        for name, func in self.metrics.items():
            v = func(outputs, batch)
            self.log(f"{name}: {v}")
            acc_metrics[name] = v
        self.save_curves(logits, targets, epoch)
        self.save_metrics_to_csv(acc_metrics, epoch)
        return acc_metrics

    def save_curves(self, logits, targets, epoch):
        """Saves the ROC and PRC curves to a csv file"""
        roc_name = os.path.join(self.run_folder, 'checkpoints', f'roc_curve_{epoch}.npz')
        prc_name = os.path.join(self.run_folder, 'checkpoints', f'prc_curve_{epoch}.npz')
        probas = torch.sigmoid(logits).cpu().numpy()
        fpr, tpr, threshold_roc = roc_curve(targets, probas)
        precision, recall, threshold_pr = precision_recall_curve(targets, probas)
        np.savez_compressed(roc_name, fpr=fpr, tpr=tpr, threshold=threshold_roc)
        np.savez_compressed(prc_name, precision=precision, recall=recall, threshold=np.append(threshold_pr, 1))

    def save_metrics_to_csv(self, metrics: dict, epoch: int):
        """Saves the metrics to a csv file"""
        metrics_name = os.path.join(self.run_folder, 'checkpoints', f'validation_scores_{epoch}.csv')
        with open(metrics_name, 'w') as file:
            file.write('metric,value\n')
            for key, value in metrics.items():
                file.write(f'{key},{value}\n')

    def get_val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.args.get('val_batch_size', self.args['batch_size']), 
            shuffle=self.args.get('shuffle', True), 
            collate_fn=self.args['collate_fn']
        )
    
    def get_tqdm(self, dataloader):
        return tqdm(dataloader, total=len(dataloader), file=TqdmToLogger(self.logger) if self.logger else None)
    
    def _compute_avg_metrics(self, metric_values: dict):
        """Computes the average of the metric values when metric is not zero and not NaN"""
        averages = {}
        for name, values in metric_values.items():
            values_array = np.array(values)
            select_mask = (values_array == 0) | (np.isnan(values_array))
            if select_mask.sum() > 0:
                self.log(f'Warning: {select_mask.sum()} NaN or zero values for metric {name}')
            non_zero_values = values_array[~select_mask]
            
            if non_zero_values.size:
                averages[name] = np.mean(non_zero_values)
            else:
                averages[name] = 0
        return averages

    def to_device(self, batch: dict) -> None:
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
        if self.run is not None:
            self.run.log_metric("GPU Memory Allocated in GB", memory_allocated)
            self.run.log_metric("GPU Max Memory Allocated in GB", max_memory_reserved)
            self.run.log_metric("GPU Memory Cached in GB", memory_cached)
            
        else:
            self.log(f"GPU Max Memory Allocated in GB: {max_memory_reserved}")
            self.log(f"GPU Memory Allocated in GB: {memory_allocated}")
            self.log(f"GPU Memory Cached in GB: {memory_cached}")
            

    def run_log(self, name, value):
        if self.run is not None:
            self.run.log_metric(name=name, value=value)

    def save_setup(self):
        """Saves the config and model config"""
        self.model.config.save_pretrained(self.run_folder)  
        with open(os.path.join(self.run_folder, 'pretrain_config.yaml'), 'w') as file:
            yaml.dump(self.cfg.to_dict(), file)
        self.log(f'Saved config to {self.run_folder}')  

    def save_checkpoint(self, id, **kwargs):
        """Saves a checkpoint"""
        # Model/training specific
        checkpoint_name = os.path.join(self.run_folder, 'checkpoints', f'checkpoint_{id}.pt')

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            **kwargs
        }, checkpoint_name)

    def info(self, message):
        """Prints an info message"""
        if self.args['info']:
            print(f'[INFO] {message}')