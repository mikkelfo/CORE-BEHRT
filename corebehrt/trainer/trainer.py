import os
import yaml
import torch
from collections import namedtuple

from torch.utils.data import DataLoader, Dataset
from corebehrt.common.config import Config, instantiate
from corebehrt.dataloader.collate_fn import dynamic_padding
from corebehrt.trainer.utils import (compute_avg_metrics,
                                   get_tqdm)

yaml.add_representer(Config, lambda dumper, data: data.yaml_repr(dumper))

BEST_MODEL_ID = 999 # For backwards compatibility
DEFAULT_CHECKPOINT_FREQUENCY = 100
class EHRTrainer:
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
        self.log("Initialize metrics")
        self.metrics = {k: instantiate(v) for k, v in metrics.items()} if metrics else {}
        
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
    
    def _set_default_args(self, args):
        default_args = {
            'save_every_k_steps': float('inf'),
            'collate_fn': dynamic_padding}
        self.args = {**default_args, **args}
        if not (self.args['effective_batch_size'] % self.args['batch_size'] == 0):
            raise ValueError('effective_batch_size must be a multiple of batch_size')

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
        self.validate_and_log(0, [0], dataloader)
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
        self.validate_and_log(epoch, epoch_loss, train_loop)
        torch.cuda.empty_cache()
        del train_loop
        del epoch_loss

    def _clip_gradients(self):
        # Then clip them if needed
        if self.cfg.trainer_args.get('gradient_clip', False):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.trainer_args.gradient_clip.get('max_norm', 1.0))

    def _train_step(self, batch: dict):
        self.optimizer.zero_grad()
        self.batch_to_device(batch)

        outputs = self.model(batch)
        unscaled_loss = outputs.loss
        unscaled_loss.backward()

        return unscaled_loss

    def _update_and_log(self, step_loss, train_loop, epoch_loss):
        """Updates the model and logs the loss"""
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
        val_loss, val_metrics = self._evaluate(epoch, mode='val')
        _, test_metrics = self._evaluate(epoch, mode='test')
        if epoch==1: # for testing purposes/if first epoch is best
            self._save_checkpoint(epoch, train_loss=epoch_loss, val_loss=val_loss, val_metrics=val_metrics, test_metrics=test_metrics, final_step_loss=epoch_loss[-1], best_model=True)
        if self._should_stop_early(epoch, val_loss, epoch_loss, val_metrics, test_metrics):
            return 
        self._save_checkpoint_conditionally(epoch, epoch_loss, val_loss, val_metrics, test_metrics)
        self._self_log_results(epoch, val_loss, val_metrics, epoch_loss, len(train_loop))

    def _save_checkpoint_conditionally(self, epoch: int, epoch_loss: float, val_loss: float, val_metrics,  test_metrics: dict) -> None:
        should_save = epoch % self.args.get('checkpoint_frequency', DEFAULT_CHECKPOINT_FREQUENCY) == 0
        if should_save:
            self._save_checkpoint(epoch, train_loss=epoch_loss, val_loss=val_loss, val_metrics=val_metrics, test_metrics=test_metrics, final_step_loss=epoch_loss[-1], best_model=True)

    def _self_log_results(self, epoch: int, val_loss: float, val_metrics: dict, epoch_loss: float, len_train_loop: int)->None:
        for k, v in val_metrics.items():
            self.run_log(name = k, value = v)
        self.run_log(name='Val loss', value=val_loss)
        self.log(f'Epoch {epoch} train loss: {sum(epoch_loss) / (len_train_loop / self.accumulation_steps)}')
        self.log(f'Epoch {epoch} val loss: {val_loss}')
        self.log(f'Epoch {epoch} metrics: {val_metrics}\n')

    def _should_stop_early(self, epoch, val_loss: float, epoch_loss: float, val_metrics: dict, test_metrics:dict={}) -> bool:
        if not self.early_stopping:
            return False
        # Get the current value of the metric
        current_metric_value = val_metrics.get(self.stopping_metric, val_loss)
        self._initialize_best_metric_value(current_metric_value)
        if self._is_improvement(current_metric_value):
            self.best_metric_value = current_metric_value
            self.early_stopping_counter = 0
            self._save_checkpoint(epoch, train_loss=epoch_loss, val_loss=val_loss, val_metrics=val_metrics, test_metrics=test_metrics, final_step_loss=epoch_loss[-1], best_model=True)
            return False
        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.early_stopping_patience:
                self.log("Early stopping triggered!")
                self.stop_training = True
                return True
        return False

    def _is_improvement(self, current_metric_value):
        """Returns True if the current metric value is an improvement over the best metric value"""
        if self.stopping_metric == 'val_loss':
            return current_metric_value < self.best_metric_value
        else:
            return current_metric_value > self.best_metric_value

    def _initialize_best_metric_value(self, current_metric_value: float) -> None:
        if not hasattr(self, 'best_metric_value'):
            self.best_metric_value = current_metric_value

    def setup_training(self) -> DataLoader:
        """Sets up the training dataloader and returns it"""
        self.model.train()
        self.save_setup()
        dataloader = DataLoader(self.train_dataset, batch_size=self.args['batch_size'], sampler=self.sampler,
                                shuffle=self.args.get('shuffle', True), collate_fn=self.args['collate_fn'])
        return dataloader

    def _evaluate(self, mode='val')->tuple:
        """Returns the validation/test loss and metrics"""
        if mode == 'val':
            if self.val_dataset is None:
                raise ValueError("Validation dataset is None")
            dataloader = self.get_dataloader(self.val_dataset, self.args.get('val_batch_size'))
        elif mode == 'test':
            if self.test_dataset is None:
                raise ValueError("Test dataset is None")
            dataloader = self.get_dataloader(self.test_dataset, self.args.get('test_batch_size'))
        else:
            raise ValueError(f"Mode {mode} not supported. Use 'val' or 'test'")
        
        self.model.eval()
        loop = get_tqdm(dataloader)
        loop.set_description(mode)
        loss = 0
        
        metric_values = {name: [] for name in self.metrics}
        logits_list = [] if self.accumulate_logits else None
        targets_list = [] if self.accumulate_logits else None

        with torch.no_grad():
            for batch in loop:
                self.batch_to_device(batch)
                outputs = self.model(batch)
                loss += outputs.loss.item()

                if self.accumulate_logits:
                    logits_list.append(outputs.logits.cpu())
                    targets_list.append(batch['target'].cpu())
                else:
                    for name, func in self.metrics.items():
                        metric_values[name].append(func(outputs, batch))

        if self.accumulate_logits:
            metric_values = self.process_binary_classification_results(logits_list, targets_list)
        else:
            metric_values = compute_avg_metrics(metric_values)
        
        self.model.train()
        
        return loss / len(loop), metric_values

    def process_binary_classification_results(self, logits:list, targets:list)->dict:
        """Process results specifically for binary classification."""
        batch = {'target': torch.cat(targets)}
        outputs = namedtuple('Outputs', ['logits'])(torch.cat(logits))
        metrics = {}
        for name, func in self.metrics.items():
            v = func(outputs, batch)
            self.log(f"{name}: {v}")
            metrics[name] = v
        return metrics
    
    def get_dataloader(self, dataset, batch_size=None) -> DataLoader:
        return DataLoader(
            dataset, 
            batch_size=batch_size if batch_size is not None else self.args['batch_size'], 
            shuffle=False, 
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

    def _save_checkpoint(self, epoch:int, best_model=False, **kwargs)->None:
        """Saves a checkpoint. Model with optimizer and scheduler if available."""
        # Model/training specific
        id = epoch if not best_model else BEST_MODEL_ID
        os.makedirs(os.path.join(self.run_folder, 'checkpoints'), exist_ok=True)
        checkpoint_name = os.path.join(self.run_folder, 'checkpoints', f'checkpoint_epoch{id}_end.pt')
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