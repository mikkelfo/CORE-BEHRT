import os

import torch
import yaml
from common.config import Config, get_function, instantiate
from common.logger import TqdmToLogger
from dataloader.collate_fn import dynamic_padding
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
        run = None
    ):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if logger:
            logger.info(f"Run on {self.device}")
        self.run_folder = os.path.join(cfg.paths.output_path, cfg.paths.run_name)
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        if metrics:
            self.metrics = {k: instantiate(v) for k, v in metrics.items()}
        else:
            self.metrics = {}
        
        self.sampler = sampler
        self.cfg = cfg
        self.logger = logger
        self.run = run
        if self.cfg.trainer_args.get('mixed_precision', False):#
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
        self.update_attributes(**kwargs)
        self.validate_training()

        self.accumulation_steps: int = self.args['effective_batch_size'] // self.args['batch_size']
        dataloader = self.setup_training()

        for epoch in range(self.args['epochs']):
            self.train_epoch(epoch, dataloader)

    def train_epoch(self, epoch: int, dataloader: DataLoader):
        train_loop = tqdm(enumerate(dataloader), total=len(dataloader), file=TqdmToLogger(self.logger) if self.logger else None)
        train_loop.set_description(f'Train {epoch}')
        epoch_loss = []
        step_loss = 0
        for i, batch in train_loop:
            step_loss += self.train_step(batch).item()
            #if i<50:
             #   self.run_log_gpu()
            if (i+1) % self.accumulation_steps == 0:
                self.clip_gradients()
                self.update_and_log(i, step_loss, train_loop, epoch_loss)
                step_loss = 0
            if ((i+1) / self.accumulation_steps) % self.args['save_every_k_steps'] == 0:
                self.save_checkpoint(id=f'epoch{epoch}_step{(i+1) // self.accumulation_steps}', train_loss=step_loss / self.accumulation_steps)
        self.validate_and_log(epoch, epoch_loss, train_loop)

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
            self.log(f'Train loss {(i+1) // self.accumulation_steps}: {step_loss / self.accumulation_steps}')
            for param_group in self.optimizer.param_groups:
                current_lr = param_group['lr']
                self.run_log('Learning Rate', current_lr)
                break
        self.run_log('Train loss', step_loss / self.accumulation_steps)
        

    def validate_and_log(self, epoch, epoch_loss, train_loop):
        val_loss, metrics = self.validate()
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
        self.save_pids()
        dataloader = DataLoader(self.train_dataset, batch_size=self.args['batch_size'], shuffle=False, collate_fn=self.args['collate_fn'])
        return dataloader

    def forward_pass(self, batch: dict):
        self.to_device(batch)
        return self.model(
            input_ids=batch['concept'],
            attention_mask=batch['attention_mask'] if 'attention_mask' in batch else None,
            token_type_ids=batch['segment'] if 'segment' in batch else None,
            position_ids={
                'age': batch['age'] if 'age' in batch else None,
                'abspos': batch['abspos'] if 'abspos' in batch else None,
                'dosage': batch['dosage'] if 'dosage' in batch else None,
                'unit': batch['unit'] if 'unit' in batch else None,
            },
            labels=batch['target'] if 'target' in batch else None,
        )

    def backward_pass(self, loss):
        loss.backward()

    def validate(self):
        """Returns the validation loss and metrics"""
        if self.val_dataset is None:
            self.log('No validation dataset provided')
            return None, None
        
        self.model.eval()
        dataloader = DataLoader(self.val_dataset, batch_size=self.args['batch_size'], shuffle=False, collate_fn=self.args['collate_fn'], sampler=self.sampler)
        val_loop = tqdm(dataloader, total=len(dataloader), file=TqdmToLogger(self.logger) if self.logger else None)
        val_loop.set_description('Validation')
        val_loss = 0
        
        metric_values = {name: [] for name in self.metrics}
        with torch.no_grad():
            for batch in val_loop:
                outputs = self.forward_pass(batch)
                val_loss += outputs.loss.item()
                for name, func in self.metrics.items():
                    metric_values[name].append(func(outputs, batch))

        self.model.train()
        
        return val_loss / len(val_loop), {name: sum(values) / len(values) for name, values in metric_values.items()}

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
        # Log the GPU memory usage
        if self.run is not None:
            memory_allocated = torch.cuda.memory_allocated(device=self.device)/1e9
            self.run.log_metric("GPU Memory Allocated", memory_allocated)

            memory_cached = torch.cuda.memory_cached(device=self.device)/1e9
            self.run.log_metric("GPU Memory Cached", memory_cached)

    def run_log(self, name, value):
        if self.run is not None:
            self.run.log_metric(name=name, value=value)

    def save_setup(self):
        """Saves the config and model config"""
        self.model.config.save_pretrained(self.run_folder)  
        with open(os.path.join(self.run_folder, 'pretrain_config.yaml'), 'w') as file:
            yaml.dump(self.cfg.to_dict(), file)
        self.log(f'Saved config to {self.run_folder}')
       
        self.train_dataset.save_vocabulary(self.run_folder)
        self.log(f'Saved vocabulary to {self.run_folder}')
       
    def save_pids(self):
        """Saves the pids of the train, val and test datasets"""
        try:
            torch.save(self.train_dataset.pids, os.path.join(self.run_folder, 'train_pids.pt'))
            torch.save(self.val_dataset.pids, os.path.join(self.run_folder, 'val_pids.pt'))
            torch.save(self.train_dataset.file_ids, os.path.join(self.run_folder, 'train_file_ids.pt'))
            torch.save(self.val_dataset.file_ids, os.path.join(self.run_folder, 'val_file_ids.pt'))            
            if self.test_dataset:
                torch.save(self.test_dataset.pids, os.path.join(self.run_folder, 'test_pids.pt'))
            self.log(f'Copied pids to {self.run_folder}')
        except AttributeError:
            self.log("Failed to save pids")
            

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