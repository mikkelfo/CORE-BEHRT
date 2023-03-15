from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import os
import uuid
import json


class EHRTrainer():
    def __init__(self, 
        model: torch.nn.Module,
        train_dataset: Dataset = None,
        test_dataset: Dataset = None,
        eval_dataset: Dataset = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler.StepLR = None,
        metrics: dict = {},
        args: dict = None,
    ):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.eval_dataset = eval_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics

        if args is None:
            self.info('No args provided, using default args')
            args = {
                'batch_size': 4,
                'accumulate_grad_batches': 1,
                'epochs': 1,
                'info': True,
                'save_every': 1,
            }
        self.args = args

    def loop(self, mode: str):
        self.info(f'Starting {mode} loop')

        self.model.train() if mode == 'train' else self.model.eval()
        dataloader = DataLoader(self.get_dataset(mode), batch_size=self.args['batch_size'], shuffle=True, collate_fn=self.args.get('collate_fn'))
        step = self.get_step(mode)
        
        self.setup_run_folder()
        self.save_setup()
        for epoch in range(self.args['epochs']):
            self.epoch_idx = epoch
            self.initialize_loop(dataloader)
            self.current_loop.set_description(f'{mode.upper()} {epoch}')

            for batch in self.current_loop:
                self.batch_idx += 1
                step(batch)

            if (epoch + 1) % self.args.get('save_every', 1) == 0: 
                self.save_checkpoint()

            self.finalize()

    def forward_pass(self, batch: dict):
        batch = self.to_device(batch)
        return self.model(
            input_ids=batch['concept'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['segment'] if 'segment' in batch else None,
            position_ids={
                'age': batch['age'] if 'age' in batch else None,
                'abspos': batch['abspos'] if 'abspos' in batch else None
            },
            labels=batch['target'] if 'target' in batch else None
        )

    def backward_pass(self, loss: torch.Tensor):
        loss.backward()
        if self.batch_idx % self.args['accumulate_grad_batches'] == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler is not None:
                self.scheduler.step()

    def train_step(self, batch: dict):       
        # Forward pass
        outputs = self.forward_pass(batch)
        
        # Backward pass and optimizer step
        self.backward_pass(outputs.loss)

        # Logging
        self.log_dict({
            'train_loss': outputs.loss.item(),
        })

    def test_step(self, batch: dict):
        # Forward pass
        with torch.no_grad():
            outputs = self.forward_pass(batch)

        # Logging
        self.log_dict({
            'test_loss': outputs.loss.item(),
        })
        self.log_dict({
            name: func(outputs, batch) for name, func in self.metrics.items()
        }, on_step=False, on_epoch=True, on_progressbar=False)

    def eval_step(self, batch: dict):
        # Forward pass
        with torch.no_grad():
            outputs = self.forward_pass(batch)
        
        # Logging
        self.log_dict({
            name: func(outputs, batch) for name, func in self.metrics.items()
        }, on_step=False, on_epoch=True, on_progressbar=False)

    def setup_run_folder(self):
        # Generate unique run_name if not provided
        if self.args.get('run_name') is None:
            random_runname = uuid.uuid4().hex
            self.info(f'Run name not provided. Using random run name: {random_runname}')
            self.run_folder = os.path.join('runs', random_runname)
        else:
            self.run_folder = os.path.join('runs', self.args['run_name'])

        if not os.path.exists(self.run_folder):
            os.mkdirs(self.run_folder)
        if os.path.exists(self.run_folder):
            self.info(f'Run folder {self.run_folder} already exists. Writing files to existing folder')

        self.info(f'Run folder: {self.run_folder}')

    def info(self, message: str):
        if self.args['info']:
            tqdm.write(message)

    def log_dict(self, d: dict, on_step=True, on_epoch=True, on_progressbar=True):
        for name, value in d.items():
            self.log(name, value, on_step, on_epoch, on_progressbar)

    def log(self, name: str, value: float, on_step=True, on_epoch=True, on_progressbar=True):
        if name not in self.metric_values:
            self.metric_values[name] = []
        self.metric_values[name].append(value)

        if on_progressbar:
            self.current_loop.set_postfix(loss=value)

        if on_step and self.batch_idx % self.args['accumulate_grad_batches'] == 0:
            N = self.batch_idx // self.args['accumulate_grad_batches']
            avg_value = sum(self.metric_values[name][-N:]) / N
            self.info(f'Step {name}: {avg_value}')

        if on_epoch and self.batch_idx == len(self.current_loop):
            avg_value = sum(self.metric_values[name]) / len(self.metric_values[name])
            self.info(f'Epoch {name}: {avg_value}')
    
    def get_step(self, mode: str) -> callable:
        if mode == 'train':
            return self.train_step
        elif mode == 'test':
            return self.test_step
        elif mode == 'eval':
            return self.eval_step

    def get_dataset(self, mode: str) -> Dataset:
        if mode == 'train':
            return self.train_dataset
        elif mode == 'test':
            return self.test_dataset
        elif mode == 'eval':
            return self.eval_dataset

    def train(self):
        self.loop('train')
    
    def test(self):
        self.args['epochs'] = 1
        self.loop('test')

    def eval(self):
        self.args['epochs'] = 1
        self.loop('eval') 

    def initialize_loop(self, loop: DataLoader):
        self.current_loop = tqdm(loop)
        self.batch_idx = 0
        self.metric_values = {}

    def finalize(self):
        self.current_loop.close()
        self.current_loop = None
        self.batch_idx = 0

    def to_device(self, batch: dict) -> dict:
        return {key: value.to(self.device) for key, value in batch.items()}

    def save_setup(self):
        config_name = os.path.join(self.run_folder, 'config.json')
        with open(config_name, 'w') as f:
            json.dump({
                'model_config': self.model.config.to_dict(),    # Maybe .to_diff_dict()?
                'args': self.args
            }, f)

    def save_checkpoint(self):
        # Model/training specific
        checkpoint_name = os.path.join(self.run_folder, f'checkpoint_{self.current_epoch}.pt')
        torch.save({
            'epoch': self.current_epoch,
            'batch_idx': self.batch_idx,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'metric_values': self.metric_values,
        }, checkpoint_name)

