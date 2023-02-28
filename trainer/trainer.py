from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import os
import uuid
import json

from evaluation.mlm import top_k


class EHRTrainer():
    def __init__(self, 
        model: torch.nn.Module,
        train_dataset: Dataset = None,
        test_dataset: Dataset = None,
        eval_dataset: Dataset = None,
        optimizer: torch.optim.optimizer.Optimizer = None,
        scheduler: torch.optim.lr_scheduler.StepLR = None,
        args: dict = None,
    ):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.eval_dataset = eval_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.args = args

    def loop(self, mode: str):
        self.info(f'Starting {mode} loop')

        self.model.train() if mode == 'train' else self.model.eval()
        dataloader = DataLoader(self.get_dataset[mode], batch_size=self.args['batch_size'], shuffle=True)

        self.initialize_loop(dataloader)
        self.current_loop.set_description(mode.capitalize())

        step = self.get_step(mode)
        for batch in self.current_loop:
            self.batch_idx += 1
            step(batch)
        self.save()
        self.finalize()

    def forward_pass(self, batch: dict[str, torch.Tensor]):
        batch.to(self.device)
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

    def train_step(self, batch: dict[str, torch.Tensor]):       
        # Forward pass
        outputs = self.forward_pass(batch)
        
        # Backward pass and optimizer step
        self.backward_pass(outputs.loss)

        # Logging
        self.log_dict({
            'train_loss': outputs.loss.item(),
        })

    def test_step(self, batch: dict[str, torch.Tensor]):
        # Forward pass
        with torch.no_grad():
            outputs = self.forward_pass(batch)

        # Logging
        self.log_dict({
            'test_loss': outputs.loss.item(),
        })
        self.log_dict({
            'test_acc_1': top_k(outputs, batch, 1),
            'test_acc_10': top_k(outputs, batch, 10),
            'test_acc_30': top_k(outputs, batch, 30),
            'test_acc_50': top_k(outputs, batch, 50),
            'test_acc_100': top_k(outputs, batch, 100),
        }, on_step=False, on_epoch=True, on_progressbar=False)

    def eval_step(self, batch: dict[str, torch.Tensor]):
        # Forward pass
        with torch.no_grad():
            outputs = self.forward_pass(batch)


    def setup_run_folder(self):
        # Generate unique run_name if not provided
        if self.args['run_name'] is None:
            self.args['run_name'] = uuid.uuid4().hex
        self.run_folder = f'runs/{self.args["run_name"]}'

        if not os.path.exists('runs'):
            os.mkdir('runs')
        if os.path.exists(self.run_folder):
            raise ValueError(f'Run folder {self.run_folder} already exists. Please choose a different run name.')
        
        os.mkdir(self.run_folder)
        self.info(f'Run folder: {self.run_folder}')

    def info(self, message: str):
        if self.args['info']:
            print(message)

    def log_dict(self, d: dict[str, float], on_step=True, on_epoch=True, on_progressbar=True):
        for name, value in d.items():
            self.log(name, value, on_step, on_epoch, on_progressbar)

    def log(self, name: str, value: float, on_step=True, on_epoch=True, on_progressbar=True):
        if name not in self.metric_values:
            self.metric_values[name] = []
        self.metric_values[name].append(value)

        if on_progressbar:
            self.current_loop.set_postfix(name=value)

        if on_step and self.batch_idx % self.args['accumulate_grad_batches'] == 0:
            N = self.batch_idx / self.args['accumulate_grad_batches']
            avg_value = sum(self.metric_values[name][-N:]) / N
            self.info(f'{name}: {avg_value}')

        if on_epoch and self.batch_idx == len(self.current_loop):
            avg_value = sum(self.metric_values[name]) / len(self.metric_values[name])
            self.info(f'{name}: {avg_value}')
    
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
        self.loop('test')

    def eval(self):
        self.loop('eval') 

    def initialize_loop(self, loop: DataLoader):
        self.current_loop = tqdm(loop)
        self.batch_idx = 0

    def finalize(self):
        self.current_loop.close()
        self.current_loop = None
        self.batch_idx = 0

    def save(self):
        # Model/training specific
        torch.save(self.model.state_dict(), f'{self.run_folder}/model.pt')
        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), f'{self.run_folder}/optimizer.pt')
        if self.scheduler is not None:
            torch.save(self.scheduler.state_dict(), f'{self.run_folder}/scheduler.pt')

        with open(f'{self.run_folder}/args.json', 'w') as f:
            json.dump(self.args, f)

        with open(f'{self.run_folder}/model_config.json', 'w') as f:
            json.dump(self.model.config, f)

        avg_metrics = {name: sum(values) / len(values) for name, values in self.metrics.items()}
        with open(f'{self.run_folder}/metrics.json', 'w') as f:
            json.dump(avg_metrics, f)

