import os
import json
import uuid
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from hydra.utils import instantiate
from torch.utils.data import Dataset, DataLoader
from src.dataloader.collate_fn import dynamic_padding


class EHRTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: Dataset = None,
        test_dataset: Dataset = None,
        val_dataset: Dataset = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler.StepLR = None,
        metrics: dict = {},
        args: dict = {},
        sampler=None,
        cfg=None,
    ):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print("Running on:", self.device)

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

        default_args = {
            "save_every_k_steps": float("inf"),
            "collate_fn": dynamic_padding,
        }

        self.args = {**default_args, **args}

    def update_attributes(self, **kwargs):
        for key, value in kwargs.items():
            if key == "args":
                self.args = {**self.args, **value}
            else:
                setattr(self, key, value)

    def validate_training(self):
        assert self.model is not None, "No model provided"
        assert self.train_dataset is not None, "No training dataset provided"
        assert self.optimizer is not None, "No optimizer provided"

    def train(self, **kwargs):
        self.update_attributes(**kwargs)
        self.validate_training()

        accumulation_steps: int = (
            self.args["effective_batch_size"] // self.args["batch_size"]
        )
        dataloader = self.setup_training()

        for epoch in range(self.args["epochs"]):
            train_loop = tqdm(enumerate(dataloader), total=len(dataloader))
            train_loop.set_description(f"Train {epoch}")
            epoch_loss = []
            step_loss = 0
            for i, batch in train_loop:
                # Train step
                step_loss += self.train_step(batch).item()

                # Accumulate gradients
                if (i + 1) % accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler is not None:
                        self.scheduler.step()

                    train_loop.set_postfix(loss=step_loss / accumulation_steps)
                    if self.args["info"]:
                        tqdm.write(
                            f"Train loss {(i+1) // accumulation_steps}: {step_loss / accumulation_steps}"
                        )
                    epoch_loss.append(step_loss / accumulation_steps)
                    step_loss = 0

                # Save iteration checkpoint
                if ((i + 1) / accumulation_steps) % self.args[
                    "save_every_k_steps"
                ] == 0:
                    self.save_checkpoint(
                        id=f"epoch{epoch}_step{(i+1) // accumulation_steps}",
                        train_loss=step_loss / accumulation_steps,
                    )

            # Validate (returns None if no validation set is provided)
            val_loss, metrics = self.validate()

            # Save epoch checkpoint
            self.save_checkpoint(
                id=f"epoch{epoch}_end",
                train_loss=epoch_loss,
                val_loss=val_loss,
                metrics=metrics,
                final_step_loss=epoch_loss[-1] if epoch_loss else None,
            )

            # Print epoch info
            tqdm.write(
                f"Epoch {epoch} train loss: {sum(epoch_loss) / (len(train_loop) / accumulation_steps)}"
            )
            tqdm.write(f"Epoch {epoch} val loss: {val_loss}")
            tqdm.write(f"Epoch {epoch} metrics: {metrics}\n")

    def setup_training(self) -> tqdm:
        self.model.train()
        self.setup_run_folder()
        self.save_setup()

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=self.sampler is None,
            collate_fn=self.args["collate_fn"],
            sampler=self.sampler,
        )
        return dataloader

    def train_step(self, batch: dict):
        outputs = self.forward_pass(batch)
        self.backward_pass(outputs.loss)

        return outputs.loss

    def forward_pass(self, batch: dict):
        self.to_device(batch)  # Moves batch to device in-place
        return self.model(batch)

    def backward_pass(self, loss):
        loss.backward()

    def validate(self):
        if self.val_dataset is None:
            return None, None

        self.model.eval()
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,
            collate_fn=self.args["collate_fn"],
        )
        val_loop = tqdm(dataloader, total=len(dataloader))
        val_loop.set_description("Validation")
        val_loss = 0
        metric_values = {name: [] for name in self.metrics}
        with torch.no_grad():
            for batch in val_loop:
                outputs = self.forward_pass(batch)
                val_loss += outputs.loss.item()

                for name, func in self.metrics.items():
                    metric_values[name].append(func(outputs, batch))

        self.model.train()
        return val_loss / len(val_loop), {
            name: sum(values) / len(values) for name, values in metric_values.items()
        }

    def to_device(self, batch: dict) -> None:
        for key, value in batch.items():
            batch[key] = value.to(self.device)

    def setup_run_folder(self):
        # Generate unique run_name if not provided
        if self.args.get("run_name") is None:
            random_runname = uuid.uuid4().hex
            self.info(f"Run name not provided. Using random run name: {random_runname}")
            self.args["run_name"] = random_runname
        self.run_folder = os.path.join("runs", self.args["run_name"])

        if os.path.exists(self.run_folder):
            self.info(
                f"Run folder {self.run_folder} already exists. Writing files to existing folder"
            )
        if not os.path.exists(self.run_folder):
            os.makedirs(self.run_folder)

        self.info(f"Run folder: {self.run_folder}")

    def save_setup(self):
        config_name = os.path.join(self.run_folder, "config.json")
        with open(config_name, "w") as f:
            json.dump(
                {
                    "model_config": self.model.config.to_dict(),  # Maybe .to_diff_dict()?
                    "cfg": OmegaConf.to_container(self.cfg, resolve=True)
                    # 'args': self.args
                },
                f,
            )

    def save_checkpoint(self, id, **kwargs):
        # Model/training specific
        checkpoint_name = os.path.join(self.run_folder, f"checkpoint_{id}.pt")

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict()
                if self.scheduler is not None
                else None,
                **kwargs,
            },
            checkpoint_name,
        )

    def info(self, message):
        if self.args["info"]:
            print(f"[INFO] {message}")
