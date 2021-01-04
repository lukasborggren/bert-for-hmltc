from os.path import join

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from termcolor import colored
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

from evaluator import ModelEvaluator


class ModelTrainer:
    """Class for training a classifier."""

    def __init__(self, args, model, logger):
        self.args = args
        self.model = model
        self.logger = logger

        self.dataloader: DataLoader
        self.num_train_steps: int

        self.optimizer: AdamW
        self.scheduler: LambdaLR

        if args["do_eval"]:
            self.evaluator = ModelEvaluator(args, model, logger)

    def load_optimizer(self):
        """Loads the AdamW optimizer used during training."""
        param_optimizer = list(self.model.named_parameters())

        # Excluded to reproduce the behaviour of the original optimizer.
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        apply = [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
        no_apply = [p for n, p in param_optimizer if any(nd in n for nd in no_decay)]
        grouped_params = [
            {"params": apply, "weight_decay": 0.01},
            {"params": no_apply, "weight_decay": 0.0},
        ]

        self.optimizer = AdamW(
            grouped_params,
            lr=self.args["learning_rate"],
            correct_bias=False,
        )

    def load_scheduler(self):
        """Loads the scheduler controlling the learning rate during training."""
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args["warmup_proportion"] * self.num_train_steps,
            num_training_steps=self.num_train_steps,
        )

    def train(self):
        """Performs model training using labeled data."""
        self.model.to(self.args["device"])
        self.load_optimizer()
        self.load_scheduler()

        for i in range(self.args["num_train_epochs"]):
            train_loss = 0
            self.model.train()

            for step, batch in enumerate(tqdm(self.dataloader, desc="Batch")):
                batch = tuple(t.to(self.args["device"]) for t in batch)

                if self.args["use_parents"]:
                    input_ids, input_mask, segment_ids, label_ids, parent_labels = batch
                    outputs = self.model(
                        input_ids,
                        segment_ids,
                        input_mask,
                        label_ids,
                        parent_labels=parent_labels,
                    )
                else:
                    input_ids, input_mask, segment_ids, label_ids = batch
                    # Forward pass, compute loss for prediction
                    outputs = self.model(input_ids, segment_ids, input_mask, label_ids)

                loss = outputs[0]
                # Backward pass, compute gradient of loss w.r.t. model parameters
                loss.backward()
                train_loss += loss.item()

                self.optimizer.step()  # Update model parameters
                self.scheduler.step()  # Update learning rate schedule
                self.optimizer.zero_grad()  # Set gradients of model parameters to zero

            self.logger.info(colored(f"TRAINING EPOCH {i + 1} COMPLETE", "green"))
            self.logger.info(f"Training loss = {train_loss / len(self.dataloader)}")
            self.logger.info(f"Learning rate = {self.scheduler.get_last_lr()[0]}")

            if self.args["do_eval"]:
                self.evaluator.evaluate(i + 1)

            if self.args["save_checkpoints"] and 0 < i < 3:
                self._save_model(i + 1)

        self._save_model()

    def _save_model(self, epoch=None):
        """Saves a checkpoint or complete model"""
        if epoch:
            state = {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            output_model_file = join(
                self.args["DATA_PATH"],
                f'model_files/{self.args["session_num"]}_epoch_{epoch}.ckpt',
            )
        else:
            state = self.model.state_dict()
            output_model_file = join(
                self.args["DATA_PATH"],
                f'model_files/{self.args["session_num"]}_finetuned_pytorch_model.bin',
            )
        torch.save(state, output_model_file)
