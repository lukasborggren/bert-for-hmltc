from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm


class ModelTrainer:
    """Class for training a classifier."""

    def __init__(self, args, processor, model, logger):
        self.args = args
        self.processor = processor
        self.model = model
        self.logger = logger

        self.device = "cpu"
        self.train_dataloader: DataLoader
        self.num_train_steps: int

        self.optimizer: AdamW
        self.scheduler: LambdaLR

    def prepare_training_data(self, file_name):
        """Creates a PyTorch Dataloader from a CSV file, which is used
        as input to the classifiers.
        """
        train_examples = self.processor.get_examples(file_name, "train")

        self.num_train_steps = int(
            len(train_examples)
            / self.args["train_batch_size"]
            * self.args["num_train_epochs"]
        )

        train_features = self.processor.convert_examples_to_features(
            train_examples,
            self.args["max_seq_length"],
        )
        self.train_dataloader = self.processor.pack_features_in_dataloader(
            train_features,
            self.args["train_batch_size"],
            "train",
        )

    def load_optimizer(self):
        """Loads the AdamW optimizer used during training."""
        param_optimizer = list(self.model.named_parameters())

        # Weight decay denotes the lambda in an L2 penalty added to the loss
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args["learning_rate"],
            correct_bias=False,
        )

    def load_scheduler(self):
        """Loads the scheduler which controls the learning rate during training."""
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args["warmup_proportion"] * self.num_train_steps,
            num_training_steps=self.num_train_steps,
        )

    def train(self):
        """Performs model training using labeled data."""
        self.model.to(self.device)
        self.load_optimizer()
        self.load_scheduler()

        global_step = 0
        self.model.train()

        for i_ in tqdm(range(int(self.args["num_train_epochs"])), desc="Epoch"):
            tr_loss, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, parent_labels = batch

                # Forward pass, compute loss for prediction
                outputs = self.model(input_ids, segment_ids, input_mask, label_ids)
                loss = outputs[0]

                # Backward pass, compute gradient of loss w.r.t. model parameters
                loss.backward()
                tr_loss += loss.item()
                nb_tr_steps += 1

                self.optimizer.step()  # Update model parameters
                self.scheduler.step()  # Update learning rate schedule
                self.optimizer.zero_grad()  # Set gradients of model parameters to zero
                global_step += 1

            self.logger.info(f"Loss after epoch {i_+1}: {tr_loss / nb_tr_steps}")
            self.logger.info(
                f"Learning rate after epoch {i_+1}: {self.scheduler.get_last_lr()[0]}"
            )

        self.model.save()
