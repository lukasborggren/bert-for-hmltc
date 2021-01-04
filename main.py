import logging
import os
from os.path import join
import random

import numpy as np
import torch
from transformers import BertTokenizer

from evaluator import ModelEvaluator
from models import BertBigBang, BertIterative
from processor import TextProcessor
from trainer import ModelTrainer


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def create_baseline(args, num_labels):
    MODEL_PATH = join(args["DATA_PATH"], "model_files/bert-base-uncased")
    config = join(MODEL_PATH, "config.json")
    model_state_dict = torch.load(join(MODEL_PATH, "pytorch_model.bin"))
    model = BertBigBang.from_pretrained(
        config, num_labels=num_labels, state_dict=model_state_dict
    )

    return model


def create_experimental(args, num_labels):
    MODEL_PATH = join(args["DATA_PATH"], "model_files/bert-base-uncased")
    config = join(MODEL_PATH, "config.json")
    model_state_dict = torch.load(join(MODEL_PATH, "pytorch_model.bin"))
    model = BertIterative.from_pretrained(
        config, num_labels=num_labels, state_dict=model_state_dict
    )

    return model


def load_tokenizer(args):
    tokenizer = BertTokenizer.from_pretrained(
        join(args["DATA_PATH"], "model_files/bert-base-uncased"), local_files_only=True
    )
    return tokenizer


def prepare_data(args, processor, file_name, set_type):
    examples = processor.get_examples(file_name, set_type)

    num_train_steps = None
    if set_type == "train":
        num_train_steps = int(
            len(examples) / args["batch_size"] * args["num_train_epochs"]
        )

    features = processor.convert_examples_to_features(examples)
    dataloader = processor.pack_features_in_dataloader(features, set_type)

    return dataloader, num_train_steps


if __name__ == "__main__":
    args = {
        "max_seq_length": 350,
        "num_train_epochs": 4,
        "batch_size": 26,
        "learning_rate": 5e-5,
        "threshold": 0.5,
        "warmup_proportion": 0.1,
        "seed": 0,
        "do_train": False,
        "do_eval": True,
        "save_checkpoints": True,
        "use_parents": True,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "DATA_PATH": str,
        "session_num": 13,
    }

    if os.environ["HOME"] == "/root":
        args["DATA_PATH"] = "/content/gdrive/MyDrive/bert-for-hmltc/data"
    else:
        args["DATA_PATH"] = "data"

    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])

    logger.info("Initializing…")
    tokenizer = load_tokenizer(args)
    processor = TextProcessor(args, tokenizer, logger, "topic_list.json")

    if args["use_parents"]:
        model = create_experimental(args, len(processor.labels))
    else:
        model = create_baseline(args, len(processor.labels))

    model_state_dict = torch.load(
        join(args["DATA_PATH"], "model_files/13_finetuned_pytorch_model.bin"),
        map_location="cpu",
    )
    model.load_state_dict(model_state_dict)
    if args["do_train"]:
        trainer = ModelTrainer(args, model, logger)

        logger.info("Loading data…")
        trainer.dataloader, trainer.num_train_steps = prepare_data(
            args, processor, "train_ext.pkl", "train"
        )
        if args["do_eval"]:
            trainer.evaluator.dataloader, _ = prepare_data(
                args, processor, "dev_raw.pkl", "dev"
            )

        logger.info("Training…")
        trainer.train()

    else:
        evaluator = ModelEvaluator(args, model, logger)
        logger.info("Loading data…")
        evaluator.dataloader, _ = prepare_data(args, processor, "test_raw.pkl", "dev")
        logger.info("Evaluating…")
        result = evaluator.evaluate()
