import logging
import random
from os.path import join

import numpy as np
import torch
from transformers import BertTokenizer

from models import BertBaseline, BertExperimental
from processor import TextProcessor

from trainer import ModelTrainer
from evaluator import ModelEvaluator


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_baseline(args, num_labels):
    MODEL_PATH = join(args["DATA_PATH"], "model_files/bert-base-uncased")
    config = join(MODEL_PATH, "config.json")
    model_state_dict = torch.load(join(MODEL_PATH, "pytorch_model.bin"))
    model = BertBaseline.from_pretrained(
        config, num_labels=num_labels, state_dict=model_state_dict
    )

    return model


def load_experimental(num_labels):
    DATA_PATH = "mltc/data/model_files/"
    config = DATA_PATH + "config.json"
    model_state_dict = torch.load(DATA_PATH + "finetuned_2020-08-03_pytorch_model.bin")
    del model_state_dict["classifier.weight"]
    del model_state_dict["classifier.bias"]

    model = BertExperimental.from_pretrained(
        config, num_labels=num_labels, state_dict=model_state_dict
    )

    print(model.modules)
    [print(name, param.requires_grad) for name, param in model.named_parameters()]

    return model


def load_tokenizer(args):
    tokenizer = BertTokenizer.from_pretrained(
        join(args["DATA_PATH"], "model_files/bert-base-uncased"), local_files_only=True
    )
    return tokenizer


if __name__ == "__main__":
    args = {
        "max_seq_length": 512,
        "num_train_epochs": 2,
        "train_batch_size": 12,
        "eval_batch_size": 12,
        "learning_rate": 1e-1,
        "warmup_proportion": 0.1,
        "seed": 0,
        "do_train": True,
        "do_eval": True,
        "save_checkpoints": False,
        "use_parents": True,
        "DATA_PATH": "data",
        # "DATA_PATH": "/content/gdrive/MyDrive/bert-for-hmltc/data"
        "device": "cpu",
    }

    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])

    logger.info("Initializing…")
    tokenizer = load_tokenizer(args)
    processor = TextProcessor(args, tokenizer, logger, "topic_list.json")
    model = load_baseline(args, len(processor.labels))

    if args["do_train"]:
        logger.info("Training…")
        trainer = ModelTrainer(args, processor, model, logger)
        trainer.prepare_training_data("test_raw.pkl")
        if args["do_eval"]:
            trainer.evaluator.prepare_eval_data("dev_raw.pkl")
        trainer.train()
    else:
        logger.info("Evaluating…")
        evaluator = ModelEvaluator(args, processor, model, logger)
        evaluator.prepare_eval_data("dev_raw.pkl")
        results = evaluator.evaluate()
        print(results)
