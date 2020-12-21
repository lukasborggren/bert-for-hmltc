import logging
import random
from os.path import join

import numpy as np
import torch
from transformers import BertTokenizer

from models import BertBaseline, BertExperimental
from processor import MultiLabelTextProcessor

from trainer import ModelTrainer

# from evaluator import ModelEvaluator


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_baseline(num_labels, BASE_PATH):
    config = join(BASE_PATH, "config.json")
    model_state_dict = torch.load(join(BASE_PATH, "pytorch_model.bin"))

    model = BertBaseline.from_pretrained(
        config, num_labels=num_labels, state_dict=model_state_dict
    )

    return model


def load_experimental(num_labels):
    BASE_PATH = "mltc/data/model_files/"
    config = BASE_PATH + "config.json"
    model_state_dict = torch.load(BASE_PATH + "finetuned_2020-08-03_pytorch_model.bin")
    del model_state_dict["classifier.weight"]
    del model_state_dict["classifier.bias"]

    model = BertExperimental.from_pretrained(
        config, num_labels=num_labels, state_dict=model_state_dict
    )

    print(model.modules)
    [print(name, param.requires_grad) for name, param in model.named_parameters()]

    return model


def load_tokenizer(BASE_PATH):
    tokenizer = BertTokenizer.from_pretrained(BASE_PATH, local_files_only=True)
    return tokenizer


if __name__ == "__main__":
    args = {
        "max_seq_length": 512,
        "num_train_epochs": 4.0,
        "train_batch_size": 32,
        "eval_batch_size": 32,
        "learning_rate": 3e-5,
        "warmup_proportion": 0.1,
        "seed": 0,
        "do_train": True,
        "use_parents": True,
    }

    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])

    logger.info("Initializing…")
    BASE_PATH = "data/model_files/bert-base-uncased"
    tokenizer = load_tokenizer(BASE_PATH)
    processor = MultiLabelTextProcessor(tokenizer, logger, "topic_list.json")
    model = load_baseline(len(processor.labels), BASE_PATH)

    if args["do_train"]:
        logger.info("Training…")
        trainer = ModelTrainer(args, processor, model, logger)
        trainer.prepare_training_data("dev_raw.pkl")
        # trainer.train()

    # logger.info("Evaluating…")
    # evaluator = ModelEvaluator(args, processor, model, logger)
    # evaluator.prepare_eval_data("test_sub.csv", "test_sub_parent_labels.csv")
    # results = evaluator.evaluate()
    # print(results)
