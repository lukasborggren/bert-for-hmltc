import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

import torch
from torch.utils.data import DataLoader


class ModelEvaluator:
    """Class for evaluating the text classification models."""

    def __init__(self, args, processor, model, logger):
        self.args = args
        self.processor = processor
        self.model = model
        self.logger = logger

        self.eval_dataloader: DataLoader

    def prepare_eval_data(self, file_name):
        """Creates a PyTorch Dataloader from a CSV file, which is used
        as input to the classifiers.
        """
        eval_examples = self.processor.get_examples(file_name, "eval")
        eval_features = self.processor.convert_examples_to_features(
            eval_examples, self.args["max_seq_length"]
        )
        self.eval_dataloader = self.processor.pack_features_in_dataloader(
            eval_features, self.args["eval_batch_size"], "eval"
        )

    def evaluate(self):
        """Evaluates a classifier using labeled data.
        Calculates and returns accuracy, precision, recall F1 score and ROC AUC.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss = []
        pred, labels = None, None
        self.model.eval()

        for batch in self.eval_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.no_grad():
                outputs = self.model(input_ids, segment_ids, input_mask, label_ids)
                tmp_loss, logits = outputs[:2]

            logits = logits.sigmoid()
            logits = (logits > 0.5).float()

            if pred is None:
                pred = logits.detach().cpu().numpy()
            else:
                pred = np.concatenate((pred, logits.detach().cpu().numpy()), axis=0)

            if labels is None:
                labels = label_ids.detach().cpu().numpy()
            else:
                labels = np.concatenate(
                    (labels, label_ids.detach().cpu().numpy()), axis=0
                )

            loss.append(tmp_loss.mean().item())

        f1 = f1_score(labels, pred, average="micro")
        f1_macro = f1_score(labels, pred, average="macro")
        recall = recall_score(labels, pred, average="micro")
        precision = precision_score(labels, pred, average="micro")
        accuracy = accuracy_score(labels, pred)

        result = {
            "loss": sum(loss) / len(loss),
            "accuracy": accuracy,
            "f1": f1,
            "f1_macro": f1_macro,
            "precision": precision,
            "recall": recall,
        }

        # with open("data/topic_list.json", "r") as f:
        #     topic_list = json.load(f)

        # for row in pred:
        #     topics = [topic_list[ind] for ind, ohe in enumerate(row) if ohe == 1.0]
        #     print(len(topics))

        # self.save_result(result)
        return result

    def save_result(self, result):
        """Saves the evaluation results as a text file."""
        output_eval_file = "data/results/eval_results.txt"
        with open(output_eval_file, "w") as writer:
            self.logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                self.logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
