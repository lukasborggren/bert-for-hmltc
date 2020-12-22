import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score
import torch
from torch.utils.data import DataLoader


def accuracy_score(true, pred):
    return np.mean(np.mean((pred == true), axis=1))


class ModelEvaluator:
    """Class for evaluating the text classification models."""

    def __init__(self, args, model, logger):
        self.args = args
        self.model = model
        self.logger = logger

        self.dataloader: DataLoader

    def _handle_out(self, outputs):
        tmp_loss, logits = outputs[:2]
        logits = logits.sigmoid()
        logits = (logits > 0.5).float()
        return tmp_loss, logits

    def evaluate(self):
        """Evaluates a classifier using labeled data.
        Calculates and returns accuracy, precision, recall F1 score and ROC AUC.
        """
        loss = []
        pred, labels = None, None
        self.model.eval()

        for batch in self.dataloader:
            batch = tuple(t.to(self.args["device"]) for t in batch)

            if self.args["use_parents"]:
                input_ids, input_mask, segment_ids, label_ids, parent_labels = batch
                for i_ in range(4):
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids,
                            segment_ids,
                            input_mask,
                            label_ids,
                            parent_labels=parent_labels,
                        )
                    tmp_loss, logits = self._handle_out(outputs)
                    parent_labels = logits
            else:
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.no_grad():
                    outputs = self.model(input_ids, segment_ids, input_mask, label_ids)
                tmp_loss, logits = self._handle_out(outputs)

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

        f1 = f1_score(labels, pred, average="micro", zero_division=0)
        f1_macro = f1_score(labels, pred, average="macro", zero_division=0)
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
