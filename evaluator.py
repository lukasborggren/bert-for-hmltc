from datetime import date

import numpy as np
from sklearn.metrics import (
    roc_curve,
    auc,
)

import torch
from torch.utils.data import DataLoader

from metrics import accuracy_thresh, fbeta, pairwise_confusion_matrix

import pandas as pd
from tqdm import tqdm


class ModelEvaluator:
    """Class for evaluating and testing the text classification models.
    Evaluation is done with labeled data whilst testing/prediction is done
    with unlabeled data.
    """

    def __init__(self, args, processor, model, logger):
        self.args = args
        self.processor = processor
        self.model = model
        self.logger = logger

        self.device = "cpu"
        self.eval_dataloader: DataLoader

    def prepare_eval_data(self, file_name, parent_labels=None):
        """Creates a PyTorch Dataloader from a CSV file, which is used
        as input to the classifiers.
        """
        eval_examples = self.processor.get_examples(file_name, "eval", parent_labels)
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
        all_logits = None
        all_labels = None

        self.model.eval()
        eval_loss, eval_accuracy, eval_f1, eval_prec, eval_rec = 0, 0, 0, 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        for batch in self.eval_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, parent_labels = batch

            with torch.no_grad():
                # parent_labels is of boolean type if there are no parent labels
                if parent_labels.dtype != torch.bool:
                    outputs = self.model(
                        input_ids,
                        segment_ids,
                        input_mask,
                        label_ids,
                        parent_labels=parent_labels,
                    )
                else:
                    outputs = self.model(input_ids, segment_ids, input_mask, label_ids)

                tmp_eval_loss, logits = outputs[:2]

            tmp_eval_accuracy = accuracy_thresh(logits, label_ids)
            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy
            f1, prec, rec = fbeta(logits, label_ids)
            eval_f1 += f1
            eval_prec += prec
            eval_rec += rec

            if all_logits is None:
                all_logits = logits.detach().cpu().numpy()
            else:
                all_logits = np.concatenate(
                    (all_logits, logits.detach().cpu().numpy()), axis=0
                )

            if all_labels is None:
                all_labels = label_ids.detach().cpu().numpy()
            else:
                all_labels = np.concatenate(
                    (all_labels, label_ids.detach().cpu().numpy()), axis=0
                )

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        eval_f1 = eval_f1 / nb_eval_steps
        eval_prec = eval_prec / nb_eval_steps
        eval_rec = eval_rec / nb_eval_steps

        # ROC-AUC calcualation
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        confusion_matrices = []

        for i in range(len(self.processor.labels)):
            fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_logits[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

            confusion_matrices += [
                pairwise_confusion_matrix(
                    all_logits[:, [13, i]], all_labels[:, [13, i]]
                )
            ]

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(
            all_labels.ravel(), all_logits.ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        result = {
            "eval_loss": eval_loss,
            "eval_accuracy": eval_accuracy,
            "roc_auc": roc_auc,
            "eval_f1": eval_f1,
            "eval_prec": eval_prec,
            "eval_rec": eval_rec,
            # "confusion_matrices": confusion_matrices,
        }

        self.save_result(result)
        return result

    def save_result(self, result):
        """Saves the evaluation results as a text file."""
        d = date.today().strftime("%Y-%m-%d")
        output_eval_file = f"mltc/data/results/eval_results_{d}.txt"
        with open(output_eval_file, "w") as writer:
            self.logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                self.logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    def predict(self, file_name):
        """Makes class predicitons for unlabeled data.
        Returns the estimated probabilities for each of the labels.
        """
        test_examples = self.processor.get_examples(file_name, "test")
        test_features = self.processor.convert_examples_to_features(
            test_examples, self.args["max_seq_length"]
        )
        test_dataloader = self.processor.pack_features_in_dataloader(
            test_features, self.args["eval_batch_size"], "test"
        )

        # Hold input data for returning it
        input_data = [
            {"id": input_example.guid, "text": input_example.text_a}
            for input_example in test_examples
        ]

        self.logger.info("***** Running prediction *****")
        self.logger.info("  Num examples = %d", len(test_examples))
        self.logger.info("  Batch size = %d", self.args["eval_batch_size"])

        all_logits = None
        self.model.eval()

        for step, batch in enumerate(
            tqdm(test_dataloader, desc="Prediction Iteration")
        ):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids = batch

            with torch.no_grad():
                outputs = self.model(input_ids, segment_ids, input_mask)
                logits = outputs[0]
                logits = logits.sigmoid()

            if all_logits is None:
                all_logits = logits.detach().cpu().numpy()
            else:
                all_logits = np.concatenate(
                    (all_logits, logits.detach().cpu().numpy()), axis=0
                )

        return pd.merge(
            pd.DataFrame(input_data),
            pd.DataFrame(all_logits),
            left_index=True,
            right_index=True,
        )
