from os.path import join
import pickle

import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import torch
from torch.utils.data import DataLoader


class ModelEvaluator:
    """Class for evaluating the text classification models."""

    def __init__(self, args, model, logger):
        self.args = args
        self.model = model
        self.logger = logger

        self.dataloader: DataLoader

        if args["use_parents"]:
            with open(join(args["DATA_PATH"], "children_dict.pkl"), "rb") as f:
                self.children_dict = pickle.load(f)
            self.parent_dict = {1: (0, 7), 2: (7, 59), 3: (59, 136)}

    def _handle_out(self, outputs):
        tmp_loss, logits = outputs[:2]
        logits = logits.sigmoid()
        logits = (logits > self.args["threshold"]).float()
        return tmp_loss, logits

    def _get_pred(self, i, parent_labels, logits, zeros):
        tmp = torch.clone(zeros)
        if i == 0:
            ind = self.children_dict[-1]
            tmp[:, ind[0] : ind[1]] = logits[:, ind[0] : ind[1]]
            labels = tmp.to(self.args["device"])
        else:
            ind_range = self.parent_dict[i]
            parent_ind = (parent_labels == 1.0).nonzero()
            inds = [None] * list(logits.shape)[0]
            for _, x in enumerate(parent_ind.detach().cpu().numpy()):
                in_range = ind_range[0] <= x[1] < ind_range[1]
                if x[1] in self.children_dict and in_range:
                    if inds[x[0]]:
                        inds[x[0]].append(self.children_dict[x[1]])
                    else:
                        inds[x[0]] = [self.children_dict[x[1]]]
            for x in inds:
                if x:
                    for ind in x:
                        if ind[1] == list(logits.shape)[1]:
                            tmp[:, ind[0] :] = logits[:, ind[0] :]
                        else:
                            tmp[:, ind[0] : ind[1]] = logits[:, ind[0] : ind[1]]
            labels = parent_labels + tmp.to(self.args["device"])

        return labels

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
                tmp_loss = torch.Tensor([0]).to(self.args["device"])
                zeros = torch.zeros(parent_labels.shape, dtype=torch.float)
                for i in range(4):
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids,
                            segment_ids,
                            input_mask,
                            label_ids,
                            parent_labels=parent_labels,
                        )
                    iter_loss, tmp_logits = self._handle_out(outputs)
                    tmp_loss += iter_loss
                    logits = self._get_pred(i, parent_labels, tmp_logits, zeros)
                    parent_labels = logits

                tmp_loss = torch.mul(tmp_loss, 1 / 4)
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

            loss.append(tmp_loss.item())

        f1 = f1_score(labels, pred, average="micro", zero_division=0)
        recall = recall_score(labels, pred, average="micro")
        precision = precision_score(labels, pred, average="micro")
        accuracy = accuracy_score(labels, pred)

        result = {
            "loss": sum(loss) / len(loss),
            "accuracy": accuracy,
            "f1": f1,
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
