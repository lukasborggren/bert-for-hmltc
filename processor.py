import json
from os.path import join

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(
        self, guid, text_a, text_b=None, labels=None, brand=None, parent_labels=None
    ):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            labels: (Optional) [string]. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b  # Not used
        self.labels = labels
        self.parent_labels = parent_labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self, input_ids, input_mask, segment_ids, label_ids, parent_label_ids=None
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.parent_label_ids = parent_label_ids


class TextProcessor:
    """Class for processing and transforming data to be used as input to the
    classifiers."""

    def __init__(self, args, tokenizer, logger, labels):
        self.args = args
        self.tokenizer = tokenizer
        self.logger = logger

        self.data_dir = join(args["DATA_PATH"], "dataframes")
        with open(join(args["DATA_PATH"], labels), "r") as f:
            self.labels = json.load(f)

    def _create_examples(self, df, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, row) in enumerate(df.values):
            input_example = None
            guid = i
            text_a = row[0]

            if set_type != "test":
                labels = row[1]
                if self.args["use_parents"] and set_type == "train":
                    parent_labels = row[2]
                    input_example = InputExample(
                        guid=guid,
                        text_a=text_a,
                        labels=labels,
                        parent_labels=parent_labels,
                    )
            else:
                labels = []

            if not input_example:
                input_example = InputExample(guid=guid, text_a=text_a, labels=labels)

            examples.append(input_example)
        return examples

    def get_examples(self, file_name, set_type):
        """Gets input data from pickle and loads it in a Pandas DataFrame."""
        df = pd.read_pickle(join(self.data_dir, file_name))
        return self._create_examples(df, set_type)

    # The two functions below are heavily inspired by their counterparts in:
    # https://github.com/google-research/bert/blob/master/extract_features.py
    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def convert_examples_to_features(self, examples):
        """Loads a data file into a list of `InputBatch`s."""
        features = []
        max_seq_length = self.args["max_seq_length"]
        for (ex_index, example) in enumerate(examples):
            tokens_a = self.tokenizer.tokenize(example.text_a)

            tokens_b = None
            if example.text_b:
                tokens_b = self.tokenizer.tokenize(example.text_b)
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[: (max_seq_length - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0   0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambigiously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            label_ids = []
            for label in example.labels:
                label_ids.append(float(label))

            if example.parent_labels:
                parent_label_ids = []
                for label in example.parent_labels:
                    parent_label_ids.append(float(label))

            if ex_index < 0:
                self.logger.info("*** Example ***")
                self.logger.info("guid: %s" % (example.guid))
                self.logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                self.logger.info(
                    "input_ids: %s" % " ".join([str(x) for x in input_ids])
                )
                self.logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask])
                )
                self.logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids])
                )
                self.logger.info("label: %s (id = %s)" % (example.labels, label_ids))
                self.logger.info(
                    "label: %s (id = %s)" % (example.parent_labels, parent_label_ids)
                )

            if example.parent_labels:
                input_features = InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_ids=label_ids,
                    parent_label_ids=parent_label_ids,
                )
            else:
                input_features = InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_ids=label_ids,
                )
            features.append(input_features)

        return features

    def pack_features_in_dataloader(self, features, set_type):
        """Transforms input features to tensors and packs them in a PyTorch
        DataLoader."""
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in features], dtype=torch.long
        )

        if set_type != "test":
            all_label_ids = torch.tensor(
                [f.label_ids for f in features], dtype=torch.float
            )
            if self.args["use_parents"]:
                if features[0].parent_label_ids:  # Using parents train
                    all_parent_label_ids = torch.tensor(
                        [f.parent_label_ids for f in features], dtype=torch.float
                    )
                else:  # Using parents dev
                    shape = (len(features), len(self.labels))
                    all_parent_label_ids = torch.zeros(shape, dtype=torch.float)
                data = TensorDataset(
                    all_input_ids,
                    all_input_mask,
                    all_segment_ids,
                    all_label_ids,
                    all_parent_label_ids,
                )
            else:  # Not using parents train & dev
                data = TensorDataset(
                    all_input_ids, all_input_mask, all_segment_ids, all_label_ids
                )
        elif self.args["use_parents"]:  # Using parents test
            shape = (len(features), len(self.labels))
            all_parent_label_ids = torch.zeros(shape, dtype=torch.float)
            data = TensorDataset(
                all_input_ids,
                all_input_mask,
                all_segment_ids,
                all_parent_label_ids,
            )
        else:  # Not using parents test
            data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        if set_type != "test":
            sampler = RandomSampler(data)
        else:
            sampler = SequentialSampler(data)
        dataloader = DataLoader(
            data, sampler=sampler, batch_size=self.args["batch_size"]
        )

        return dataloader
