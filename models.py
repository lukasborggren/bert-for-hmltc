from datetime import date

from transformers import BertForSequenceClassification, BertModel, BertLayer

import torch
from torch.nn import Dropout, Linear, BCEWithLogitsLoss


class BertBaseline(BertForSequenceClassification):
    def __init__(self, config, num_labels=2):
        super(BertBaseline, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = Dropout(config.hidden_dropout_prob)
        self.classifier = Linear(config.hidden_size, config.num_labels)
        self.apply(self._init_weights)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
    ):

        outputs = self.bert(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()

            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1, self.num_labels)
            )
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

    def freeze_bert_embeddings(self):
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

    def unfreeze_bert_embeddings(self):
        for param in self.bert.embeddings.parameters():
            param.requires_grad = True

    def freeze_bert_encoder(self):
        for param in self.bert.encoder.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.encoder.parameters():
            param.requires_grad = True

    def save(self):
        """Saves the model as a binary file."""
        model_to_save = (
            self.module if hasattr(self, "module") else self
        )  # Only save the model itself
        d = date.today().strftime("%Y-%m-%d")
        output_model_file = f"mltc/data/model_files/finetuned_{d}_pytorch_model.bin"
        torch.save(model_to_save.state_dict(), output_model_file)


class BertExperimental(BertBaseline):
    """A subclass that freezes the layers from a pretrained classifier whilst adding
    an additional transformer block that is finetuned during training.

    The size of a saved model (without the base blocks) is ~28.4 MB.
    """

    def __init__(self, config, num_labels=2):
        super(BertExperimental, self).__init__(config)
        self.bert = BertModel(config)
        self.freeze_bert_embeddings()
        self.freeze_bert_encoder()
        self.bert.encoder.add_module("12", BertLayer(config))
        self.apply(self._init_weights)
