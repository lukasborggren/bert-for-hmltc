import torch
from torch.nn import Dropout, Linear, Sequential, BCEWithLogitsLoss, ReLU
from transformers import BertForSequenceClassification, BertModel


class BertBigBang(BertForSequenceClassification):
    def __init__(self, config):
        super(BertBigBang, self).__init__(config)
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
        dropout_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_output)

        outputs = (logits,)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1, self.num_labels)
            )
            outputs = (loss,) + outputs

        return outputs  # (loss), logits


class BertIterative(BertForSequenceClassification):
    def __init__(self, config):
        super(BertIterative, self).__init__(config)
        config.update({"mlp_size": 1024})

        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = Dropout(config.hidden_dropout_prob)
        self.mlp = Sequential(
            Linear(config.hidden_size + config.num_labels, config.mlp_size),
            ReLU(),
            Linear(config.mlp_size, config.mlp_size),
            ReLU(),
        )
        self.classifier = Linear(config.mlp_size, config.num_labels)
        self.apply(self._init_weights)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        parent_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        pooled_output = outputs[1]
        dropout_output = self.dropout(pooled_output)
        concat_output = torch.cat((dropout_output, parent_labels), dim=1)
        mlp_output = self.mlp(concat_output)
        logits = self.classifier(mlp_output)
        outputs = (logits,)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1, self.num_labels)
            )
            outputs = (loss,) + outputs

        return outputs  # (loss), logits
