

from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
import torch.nn as nn
import torch


class BertForTokenClassification(BertPreTrainedModel):

    def __init__(self, config, num_labels):
        super(BertForTokenClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        return logits


    def loss(self, logits, attention_mask, labels):

        # Only keep active parts of the loss
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = self.loss_fct(active_logits, active_labels)

            total = active_labels.size(0)
            _, pred = torch.max(active_logits, 1)
            correct = (pred == active_labels).sum().item()

        else:
            logits = logits.view(-1, self.num_labels)
            labels = labels.view(-1)
            loss = self.loss_fct(logits, labels)

            total = labels.size(0)
            _, pred = torch.max(logits, 1)
            correct = (pred == labels).sum().item()


        return loss, total, correct

# rdoc is not typical sentence classification, since we need to consider the category
# self-attention -> concatenate category -> softmax
class BertForSequenceClassification_rdoc(BertPreTrainedModel):

    def __init__(self, config, num_labels, num_category):
        super(BertForSequenceClassification_rdoc, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.category_emb = nn.Embedding(num_category, config.hidden_size)

        self.classifier = nn.Linear(config.hidden_size + config.hidden_size, num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        self.apply(self.init_bert_weights)



    def forward(self, categories, input_ids, token_type_ids=None, attention_mask=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        category_reps = self.category_emb(categories)
        pooled_output = torch.cat((pooled_output, category_reps), dim=-1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

    def loss(self, logits, labels):
        loss = self.loss_fct(logits, labels)

        total = labels.size(0)
        _, pred = torch.max(logits, 1)
        correct = (pred == labels).sum().item()

        return loss, total, correct


class BertForSequenceClassification(BertPreTrainedModel):

    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

