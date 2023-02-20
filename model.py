import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils import logging

from noise_mechanism import PurMech, LapMech

logger = logging.get_logger(__name__)


class BertForSequenceClassificationLDP(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.noise_para = config.noise_para

        self.project_1 = nn.Linear(config.hidden_size, config.noise_para["proj_dim"]) if config.noise_para[
                                                                                             "proj_dim"] > 0 else None
        self.project_2 = nn.Linear(config.noise_para["proj_dim"], config.hidden_size) if config.noise_para[
                                                                                             "proj_dim"] > 0 else None
        self.activation = nn.Tanh()

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        if self.project_1 is not None:
            pooled_output = self.project_1(pooled_output)
            pooled_output = self.activation(pooled_output)

        if self.noise_para['epsilon1'] > 0:
            if self.noise_para['noise_type'] == 'PurMech':
                pooled_output = PurMech(pooled_output, self.noise_para['epsilon1'])
            elif self.noise_para['noise_type'] == 'LapMech':
                pooled_output = LapMech(pooled_output, self.noise_para['epsilon1'])
            else:
                raise NotImplementedError("Noise Type can be either [PurMech] or [LapMech]")

        if self.project_2 is not None:
            pooled_output = self.project_2(pooled_output)
            pooled_output = self.activation(pooled_output)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=pooled_output,
            attentions=outputs.attentions,
        )
