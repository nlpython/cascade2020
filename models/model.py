import torch
from torch import nn
from torch.optim import AdamW
from transformers import BertModel, BertTokenizer, BertPreTrainedModel
from torch.utils.data import DataLoader
from processer.utils import read_json, build_corpus
from models.loss import FocalLoss
from models.crf import CRF
import torch.nn.functional as F
from hyperparameter import args

class CascadeForNer(BertPreTrainedModel):

    def __init__(self, config):
        super(CascadeForNer, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size // 2, bidirectional=True,
                            batch_first=True)
        self.crf_linear = nn.Linear(config.hidden_size, args.num_orders)
        self.entity_linear = nn.Linear(config.hidden_size, args.num_entitys)
        self.crf = CRF(num_tags=args.num_orders, batch_first=True)

        self.loss_fc = FocalLoss(ignore_index=0)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, orders=None, entitys=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        cls = outputs[1].unsqueeze(1).repeat(1, input_ids.shape[1], 1)
        sequence_output = sequence_output + cls

        sequence_output, _ = self.lstm(sequence_output)
        crf_logits = self.crf_linear(sequence_output)
        entity_logits = self.entity_linear(sequence_output)

        outputs = (crf_logits, entity_logits)
        if orders is not None:
            # compute crf loss
            crf_loss = -1 * self.crf(emissions=crf_logits, tags=orders, mask=attention_mask)
            # compute entitys loss
            # output: [batch_size, seq_len, tag_entity_size] => [batchsize*len, tag_entity_size]
            entity_logits = entity_logits.reshape(-1, entity_logits.shape[-1])
            entitys = entitys.reshape(-1)
            # logit = F.log_softmax(entity_logits, dim=1)
            # loss_entity = F.nll_loss(logit, entitys)
            loss_entity = self.loss_fc(entity_logits, entitys)

            outputs =(crf_loss, loss_entity) + outputs # (loss), scores
        return outputs

if __name__ == '__main__':
    pass







