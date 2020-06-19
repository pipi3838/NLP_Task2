from src.model.nets import BaseNet
from transformers import BertModel
from torch import nn

class CustomBert(BaseNet):
    def __init__(self, pretrained_model_name_or_path, dropout_rate, hidden_size, num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path, output_attentions=False, output_hidden_states=False)
        # bert_config = self.bert.config
        # bert_config.num_labels = 512

        # self.lstm = nn.LSTM(hidden_size, 128)
        # self.classifier = nn.Linear(hidden_size, num_labels)
        # self.lstm = nn.LSTM(hidden_size, 128, bidirectional=True)
        self.cause_classifier = nn.Linear(hidden_size, num_labels)
        self.effect_classifier = nn.Linear(hidden_size, num_labels)
        # self._init_weights(self.bert)
        # self._init_weights(self.classifier)
    def forward(self, input_ids, attention_mask, cause_start_pos=None, cause_end_pos=None, effect_start_pos=None, effect_end_pos=None):
        outputs = self.bert(input_ids, token_type_ids=None, attention_mask=attention_mask)
        # outputs = self.dropout(outputs[0])
        # logits, _ = self.lstm(outputs[0])
        cause_logits = self.cause_classifier(outputs[0])
        effect_logits = self.effect_classifier(outputs[0])

        cause_start_logits, cause_end_logits = cause_logits.split(1, dim=-1) 
        effect_start_logits, effect_end_logits = effect_logits.split(1, dim=-1)

        cause_start_logits = cause_start_logits.squeeze(-1)
        cause_end_logits = cause_end_logits.squeeze(-1)
        effect_start_logits = effect_start_logits.squeeze(-1)
        effect_end_logits = effect_end_logits.squeeze(-1)
        
        return cause_start_logits, cause_end_logits, effect_start_logits, effect_end_logits