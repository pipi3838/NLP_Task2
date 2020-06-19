from src.model.nets import BaseNet
import transformers
from torch import nn

class pretrainedNet(BaseNet):
    def __init__(self, model_type, trained_path, num_labels):
        super().__init__()
        model_class = getattr(transformers,self.MODEL_CLASSES[model_type])
        self.model = model_class.from_pretrained(trained_path, num_labels=num_labels, output_attentions=False, output_hidden_states=False)
        # self.softmax = nn.Softmax()
    def forward(self, input_ids, attention_mask, start_pos=None, end_pos=None):
        if start_pos is not None and end_pos is not None:
            loss, start_logits, end_logits = self.model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, start_positions=start_pos, end_positions=end_pos)
            return loss, start_logits, end_logits
        else: 
            start_logits, end_logits = self.model(input_ids=input_ids,token_type_ids=None,attention_mask=attention_mask)
            return start_logits, end_logits

    # def save_pretrained(self, saved_dir):
    #     saved_dir = str(saved_dir)
    #     path = '/'.join(saved_dir.split('/')[0:-2]) + '/module/'
    #     model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Take care of distributed/parallel training
    #     model_to_save.save_pretrained(path)