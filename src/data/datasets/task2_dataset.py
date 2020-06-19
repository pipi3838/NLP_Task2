from src.data.datasets import BaseDataset
import os
import torch
import transformers
import pandas as pd
import numpy as np

def sent_tokenize(sentences, tokenizer, max_length):
    ids,attention_masks = [],[]
    for sent in sentences:
        encoded_sent = tokenizer.encode_plus(str(sent), add_special_tokens=True, max_length=max_length, pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')
        ids.append(encoded_sent['input_ids'])
        attention_masks.append(encoded_sent['attention_mask'])
    ids = torch.cat(ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return ids, attention_masks

class Task2Dataset(BaseDataset):
    def __init__(self, data_dir, max_length, cause_or_effect, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer

        if self.type == 'train' or self.type == 'valid':
            if self.type == 'train': file_path = os.path.join(data_dir, 'train.tsv')
            else: file_path = os.path.join(data_dir, 'val.tsv')
            data = pd.read_csv(file_path, delimiter='\t')
            # self.text_len = data['text_len'].values
            if cause_or_effect == 'cause':
                self.start = torch.tensor(data['cause_start'].values).to(torch.int64)
                self.end = torch.tensor(data['cause_end'].values).to(torch.int64)
            elif cause_or_effect == 'effect':
                self.start = torch.tensor(data['effect_start'].values).to(torch.int64)
                self.end = torch.tensor(data['effect_end'].values).to(torch.int64)
            # self.labels = torch.tensor(self.labels).to(torch.int64)
        elif self.type == 'test':
            file_path = os.path.join(data_dir,'val.tsv')
            data = pd.read_csv(file_path,delimiter='\t')
            self.text_id = data['id'].values

        self.sents = data['text'].values
        self.ids, self.attention_masks = sent_tokenize(self.sents,self.tokenizer,max_length)
        # self.max_length = max_length

    def __getitem__(self,index):
        if self.type == 'train' or self.type == 'valid':
            return {'ids': self.ids[index], 
                    'masks': self.attention_masks[index], 
                    'start_pos': self.start[index],
                    'end_pos': self.end[index],
                    }
        if self.type == 'test':
            return {'ids': self.ids[index],'masks': self.attention_masks[index],'text_id': self.text_id[index], 'sents': self.sents[index]}
    
    def __len__(self):
        return self.ids.size()[0]    
