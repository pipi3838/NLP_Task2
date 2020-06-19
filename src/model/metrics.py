import torch
from torch import nn
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

class F1Score(nn.Module):
    def __init__(self):
        super().__init__()
        self.pred_len = 1e-10
        self.target_len = 1e-10
        self.match = 0.0

    def forward(self, start_logits, end_logits, start_pos, end_pos):
        start_pred = torch.argmax(start_logits, dim=-1)
        end_pred = torch.argmax(end_logits, dim=-1)
        
        for sp, ep, start, end in zip(start_pred, end_pred, start_pos, end_pos):
            sp, ep, start, end = sp.item(), ep.item(), start.item(), end.item()
            if sp < ep: self.pred_len += ep - sp
            self.target_len += end - start
            for idx in range(sp, ep+1):
                if idx in range(start, end+1):
                    self.match += 1 

        prec = (self.match / self.pred_len) + 1e-10
        recall = (self.match / self.target_len) + 1e-10
        f1 = (2 * prec * recall) / (prec + recall)
    
        return (prec, recall, f1)

    def reset_count(self):
        self.pred_len = 1e-10
        self.target_len = 1e-10
        self.match = 0

    def get_name(self):
        return 'F1Score'