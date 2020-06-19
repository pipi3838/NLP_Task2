import torch
from torch import nn
import numpy as np

__all__ = [
    'CrossEntropyLossWrapper',
    # 'DiceLoss',
]
device = 'cuda:0'

def get_span(start_pos, shape):
    res = np.zeros(shape)
    sent_len = shape[1]
    for r,p in zip(res, start_pos): 
        if p < sent_len: r[p] = 1
    return torch.tensor(res).to(device, dtype=torch.float)

class CrossEntropyLossWrapper(nn.Module):

    def __init__(self, with_logits=True, weight=None, class_weight=None, **kwargs):
        super().__init__()

        # self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignored_index)

    def forward(self, cause_start_logits, cause_end_logits, effect_start_logits, effect_end_logits, \
cause_start_pos, cause_end_pos, effect_start_pos, effect_end_pos):

        ignored_index = cause_start_logits.size(1)
        self.loss_fn = nn.BCEWithLogitsLoss()
        # self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignored_index)
        res_shape = cause_start_logits.shape
        
        # print(cause_start_pos.shape)

        # if len(cause_start_pos.size()) > 1: cause_start_pos = cause_start_pos.squeeze(-1)
        # if len(cause_end_pos.size()) > 1: cause_end_pos = cause_end_pos.squeeze(-1)
        # if len(effect_start_pos.size()) > 1: effect_start_pos = effect_start_pos.squeeze(-1)
        # if len(effect_end_pos.size()) > 1: effect_end_pos = effect_end_pos.squeeze(-1)
        
        # print(cause_start_pos.shape)

        cause_start_pos.clamp(0, ignored_index)
        cause_end_pos.clamp(0, ignored_index)
        effect_start_pos.clamp(0, ignored_index)
        effect_end_pos.clamp(0, ignored_index)

        # cause_start_logits = cause_start_logits
        # cause_end_logits = cause_end_logits
        # effect_start_logits = effect_start_logits
        # effect_end_logits = effect_end_logits

        # cause_start_pos = cause_start_pos.unsqueeze(-1)
        # cause_end_pos = cause_end_pos.unsqueeze(-1)
        # effect_start_pos = effect_start_pos.unsqueeze(-1)
        # effect_end_pos = effect_end_pos.unsqueeze(-1)

        cause_start_pos = get_span(cause_start_pos, res_shape)
        cause_end_pos = get_span(cause_end_pos, res_shape)
        effect_start_pos = get_span(effect_start_pos, res_shape)
        effect_end_pos = get_span(effect_end_pos, res_shape)
        
        # print(cause_start_pos.shape)
        # print('cause_end_logits.shape', cause_end_logits.shape)
        # raise Exception('stop for debug')

        cause_start_loss = self.loss_fn(cause_start_logits, cause_start_pos)
        cause_end_loss = self.loss_fn(cause_end_logits, cause_end_pos)
        effect_start_loss = self.loss_fn(effect_start_logits, effect_start_pos)
        effect_end_loss = self.loss_fn(effect_end_logits, effect_end_pos)

        total_loss = (cause_start_loss + cause_end_loss + effect_start_loss + effect_end_loss) / 4
        return total_loss

# class DiceLoss(nn.Module):
#     def __init__(self, **kwargs):
#         super().__init__()
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax()
#         self.eplison = 1

#     def forward(self, output, target):
#         output = self.softmax(output)[:,1]
#         target = target.type(torch.float)
#         loss = 1 - ((2 * (1 - output) * (output) * target + self.eplison) / ( (1 - output) * (output) + target + self.eplison))
#         return torch.mean(loss)