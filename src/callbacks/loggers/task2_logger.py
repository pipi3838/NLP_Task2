from torchvision.utils import make_grid
from src.callbacks.loggers import BaseLogger
import numpy as np
import random

class Task2Logger(BaseLogger):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(**kwargs)
    def _add_text(self, epoch, train_batch, train_output, valid_batch, valid_output,tokenizer):
        pass
        # print(train_batch.size)
        # print(train_output.size)
        # train_positive_list = np.unique(np.where(train_batch['label'].cpu().numpy() != 0))
        # if len(train_positive_list) == 0: train_slice_id = random.randint(0,train_output[2].size(0) - 1)
        # else: train_slice_id = random.choice(train_positive_list)
        # val_positive_list = np.unique(np.where(val_batch['label'].cpu().numpy() != 0))
        # if len(val_positive_list) == 0: val_slice_id = random.randint(0,val_output[2].size(0) - 1)
        # else: val_slice_id = random.choice(val_positive_list)
        # train_word = tokenizer.convert_ids_to_tokens(train_batch['ids'][train_slice_id])
        # train_pred = train_output[1][1]
        # self.writer.add_text()