from src.runner.predictors import BasePredictor
import torch

class Task2Trainer(BasePredictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _test_step(self,batch):
        inputs,labels,masks = batch['ids'].to(self.device), batch['labels'].to(self.device), batch['masks'].to(self.device)
        logits = self.net(inputs, attention_mask=masks)
        return {
            'loss': model_loss,
            'metrics': metrics,
            'outputs': logits 
        }