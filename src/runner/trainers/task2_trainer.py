from src.runner.trainers import BaseTrainer
import torch

class Task2Trainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _train_step(self, batch):
        inputs, masks = batch['ids'].to(self.device), batch['masks'].to(self.device)
        start_pos, end_pos = batch['start_pos'].to(self.device), batch['end_pos'].to(self.device)
        model_loss, start_logits, end_logits = self.net(inputs, attention_mask=masks, start_pos=start_pos, end_pos=end_pos)
        # metrics = {metric.get_name(): metric(start_logits, end_logits, start_pos, end_pos) for metric in self.metric_fns}
        metrics = {}
        for metric in self.metric_fns:
            res = metric(start_logits, end_logits, start_pos, end_pos)
            metric_name = metric.get_name()
            if metric_name == 'F1Score':
                metrics['Prec'] = res[0]
                metrics['Recall'] = res[1]
                metrics['F1Score'] = res[2]
            else: metrics[metric_name] = res
        return {
            'loss': model_loss,
            'metrics': metrics,
            'outputs': (start_logits, end_logits)
        }
        
    def _valid_step(self, batch):
        inputs, masks = batch['ids'].to(self.device), batch['masks'].to(self.device)
        start_pos, end_pos = batch['start_pos'].to(self.device), batch['end_pos'].to(self.device)
        model_loss, start_logits, end_logits = self.net(inputs, attention_mask=masks, start_pos=start_pos, end_pos=end_pos)
        # metrics = {metric.get_name(): metric(start_logits, end_logits, start_pos, end_pos) for metric in self.metric_fns}
        metrics = {}
        for metric in self.metric_fns:
            res = metric(start_logits, end_logits, start_pos, end_pos)
            metric_name = metric.get_name()
            if metric_name == 'F1Score':
                metrics['Prec'] = res[0]
                metrics['Recall'] = res[1]
                metrics['F1Score'] = res[2]
            else: metrics[metric_name] = res
        return {
            'loss': model_loss,
            'metrics': metrics,
            'outputs': (start_logits, end_logits)
        }
        
        
    