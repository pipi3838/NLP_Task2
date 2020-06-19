from src.runner.trainers import BaseTrainer
import torch
import numpy as np

def valid_order(l, r):
    if l > r: return 0, 0
    if l == -1 or r == -1: return 0,0
    else: return l, r


class CustomBertTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.softmax = torch.nn.Softmax()

    def _train_step(self,batch):
        inputs, masks = batch['ids'].to(self.device), batch['masks'].to(self.device)
        cause_start_pos, cause_end_pos = batch['cause_start_pos'].to(self.device), batch['cause_end_pos'].to(self.device)
        effect_start_pos, effect_end_pos = batch['effect_start_pos'].to(self.device), batch['effect_end_pos'].to(self.device)
        cause_start_logits, cause_end_logits, effect_start_logits, effect_end_logits = \
        self.net(inputs, attention_mask=masks, cause_start_pos=cause_start_pos, cause_end_pos=cause_end_pos, effect_start_pos=effect_start_pos, effect_end_pos=effect_end_pos)
        
        loss = self.loss_fns.cross_entropy_loss(cause_start_logits, cause_end_logits, effect_start_logits, \
        effect_end_logits, cause_start_pos, cause_end_pos, effect_start_pos, effect_end_pos)

        # metrics = {metric.get_name(): metric(start_logits, end_logits, start_pos, end_pos) for metric in self.metric_fns}
        metrics = {}
        sent_len = cause_start_logits.shape[1]
        for metric in self.metric_fns:
            pred = np.zeros(cause_start_logits.shape)
            target = np.zeros(cause_start_logits.shape)

            pred_cause_start, pred_cause_end = torch.argmax(cause_start_logits, dim=1), torch.argmax(cause_end_logits, dim=1)
            pred_effect_start, pred_effect_end = torch.argmax(effect_start_logits, dim=1), torch.argmax(effect_end_logits, dim=1)

            for p, pcs, pce, pes, pee in zip(pred, pred_cause_start, pred_cause_end, pred_effect_start, pred_effect_end):
                pcs, pce, pes, pee = pcs.item(), pce.item(), pes.item(), pee.item()
                # print(pcs.item(), pce.item(), pes.item(), pee.item())
                pcs, pce = valid_order(pcs, pce)
                pes, pee = valid_order(pes, pee)
                print('pred', pcs, pce, pes, pee)
                p[pcs:pce+1] = 1
                p[pes:pee+1] = 1
            for t, tcs, tce, tes, tee in zip(target, cause_start_pos, cause_end_pos, effect_start_pos, effect_end_pos):
                tcs, tce, tes, tee = tcs.item(), tce.item(), tes.item(), tee.item()
                tcs, tce = valid_order(tcs, tce)
                tes, tee = valid_order(tes, tee)
                print('target', tcs, tce, tes, tee)
                t[tcs:tce+1] = 1
                t[tes:tee+1] = 1
                      
            res = metric(pred, target)
            metric_name = metric.get_name()
            if metric_name == 'Accuracy':
                metrics['precision'] = res[0]
                metrics['recall'] = res[1]
                metrics['F1Score'] = res[2]
            else: metrics[metric_name] = res
        return {
            'loss': loss,
            'metrics': metrics,
            'outputs': (cause_start_logits, cause_end_logits, effect_start_logits, effect_end_logits)
        }
    def _valid_step(self,batch):
        inputs, masks = batch['ids'].to(self.device), batch['masks'].to(self.device)
        cause_start_pos, cause_end_pos = batch['cause_start_pos'].to(self.device), batch['cause_end_pos'].to(self.device)
        effect_start_pos, effect_end_pos = batch['effect_start_pos'].to(self.device), batch['effect_end_pos'].to(self.device)
        cause_start_logits, cause_end_logits, effect_start_logits, effect_end_logits = \
        self.net(inputs, attention_mask=masks, cause_start_pos=cause_start_pos, cause_end_pos=cause_end_pos, effect_start_pos=effect_start_pos, effect_end_pos=effect_end_pos)
        
        loss = self.loss_fns.cross_entropy_loss(cause_start_logits, cause_end_logits, effect_start_logits, \
        effect_end_logits, cause_start_pos, cause_end_pos, effect_start_pos, effect_end_pos)

        # metrics = {metric.get_name(): metric(start_logits, end_logits, start_pos, end_pos) for metric in self.metric_fns}
        metrics = {}
        sent_len = cause_start_logits.shape[1]
        for metric in self.metric_fns:
            pred = np.zeros(cause_start_logits.shape)
            target = np.zeros(cause_start_logits.shape)

            pred_cause_start, pred_cause_end = torch.argmax(cause_start_logits, dim=1), torch.argmax(cause_end_logits, dim=1)
            pred_effect_start, pred_effect_end = torch.argmax(effect_start_logits, dim=1), torch.argmax(effect_end_logits, dim=1)

            for p, pcs, pce, pes, pee in zip(pred, pred_cause_start, pred_cause_end, pred_effect_start, pred_effect_end):
                pcs, pce, pes, pee = pcs.item(), pce.item(), pes.item(), pee.item()
                # print(pcs.item(), pce.item(), pes.item(), pee.item())
                pcs, pce = valid_order(pcs, pce)
                pes, pee = valid_order(pes, pee)
                p[pcs:pce+1] = 1
                p[pes:pee+1] = 1
            for t, tcs, tce, tes, tee in zip(target, cause_start_pos, cause_end_pos, effect_start_pos, effect_end_pos):
                tcs, tce = valid_order(tcs, tce)
                tes, tee = valid_order(tes, tee)
                t[tcs:tce+1] = 1
                t[tes:tee+1] = 1
                      
            res = metric(pred, target)
            metric_name = metric.get_name()
            if metric_name == 'Accuracy':
                metrics['precision'] = res[0]
                metrics['recall'] = res[1]
                metrics['F1Score'] = res[2]
            else: metrics[metric_name] = res
        return {
            'loss': loss,
            'metrics': metrics,
            'outputs': (cause_start_logits, cause_end_logits, effect_start_logits, effect_end_logits)
        }