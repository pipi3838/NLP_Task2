import logging
import torch
from tqdm import tqdm

from src.runner.utils import EpochLog

LOGGER = logging.getLogger(__name__.split('.')[-1])


class BasePredictor:
    """The base class for all predictors.
    Args:
        saved_dir (Path): The root directory of the saved data.
        device (torch.device): The device.
        test_dataloader (Dataloader): The testing dataloader.
        net (BaseNet): The network architecture.
        loss_fns (LossFns): The loss functions.
        loss_weights (LossWeights): The corresponding weights of loss functions.
        metric_fns (MetricFns): The metric functions.
    """

    def __init__(self, saved_dir, device, test_dataloader,
                 net, metric_fns):
        self.saved_dir = saved_dir
        self.device = device
        self.test_dataloader = test_dataloader
        self.net = net.to(device)
        # self.loss_fns = loss_fns
        # self.loss_weights = loss_weights
        self.metric_fns = metric_fns

    def predict(self):
        """The testing process.
        """
        self.net.eval()
        dataloader = self.test_dataloader
        pbar = tqdm(dataloader, desc='test', ascii=True)

        # epoch_log = EpochLog()
        out = open('./ans.csv','w')
        out.write('Index;Text;Cause;Effect\n')

        for i, batch in enumerate(pbar):
            with torch.no_grad():
                inputs, masks = batch['ids'].to(self.device), batch['masks'].to(self.device)
                cause_start_logits, cause_end_logits, effect_start_logits, effect_end_logits = self.net(inputs, attention_mask=masks)

                input_id = batch['text_id'] 
                sents = batch['sents']


                print(cause_start_logits)
                pred_cause_start, pred_cause_end = torch.argmax(cause_start_logits, dim=1), torch.argmax(cause_end_logits, dim=1)
                pred_effect_start, pred_effect_end = torch.argmax(effect_start_logits, dim=1), torch.argmax(effect_end_logits, dim=1)

                for idx, sent, cs, ce, es, ee in zip(input_id, sents, pred_cause_start, pred_cause_end, pred_effect_start, pred_effect_end):
                    print(cs,ce,es,ee)
                    out.write('{};{};{}\n'.format(idx, sent[cs:ce+1], sent[es:ee+1]))

            if (i + 1) == len(dataloader) and not dataloader.drop_last:
                batch_size = len(dataloader.dataset) % dataloader.batch_size
            else:
                batch_size = dataloader.batch_size
            # epoch_log.update(batch_size, loss, losses, metrics)

            # pbar.set_postfix(**epoch_log.on_step_end_log)
        # test_log = epoch_log.on_epoch_end_log
        # LOGGER.info(f'Test log: {test_log}.')


    def _test_step(self, batch):
        pass

    def load(self, path):
        """Load the model checkpoint.
        Args:
            path (Path): The path to load the model checkpoint.
        """
        checkpoint = torch.load(path, map_location='cpu')
        self.net.load_state_dict(checkpoint['net'])