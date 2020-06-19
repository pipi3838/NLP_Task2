import torch.nn as nn

class BaseNet(nn.Module):
    """The base class for all nets.
    """
    def _init_weights(self,module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            print('module init in 1')
            module.weight.data.normal_(mean=0.0, std=0.2)
        elif isinstance(module, nn.LayerNorm):
            print('module init in 2')
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            print('module init in 3')
            module.bias.data.zero_()

    def __init__(self):
        super().__init__()
        self.MODEL_CLASSES = {
            'bert': 'BertForQuestionAnswering',
            # 'xlm': 'XLMForSequenceClassification',
            # 'xlnet': 'XLNetForSequenceClassification',
            # 'robetra': 'RobertaForSequenceClassification'
        }
    def __repr__(self):
        params_size = sum([param.numel() for param in self.parameters() if param.requires_grad])
        return (
            super().__repr__() +
            f'\nTrainable parameters: {params_size / 1e6} M'
            f'\nMemory usage: {(params_size * 4) / (1 << 20)} MB'
        )
