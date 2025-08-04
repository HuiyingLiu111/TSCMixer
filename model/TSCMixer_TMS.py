import torch
from torch import nn
import torch.nn.init as init
from einops import rearrange
from layers.inception_time_pytorch.modules import InceptionModel


class ClassificationHead(nn.Module):
    def __init__(self, fusion_len, num_cls):
        super().__init__()
        self.fc = nn.Linear(fusion_len, num_cls)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.num_classes = configs['data']['num_classes']
        self.Dicp = configs['model']['Dicp']
        self.N = configs['model']['N']
        self.dims = configs['data']['num_dims']
        self.device = torch.device('cuda:0') if configs['training']['use_gpu'] else torch.device('cpu')
        self._build_model()
        self._init_weight()

    def _build_model(self):
        self.inceptions = InceptionModel(input_size=self.dims,
                                         num_classes=self.num_classes,
                                         filters=self.Dicp,
                                         depth=self.N,).to(self.device)

        self.classifier = ClassificationHead(self.Dicp * 4, self.num_classes).to(self.device)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)

            if isinstance(m, nn.Conv1d):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)
        return None

    def forward(self, x):
        H_TMS = self.inceptions(rearrange(x, 'b l d -> b d l').to(self.device))
        out = self.classifier(H_TMS)
        return out

