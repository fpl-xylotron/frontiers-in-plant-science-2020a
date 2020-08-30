import torch
import torch.nn as nn
from torchvision import models
import sys
from pathlib import Path
source_dir = str(Path('../source'))
if source_dir not in sys.path:
    sys.path.append(source_dir)
from base_model import _BaseModel
from custom_layers import AdaptiveConcatPool2d, Flatten

default_head_params = {'concat_pool': True,
                       'dropout_prob1': 0.25,
                       'fc_size': 512,
                       'dropout_prob2': 0.5}
class ImageResNetTransferClassifier(_BaseModel):
    def __init__(self,
                 body_arch='resnet34',
                 body_num_blocks=8,
                 num_classes=2,
                 head_params=default_head_params):
        super().__init__()
        # Exactly one of default_head/custom_head needs to be specified.
        # For default_head specify dropout probs, linear sizes and number
        # of classes as a list. For custom_head provide a nn.Module.
        self.body_arch = body_arch
        self.body_num_blocks = body_num_blocks
        self.num_classes = num_classes
        self.head_params = head_params
        self.body = None
        self.head = None
        self._make_model()

    def freeze_body(self, bn_freeze=False):
        self.freeze(upto=len(self.body), bn_freeze=bn_freeze)

    def _make_model(self):
        self._make_body(self.body_arch, self.body_num_blocks)
        self._make_head(self.num_classes, self.head_params)
        body_head = list(self.body.children()) + [self.head]
        self.model = nn.Sequential(*body_head)

    def _make_body(self, arch, num_blocks):
        resnets = {'resnet18': models.resnet18,
                   'resnet34': models.resnet34,
                   'resnet50': models.resnet50,
                   'resnet101': models.resnet101,
                   'resnet152': models.resnet152}
        lyrs = list(resnets[arch](pretrained=True).children())[:num_blocks]
        self.body = nn.Sequential(*lyrs)

    def _make_head(self, num_classes, head_params):
        nfeats = self.body(torch.zeros(1, 3, 256, 256)).shape[1]
        pool = nn.AdaptiveAvgPool2d(output_size=1)
        if head_params['concat_pool']:
            nfeats *= 2
            pool = AdaptiveConcatPool2d(output_size=1)

        layers = [pool,
                  Flatten(),
                  nn.BatchNorm1d(num_features=nfeats),
                  nn.Dropout(p=head_params['dropout_prob1']),
                  nn.Linear(in_features=nfeats, out_features=head_params['fc_size']),
                  nn.ReLU(inplace=True),
                  nn.BatchNorm1d(num_features=head_params['fc_size']),
                  nn.Dropout(p=head_params['dropout_prob2']),
                  nn.Linear(in_features=head_params['fc_size'], out_features=num_classes)]
        self.head = nn.Sequential(*layers)
