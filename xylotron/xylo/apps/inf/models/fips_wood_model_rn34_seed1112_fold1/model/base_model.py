from pathlib import Path
import torch
from torch import nn
from model_utils import *

class _BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None
        self.lyrgrps = None

    def load_weights(self, wts):
        if self.model:
            if isinstance(wts, Path):
                self.model.load_state_dict(torch.load(wts, map_location=torch.device('cpu')))
            else:
                self.model.load_state_dict(wts)

    def save_weights(self, wts_file):
        parent = Path(wts_file).parent
        parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), wts_file)

    def forward(self, x):
        return self.model(x)

    def apply_grouper(self, grouper=AsIsGrouper()):
        self.lyrgrps = grouper(self.model)

    def apply_freezer(self, freezer=UnfreezeAll()):
        freezer(self.lyrgrps)

    def apply_initializer(self, initializer=BatchNormSkipZeroBiasInitializer()):
        initializer(self.lyrgrps)

    def layer_groups(self):
        return self.lyrgrps

    def flattened(self):
        return flatten(self.model)

    def summary(self):
        np, ntp = 0, 0
        for mi, m in enumerate(self.model):
            print(f"{mi}/{len(self.model)}:")
            for fm in model_flatten(m):
                print(fm)
                for p in fm.parameters():
                    print(f"\t{p.requires_grad}: {p.shape} ({p.numel()})")
                    np += p.numel()
                    if p.requires_grad:
                        ntp += p.numel()
            print('+'*10)
            print()
        print(f"Number of parameters: {np}")
        print(f"Number of trainable parameters: {ntp}")

    def children(self):
        if not self.model:
            return []
        return list(self.model.children())

    def to_device(self, device):
        self.model.to(device)
