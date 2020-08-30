import sys
from collections import defaultdict
import pprint
import torch
import torch.nn as nn

def flatten(m):
    return sum(map(flatten, m.children()), []) if len(list(m.children())) else [m]

def params(m, req_grad=None):
    if req_grad == None:
        return [p for c in flatten(m) for p in c.parameters()]
    else:
        return [p for c in flatten(m) for p in c.parameters() if p.requires_grad==req_grad]


######################################################
#################### INITIALIZING ####################
######################################################
class _ZeroBiasInitializer(object):
    def __init__(self, initfn=nn.init.kaiming_normal_, skips=None):
        '''
        Assumes batchnorm is the key for BatchNorm layer group
        '''
        self.initfn = initfn
        self.skips = skips if skips else []

    def __call__(self, lyrgrps):
        for grp in lyrgrps.groups():
            for k in grp:
                if k not in self.skips:
                    for m in flatten(grp[k]):
                        if self._init_weight(m):
                            self.initfn(m.weight)
                        if self._init_bias(m):
                            m.bias.data.fill_(0.0)

    def _init_weight(self, m):
        return hasattr(m, 'weight') and m.weight.requires_grad

    def _init_bias(self, m):
        return hasattr(m, 'bias') and hasattr(m.bias, 'data') and m.bias.requires_grad

class BatchNormSkipZeroBiasInitializer(_ZeroBiasInitializer):
    def __init__(self, initfn=nn.init.kaiming_normal_):
        '''
        Assumes batchnorm is the key for BatchNorm layer group
        '''
        super().__init__(initfn=initfn, skips=['batchnorm'])

######################################################
###################### FREEZING ######################
######################################################
class _BaseFreezer(object):
    '''
    Works on model.groups. All model.groups have the same keys.

    '''
    def __init__(self, upto=None, unfrozens=None):
        self.upto = upto
        self.unfrozens = unfrozens if unfrozens else []

    def __call__(self, g):
        if self.upto == None:
            self.upto = len(g.groups())

        for i in range(self.upto):
            self._set_status(g.groups()[i], False, self.unfrozens)

        for i in range(self.upto, len(g.groups())):
            self._set_status(g.groups()[i], True, [])

    def _set_status(self, grp, reqgrad, togglers=[]):
        for k in grp:
            status = reqgrad
            if k in togglers:
                status = not reqgrad
            for p in grp[k].parameters():
                p.requires_grad = status


class BatchNormAwareUptoFreezer(_BaseFreezer):
    def __init__(self, upto):
        '''
        Assumes that the groups have a key `batchnorm` that is
        nn.Sequential of only BatchNormXd layers.
        '''
        super().__init__(upto=upto, unfrozens=['batchnorm'])


class UptoFreezer(_BaseFreezer):
    def __init__(self, upto):
        super().__init__(upto=upto)


class FreezeAll(_BaseFreezer):
    def __init__(self):
        super().__init__()


class UnfreezeAll(_BaseFreezer):
    def __init__(self):
        super().__init__(upto=0)


######################################################
###################### GROUPING ######################
######################################################
class LayerGroups(object):
    def __init__(self, groups, keys):
        self.gps = groups
        self.kys = keys
        self.kys.sort()

    def keys(self):
        return self.kys

    def groups(self):
        return self.gps

    def flattened(self, reverse=False):
        if not reverse:
            return [g[k] for g in self.gps for k in g]
        else:
            return [g[k] for g in self.gps for k in g][::-1]

    def printout(self):
        pprint.pprint(self.gps)
        pprint.pprint(self.kys)


class _BaseGrouper(object):
    def __init__(self, groups=None):
        self.invmap = self._invmap(groups)
        self.keys= list(groups.keys()) if groups else []
        if 'other' not in self.keys:
            self.keys += ['other']

    def __call__(self, m):
        grps = []
        for c in m.children():
            mods = flatten(c)
            d = {k: [] for k in self.keys}
            for fm in mods:
                if type(fm) in self.invmap:
                    d[self.invmap[type(fm)]].append(fm)
                else:
                    d['other'].append(fm)
            grps.append(d)

        # make nn.Sequential
        for g in grps:
            for k in g:
                g[k] = nn.Sequential(*g[k])
        return LayerGroups(grps, self.keys)

    def _invmap(self, groups):
        inverse = {}
        if not groups:
            return inverse

        for k, v in groups.items():
            vals = v if isinstance(v, list) else [v]
            for e in vals:
                inverse[e] = k
        return inverse

class GenericGrouper(_BaseGrouper):
    def __init__(self, groups=None):
        super().__init__(groups=groups)


class BatchNormGrouper(_BaseGrouper):
    def __init__(self):
        super().__init__(groups={'batchnorm': [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]})

class AsIsGrouper(_BaseGrouper):
    def __init__(self):
        super().__init__(groups=None)
