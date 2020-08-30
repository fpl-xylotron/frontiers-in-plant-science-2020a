import torch
from torchvision import transforms, utils
from torchvision.transforms import functional as tvf
import math
import random
import PIL
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, PILLOW_VERSION
import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import numbers
import types
import collections
import warnings
from enum import IntEnum

rrint = random.randint

class SampElemType(IntEnum):
    IMAGE = 1
    MASK = 2
    COORD = 3
    LABEL = 4


class Transform(object):
    '''Base class from which image augmentations are subclassed.'''

    def __init__(self):
        self.calls = {SampElemType.IMAGE: self._image,
                      SampElemType.MASK: self._mask,
                      SampElemType.COORD: self._coord,
                      SampElemType.LABEL: self._label}
        self.nrc = tuple([0, 0])

    def __call__(self, sample):
        return self._tsample(sample)

    def _image(self, img):
        return img

    def _mask(self, mask):
        return mask

    def _coord(self, rcs):
        return rcs

    def _label(self, label):
        return label

    def _tsample(self, sample):
        tsample = {}
        for k in sample:
            tsample[k] = (self.calls[sample[k][1]](sample[k][0]), sample[k][1])
        return tsample

    def _nrc(self, sample):
        for k in sample:
            if sample[k][1] in [SampElemType.IMAGE, SampElemType.MASK]:
                if isinstance(sample[k][0], PIL.Image.Image):
                    self.nrc = sample[k][0].size[::-1]
                elif isinstance(sample[k][0], torch.Tensor):
                    self.nrc = sample[k][0].shape[1:]
                else:
                    self.nrc = sample[k][0].shape[0:2]
                break

    def _patch_pts(self, pts, dims):
        return [p for p in pts if p[0] >= 0 and p[0] < dims[0] and p[1] >= 0 and p[1] < dims[1]]

    def _isseqlen2(self, a):
        return isinstance(q, (list, tuple)) and (len(A) == 2)

    def _unchanged(self, sample):
        return sample


class ToTensor(Transform):
    def __init__(self):
        super().__init__()

    def __call__(self, sample):
        return self._tsample(sample)

    def _image(self, img):
        return tvf.to_tensor(img)

    def _mask(self, mask):
        return torch.from_numpy(np.asarray(mask).astype(np.int64, copy=False))

    def _coord(self, coord):
        return torch.from_numpy(np.array(coord))

    def _label(self, label):
        return torch.from_numpy(np.array(label))


class MeanStdNormalize(Transform):
    def __init__(self, mean=None, std=None, from_file=None):
        super().__init__()
        self.mean, self.std = None, None
        self.persample = False
        if mean is not None and std is not None:
            self.mean = mean
            self.std = std
        elif from_file is not None:
            df = pd.read_csv(from_file)
            self.mean = df['mean'].values
            self.std = df['std'].values
        else:
            self.persample = True

    def __call__(self, sample):
        return self._tsample(sample)

    def _image(self, tensor):
        self._compute_stats(tensor)
        return tvf.normalize(tensor, self.mean, self.std)

    def _compute_stats(self, tensor):
        if self.persample:
            self.mean = torch.Tensor([torch.mean(tensor[c, :, :]) for c in range(tensor.shape[0])])
            self.std = torch.Tensor([torch.std(tensor[c, :, :]) for c in range(tensor.shape[0])])

class ImageNetNormalize(MeanStdNormalize):
    def __init__(self):
        super().__init__(mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225]))


##############################################################################
# Resizing
##############################################################################
class Resize(Transform):
    def __init__(self, out_size=None):
        super().__init__()
        self.outsize = out_size
        self.factors = (1.0, 1.0)

    def __call__(self, sample):
        self._nrc(sample)
        self._outsize()
        self._factors()
        return self._tsample(sample)

    def _image(self, img):
        return tvf.resize(img, self.outsize, interpolation=Image.BILINEAR)

    def _mask(self, mask):
        return tvf.resize(mask, self.outsize, interpolation=Image.NEAREST)

    def _coord(self, rcs):
        return [(p[0]*self.factors[0], p[1]*self.factors[1]) for p in rcs]

    def _outsize(self):
        pass

    def _factors(self):
        self.factors = tuple([o/i for (o, i) in zip(self.outsize, self.nrc)])


class ResizeRandom(Resize):
    def __init__(self, out_nr_range, out_nc_range):
        super().__init__()
        self.nr_range = out_nr_range
        self.nc_range = out_nc_range

    def _outsize(self):
        self.outsize = (rrint(*self.nr_range), rrint(*self.nc_range))


##############################################################################
# Padding
##############################################################################
class Pad(Transform):
    def __init__(self, padding=0, ifill=0, mfill=0, padding_mode='constant'):
        '''
        padding: integer/2-tuple/4-tuple. See torchvision.tranforms.Pad
        4-tuple will be interpreted as (left, top, right, bottom)
        integer will be interpreted as same size padding on all sides.
        WARNING: Note the ordering of numbers in padding

        ifill: value in padding pixels in image
        mfill: value in padding pixels in mask

        IMAGE: valid type(s) PIL.Image.Image
        MASK: valid type(s) PIL.Image.Image
        COORD: valid type(s) 2-col np.ndarray
        LABEL: no-op
        '''
        super().__init__()
        self.padding = padding
        self.ifill = ifill
        self.mfill = mfill
        self.padding_mode = padding_mode

    def __call__(self, sample):
        self._nrc(sample)
        self._padding()
        return self._tsample(sample)

    def _image(self, img):
        return tvf.pad(img, self.padding, self.ifill, self.padding_mode)

    def _mask(self, mask):
        return tvf.pad(mask, self.padding, self.mfill, self.padding_mode)

    def _coord(self, rcs):
        if isinstance(self.padding, int):
            tl = (self.padding, self.padding)
        elif isinstance(self.padding, (tuple, list)) and len(list(self.padding)) in [2, 4]:
            tl = (self.padding[1], self.padding[0])
        else:
            tl = (0, 0)
        return rcs + tl

    def _padding(self):
        pass


class PadToEnsureSize(Pad):
    def __init__(self, out_size, ifill=0, mfill=0, padding_mode='constant'):
        '''
        ifill: value in padding pixels in image
        mfill: value in padding pixels in mask

        IMAGE: valid type(s) PIL.Image.Image
        MASK: valid type(s) PIL.Image.Image
        COORD: valid type(s) 2-col np.ndarray
        LABEL: no-op
        '''
        super().__init__()
        self.padding = (0, 0, 0, 0)
        self.outsize = out_size
        self.ifill = ifill
        self.mfill = mfill
        self.padding_mode = padding_mode

    def _padding(self):
        roff = self.outsize[0] - self.nrc[0]
        if roff < 0:
            tpad, bpad = 0, 0
        else:
            tpad, bpad = roff//2, roff - roff//2

        coff = self.outsize[1] - self.nrc[1]
        if coff < 0:
            lpad, rpad = 0, 0
        else:
            lpad, rpad = coff//2, coff - coff//2

        self.padding = (lpad, tpad, rpad, bpad)


##############################################################################
# Erase changes
##############################################################################
class EraseCutout(Transform):
    def __init__(self, hole_size=(50, 50), num_holes=1, prob=0.5):
        '''
        This is useful only in the image classification context.
        Zero masks of fixed size. The paper does not have prob.

        IMAGE: valid type(s) torch.Tensor
        MASK: no-op
        COORD: no-op
        LABEL: no-op
        '''
        super().__init__()
        self.holeshape = hole_size
        self.nholes = num_holes
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            return self._tsample(sample)
        else:
            return self._unchanged(sample)

    def _image(self, img):
        if not isinstance(img, torch.Tensor):
            print('EraseCutout: img has to be torch.Tensor')
            return None

        hgt, wdt = img.size(1), img.size(2)
        mask = torch.ones(img.shape[1:], dtype=img.dtype)
        for i in range(self.nholes):
            tlr = np.clip(rrint(0, hgt) - self.holeshape[0]//2, 0, hgt)
            tlc = np.clip(rrint(0, wdt) - self.holeshape[1]//2, 0, wdt)
            brr, brc = (tlr + self.holeshape[0]), (tlc + self.holeshape[1])
            mask[tlr:brr, tlc:brc] = 0
        img = img*mask.expand_as(img)
        return img


##############################################################################
# Color changes
##############################################################################
class ColorToGreyRandom(Transform):
    def __init__(self, num_out_channels=1, prob=0.5):
        '''
        change image to grey.
        NOTE: if transform is not applied due to prob value,
        num_out_channels will be same as input

        IMAGE: valid type(s) PIL.Image.Image
        MASK: no-op
        COORD: no-op
        LABEL: no-op
        '''
        super().__init__()
        self.prob = prob
        self.noc = num_out_channels

    def __call__(self, sample):
        if random.random() < self.prob:
            return self._tsample(sample)
        else:
            return self._unchanged(sample)

    def _image(self, img):
        return tvf.to_grayscale(img, self.noc)


class ColorJitterRandom(Transform):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, prob=0.5):
        '''
        brightness, contrast, saturation, hue are all scalars.
        torchvision.transforms.ColorJitter is used.
        range [max(0, (1 - param)), 1 + param] is uniformaly sampled for
        param in {brightness, contrast, saturation, hue}

        IMAGE: valid type(s) PIL.Image.Image
        MASK: no-op
        COORD: no-op
        LABEL: no-op
        '''
        super().__init__()
        self.tfm = transforms.ColorJitter(brightness=brightness,
                                          contrast=contrast,
                                          saturation=saturation,
                                          hue=hue)
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            self._nrc(sample)
            return self._tsample(sample)
        else:
            return self._unchanged(sample)

    def _image(self, img):
        if img.getbands()[0] == 'L':
            return self.tfm(img.convert('RGB')).convert('L')
        return self.tfm(img)


##############################################################################
# Patch extraction
##############################################################################
class Patch(Transform):
    def __init__(self, top_left=(0, 0), out_size=(0, 0)):
        '''
        IMAGE: valid type(s) PIL.Image.Image
        MASK: valid type(s) PIL.Image.Image/np.ndarray
        COORD: valid type(s) 2-col np.ndarray
        LABEL: no-op
        '''
        super().__init__()
        self.tl = top_left
        self.outsize = out_size

    def __call__(self, sample):
        self._nrc(sample)
        self._outsize_and_tl()
        return self._tsample(sample)

    def _image(self, img):
        return tvf.crop(img, self.tl[0], self.tl[1], self.outsize[0], self.outsize[1])

    def _mask(self, mask):
        if isinstance(mask, PIL.Image.Image):
            return tvf.crop(mask, self.tl[0], self.tl[1], self.outsize[0], self.outsize[1])
        elif isinstance(mask, np.ndarray) and len(mask.shape) == 2:
            startr, stopr = self.tl[0], self.tl[0] + self.outsize[0]
            startc, stopc = self.tl[1], self.tl[1] + self.outsize[1]
            return mask[startr:stopr, startc:stopc]
        else:
            return None

    def _coord(self, rcs):
        return rcs - np.array(self.tl, dtype=np.float)

    def _outsize_and_tl(self):
        pass


class PatchRandomOriginFixedSize(Patch):
    def __init__(self, out_size):
        '''
        IMAGE: valid type(s) PIL.Image.Image
        MASK: valid type(s) PIL.Image.Image/np.ndarray
        COORD: valid type(s) 2-col np.ndarray
        LABEL: no-op
        '''
        super().__init__()
        self.outsize = out_size
        if isinstance(out_size, int):
            self.outsize = (out_size, out_size)
        self.tl = (0, 0)

    def _outsize_and_tl(self):
        self.tl = (rrint(0, self.nrc[0] - self.outsize[0]),
                   rrint(0, self.nrc[1] - self.outsize[1]))


class PatchRandomOriginRandomSize(Patch):
    def __init__(self, nr_range, nc_range):
        '''
        IMAGE: valid type(s) PIL.Image.Image
        MASK: valid type(s) PIL.Image.Image/np.ndarray
        COORD: valid type(s) 2-col np.ndarray
        LABEL: no-op
        '''
        super().__init__()
        self.nr_range = nr_range
        self.nc_range = nc_range
        self.tl = (0, 0)
        self.outsize = (0, 0)


    def _outsize_and_tl(self):
        self.nr_range = (max(self.nr_range[0], 0),
                         min(self.nr_range[1], self.nrc[0]))

        self.nc_range = (max(self.nc_range[0], 0),
                         min(self.nc_range[1], self.nrc[1]))

        self.outsize = (rrint(*self.nr_range), rrint(*self.nc_range))
        self.tl = (rrint(0, self.nrc[0] - self.outsize[0]),
                   rrint(0, self.nrc[1] - self.outsize[1]))


class PatchCentralFixedSize(Patch):
    def __init__(self, out_size):
        '''
        IMAGE: valid type(s) PIL.Image.Image
        MASK: valid type(s) PIL.Image.Image/np.ndarray
        COORD: valid type(s) 2-col np.ndarray
        LABEL: no-op
        '''
        super().__init__()
        self.tl = (0, 0)
        self.outsize = out_size
        if isinstance(out_size, int):
            self.outsize = (out_size, out_size)

    def _outsize_and_tl(self):
        self.tl = (self.nrc[0]//2 - self.outsize[0]//2,
                   self.nrc[1]//2 - self.outsize[1]//2)
        if self.tl[0] < 0 or self.tl[1] < 0:
            print('PatchCentralFixedShape, tl is negative')


class PatchCentralRandomSize(Patch):
    def __init__(self, nr_range, nc_range):
        '''
        IMAGE: valid type(s) PIL.Image.Image
        MASK: valid type(s) PIL.Image.Image/np.ndarray
        COORD: valid type(s) 2-col np.ndarray
        LABEL: no-op
        '''
        super().__init__()
        self.nr_range = nr_range
        self.nc_range = nc_range
        self.tl = (0, 0)
        self.outsize = (0, 0)

    def _outsize_and_tl(self):
        self.nr_range = (max(self.nr_range[0], 0),
                         min(self.nr_range[1], self.nrc[0]))

        self.nc_range = (max(self.nc_range[0], 0),
                         min(self.nc_range[1], self.nrc[1]))

        self.outsize = (rrint(*self.nr_range), rrint(*self.nc_range))

        self.tl = (self.nrc[0]//2 - self.outsize[0]//2,
                   self.nrc[1]//2 - self.outsize[1]//2)
        if self.tl[0] < 0 or self.tl[1] < 0:
            print('PatchCentralFixedShape, tl is negative')


class PatchTiled(Transform):
    def __init__(self, out_size, overlap_ok=True, num_patches=None):
        '''
        IMAGE: valid type(s) PIL.Image.Image
        MASK: valid type(s) PIL.Image.Image/np.ndarray
        COORD: valid type(s) 2-col np.ndarray
        LABEL: no-op

        NOTE: This returns a list of transformed samples.
        '''
        super().__init__()
        self.outsize = out_size
        self.npatches = num_patches
        self.overlap = overlap_ok

        self.tls = []

    def __call__(self, sample):
        self._nrc(sample)
        self._tls()
        return [Patch(top_left=tl, out_size=self.outsize)(sample) for tl in self.tls]

    def _tls(self):
        self.tls = []
        rstep, cstep = self.outsize
        if self.overlap and self.npatches:
            rstep, cstep = tuple(np.array(self.outsize[0], dtype=np.int)//self.npatches)

        tlr = 0
        while (tlr + self.outsize[0]) < self.nrc[0]:
            tlc = 0
            while (tlc + self.outsize[1]) < self.nrc[1]:
                self.tls.append(tlr, tlc)
                tlc += cstep
            tlr += rstep

##############################################################################
# Flip image
##############################################################################
class FlipLRRandom(Transform):
    def __init__(self, prob=0.5):
        '''
        IMAGE: valid type(s) PIL.Image.Image
        MASK: valid type(s) PIL.Image.Image/np.ndarray
        COORD: valid type(s) 2-col np.ndarray
        LABEL: no-op
        '''
        super().__init__()
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            self._nrc(sample)
            return self._tsample(sample)
        else:
            return self._unchanged(sample)

    def _image(self, img):
        return tvf.hflip(img)

    def _mask(self, mask):
        return tvf.hflip(mask)

    def _coord(self, rcs):
        clast = self.nrc[1] - 1
        trcs = np.copy(rcs)
        trcs[:, 1] = clast - trcs[:, 1]
        return trcs


class FlipTDRandom(Transform):
    def __init__(self, prob=0.5):
        '''
        IMAGE: valid type(s) PIL.Image.Image
        MASK: valid type(s) PIL.Image.Image/np.ndarray
        COORD: valid type(s) 2-col np.ndarray
        LABEL: no-op
        '''
        super().__init__()
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            self._nrc(sample)
            return self._tsample(sample)
        else:
            return self._unchanged(sample)

    def _image(self, img):
        return tvf.vflip(img)

    def _mask(self, mask):
        return tvf.vflip(mask)

    def _coord(self, rcs):
        rlast = self.nrc[0] - 1
        trcs = np.copy(rcs)
        trcs[:, 0] = rlast - trcs[:, 0]
        return trcs

###########################################################################
class FilterGaussianRandom(Transform):
    def __init__(self, radius_range, prob=0.5):
        super().__init__()
        self.rads = radius_range
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            return self._tsample(sample)
        else:
            return self._unchanged(sample)

    def _image(self, img):
        return ImageFilter.GaussianBlur(random.randint(*self.rads))


class RandomSparseElastic(Transform):
    def __init__(self, grid_taps=(4, 4), magnitude=15, prob=0.5):
        super().__init__()
        self.ngt = grid_taps
        self.magnitude = magnitude
        self.prob = prob
        self.mesh = None

    def __call__(self, sample):
        if random.random() < self.prob:
            self._nrc(sample)
            self._mesh()
            return self._tsample(sample)
        else:
            return self._unchanged(sample)

    def _image(self, img):
        return img.transform(size=img.size, method=Image.MESH, data=self.mesh, resample=Image.BILINEAR)

    def _mask(self, mask):
        return mask.transform(size=mask.size, method=Image.MESH, data=self.mesh, resample=Image.NEAREST)

    def _coord(self, rcs):
        return rcs

    def _mesh(self):
        rtaps = np.linspace(0, self.nrc[0], self.ngt[0]).astype(np.int)
        ctaps = np.linspace(0, self.nrc[1], self.ngt[1]).astype(np.int)
        drtaps = np.random.randint(-self.magnitude, self.magnitude, rtaps.shape)
        drtaps[0], drtaps[-1] = 0, 0
        dctaps = np.random.randint(-self.magnitude, self.magnitude, ctaps.shape)
        dctaps[0], dctaps[-1] = 0, 0
        qrtaps, qctaps = (rtaps + drtaps), (ctaps + dctaps)

        nct, nrt = len(ctaps), len(rtaps)
        self.mesh = []
        for gr in range(nrt - 1):
            for gc in range(nct - 1):
                tr, br = rtaps[gr], rtaps[gr + 1]
                lc, rc = ctaps[gc], ctaps[gc + 1]
                bb = tuple([lc, tr, rc, br])
                tr, br = qrtaps[gr], qrtaps[gr + 1]
                lc, rc = qctaps[gc], qctaps[gc + 1]
                qc = [lc, tr, lc, br, rc, br, rc, tr]
                self.mesh.append([bb, qc])


class RandomDenseElastic(Transform):
    def __init__(self, alpha, sigma, border_clamp=True, prob=0.5):
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma
        self.border_clamp = border_clamp
        self.prob = prob
        self.rflow, self.cflow = None, None

    def __call__(self, sample):
        if random.random() < self.prob:
            self._nrc(sample)
            self._flow()
            return self._tsample(sample)
        else:
            return self._unchanged(sample)

    def _image(self, img):
        arr = np.array(img)
        if len(arr.shape) == 2:
            arr = arr.reshape(arr.shape + (-1,))
        tarr = [map_coordinates(arr[:, :, ch],
                                (self.cflow, self.rflow),
                                mode='constant',
                                order=2) for ch in range(arr.shape[-1])]
        shape = self.nrc
        if len(tarr) > 1:
            shape = self.nrc + (len(tarr),)
        return Image.fromarray(np.dstack(tarr).reshape(shape))

    def _mask(self, mask):
        arr = np.array(mask)
        if len(arr.shape) == 2:
            arr = arr.reshape(arr.shape + (-1,))
        tarr = [map_coordinates(arr[:, :, ch],
                                (self.cflow, self.rflow),
                                mode='constant',
                                order=0) for ch in range(arr.shape[-1])]
        shape = self.nrc
        if len(tarr) > 1:
            shape = self.nrc + (len(tarr),)
        return Image.fromarray(np.dstack(tarr).reshape(shape))

    def _coord(self, rcs):
        # One way to implement this is to create an image with value at rc being its index,
        # transform this image and read out locations of new points.
        return rcs

    def _flow(self):
        rfr, rfc = np.random.uniform(-1, 1, self.nrc)*self.alpha, np.random.uniform(-1, 1, self.nrc)*self.alpha
        dr, dc = gaussian_filter(rfr, self.sigma, mode='constant'), gaussian_filter(rfc, self.sigma, mode='constant')
        if self.border_clamp:
            dr, dc = self._border_clamp(dr), self._border_clamp(dc)
        r, c = np.meshgrid(np.arange(self.nrc[1]), np.arange(self.nrc[0]))
        self.rflow, self.cflow = (r + dr).reshape(-1, 1), (c + dc).reshape(-1, 1)

    def _border_clamp(self, a):
        a[0, :], a[-1, :], a[:, 0], a[:, -1] = 0.0, 0.0, 0.0, 0.0
        return a


class RandomAffine(Transform):
    def __init__(self, translate=None, rotate=None, shear=None, scale=None, ifillcolor=0, mfillcolor=0, prob=0.5):
        '''Angles in degrees, CLOCKWISE.
           Geometry params are tuples/lists of length 2.
           translate is (nrows%, ncols%) (between 0 and 1), other geometric params are ranges.
           TO DEBUG: read source of functional and transforms in torchvision and
           closed form inverse of 3X3 matrices online'''
        super().__init__()
        self.translate = translate
        self.rotate = rotate
        self.shear = shear
        self.scale = scale
        self.ifillcolor = ifillcolor
        self.mfillcolor = mfillcolor
        self.prob = prob
        self.fm = np.eye(3, dtype=np.float)
        self.im = []

    def __call__(self, sample):
        if random.random() < self.prob:
            self._nrc(sample)
            self._tmats(*self._random_params())
            return self._tsample(sample)
        else:
            return self._unchanged(sample)

    def _image(self, img):
        kwargs = {"fillcolor": self.ifillcolor} if PILLOW_VERSION[0] == '5' else {}
        return img.transform(self.nrc[::-1], Image.AFFINE, self.im, Image.BILINEAR, **kwargs)

    def _mask(self, mask):
        kwargs = {"fillcolor": self.mfillcolor} if PILLOW_VERSION[0] == '5' else {}
        return mask.transform(self.nrc[::-1], Image.AFFINE, self.im, Image.NEAREST, **kwargs)

    def _coord(self, rcs):
        crs = np.fliplr(np.array(rcs, dtype=np.float))
        crsaug = np.hstack((crs, np.ones((crs.shape[0], 1))))
        tcrs = np.dot(self.fm, crsaug.transpose()).transpose()
        return list(map(tuple, np.fliplr(tcrs[:, 0:2])))

    def _random_params(self):
        translate = (0.0, 0.0)
        if self.translate:
            dr, dc = self.translate[0]*self.nrc[0], self.translate[1]*self.nrc[1]
            translate = (np.round(random.uniform(-dr, dr)), np.round(random.uniform(-dc, dc)))

        angle = 0.0
        if self.rotate:
            angle = random.uniform(self.rotate[0], self.rotate[1])

        shear = 0.0
        if self.shear:
            shear = random.uniform(self.shear[0], self.shear[1])

        scale = 1.0
        if self.scale:
            scale = random.uniform(self.scale[0], self.scale[1])

        return translate, angle, shear, scale

    def _tmats(self, trn, ang, shr, scl):
        angle = math.radians(ang)
        shear = math.radians(shr)
        cent = (self.nrc[0]*0.5 + 0.5, self.nrc[1]*0.5 + 0.5)

        # specified translation
        tm = np.array([1.0, 0.0, trn[1],
                       0.0, 1.0, trn[0],
                       0.0, 0.0, 1.0], dtype=np.float).reshape(3, 3)
        tmi = tm.copy()
        tmi[0:2, 2] *= -1

        # translation to center
        cm = np.array([1.0, 0.0, cent[1],
                       0.0, 1.0, cent[0],
                       0.0, 0.0, 1.0], dtype=np.float).reshape(3, 3)
        cmi = cm.copy()
        cmi[0:2, 2] *= -1

        rss = np.array([math.cos(angle), -math.sin(angle + shear), 0.0,
                        math.sin(angle), math.cos(angle + shear), 0.0,
                        0.0, 0.0, 1.0], dtype=np.float).reshape(3, 3)
        rss[0:2, 0:2] *= scl

        det = math.cos(angle)*math.cos(angle + shear) + math.sin(angle)*math.sin(angle + shear)
        det *= (scl*scl)

        rssi = rss.copy()
        rssi[0, 0], rssi[1, 1] = rssi[1, 1], rssi[0, 0]
        rssi[0, 1] *= -1
        rssi[1, 0] *= -1
        rssi[0:2, 0:2] /= det

        self.fm = tm @ cm @ rss @ cmi
        self.im = list((cm @ rssi @ cmi @ tmi)[0:2, :].flatten())
