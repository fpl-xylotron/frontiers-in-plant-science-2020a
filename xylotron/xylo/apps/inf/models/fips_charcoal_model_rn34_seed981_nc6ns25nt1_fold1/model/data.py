import PIL
from PIL import Image
from torchvision import transforms
import image_augmentations as ia

def tensorify(arr):
    divfac = 4
    resize_size = (2048//divfac, 2048//divfac)
    xfm = transforms.Compose([ia.PadToEnsureSize(out_size=(2048, 2048)),
                              ia.Resize(out_size=resize_size),
                              ia.ToTensor(),
                              ia.ImageNetNormalize()])

    image = Image.fromarray(arr.astype('uint8'), 'RGB')
    sample = {'image': (image, ia.SampElemType.IMAGE)}
    sample = xfm(sample)

    return sample['image'][0].unsqueeze(0)
