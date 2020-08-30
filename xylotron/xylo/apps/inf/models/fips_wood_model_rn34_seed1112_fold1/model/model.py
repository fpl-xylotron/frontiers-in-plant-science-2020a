from pathlib import Path
from image_resnet_transfer_classifier import ImageResNetTransferClassifier

def get(cnames, wts):
    with open(cnames, 'r') as fh:
        lines = [line.strip() for line in fh.readlines()]
    cls = ImageResNetTransferClassifier(num_classes=len(lines))
    cls.load_weights(Path(wts))

    return cls, lines
