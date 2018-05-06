import torch.nn as nn


class EmptyLayer(nn.Module):

    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, *input):
        raise NotImplementedError


class DetectionLayer(nn.Module):

    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

    def forward(self, *input):
        raise NotImplementedError

