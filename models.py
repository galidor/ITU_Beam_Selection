from torch import nn
from torch.nn import functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class LidarMarcus(nn.Module):
    def __init__(self):
        super(LidarMarcus, self).__init__()
        self.layers = nn.Sequential(nn.Conv2d(10, 10, kernel_size=13, stride=1, padding=6),
                                    nn.ReLU(),
                                    nn.Conv2d(10, 30, kernel_size=11, stride=1, padding=5),
                                    nn.ReLU(),
                                    nn.Conv2d(30, 25, kernel_size=9, stride=1, padding=4),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=(2, 1)),
                                    nn.Dropout(0.3),
                                    nn.Conv2d(25, 20, kernel_size=7, stride=1, padding=3),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=(1, 2)),
                                    nn.Conv2d(20, 15, kernel_size=5, stride=1, padding=2),
                                    nn.ReLU(),
                                    nn.Dropout(0.3),
                                    nn.Conv2d(15, 10, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(10, 1, kernel_size=1, stride=1),
                                    nn.ReLU(),
                                    Flatten(),
                                    nn.Linear(1000, 256))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        return self.layers(x)