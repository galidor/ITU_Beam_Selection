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


class Lidar3D(nn.Module):
    def __init__(self):
        super(Lidar3D, self).__init__()
        dropout_prob = 0.3
        self.conv1 = nn.Conv3d(1, 10, (5, 11, 3), stride=2, padding=(2, 5, 1))
        self.bn1 = nn.BatchNorm3d(10)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(10, 20, (5, 11, 3), stride=1, padding=(2, 5, 1))
        self.bn2 = nn.BatchNorm3d(20)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv3d(20, 50, (5, 11, 3), stride=1, padding=(2, 5, 1))
        self.bn3 = nn.BatchNorm3d(20)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv3d(50, 50, (5, 11, 3), stride=(1, 1, 2), padding=(2, 5, 1))
        self.bn4 = nn.BatchNorm3d(20)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv3d(50, 20, (5, 11, 3), stride=(1, 1, 5), padding=(2, 5, 1))
        self.bn5 = nn.BatchNorm3d(20)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(20, 10, kernel_size=(3, 11), stride=(2, 5), padding=(1, 5))
        self.bn6 = nn.BatchNorm2d(10)
        self.relu6 = nn.ReLU()
        self.linear7 = nn.Linear(1000, 256)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = x.squeeze(4)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = x.view(-1, 1000)
        x = self.linear7(x)
        return x
