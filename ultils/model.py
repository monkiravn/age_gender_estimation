import torchvision.models as models
from torch import nn
import torch

class inception_V3(nn.Module):
    def __init__(self):
        super(inception_V3, self).__init__()
        self.model = models.inception_v3(pretrained=True)
        self.fc = nn.Linear(512, 100)  # For age class
        self.fc1 = nn.Linear(512, 2)  # For gender class

    def forward(self, x):
        label1 = self.fc1(x)
        label2 = torch.sigmoid(self.fc2(x))
        return {'label1': label1, 'label2': label2}

class Densenet(nn.Module):

    def __init__(self):
        super(Densenet, self).__init__()
        self.model = models.densenet121(pretrained=True)
        num_ftrs = self.model.classifier.in_features
        self.fc = nn.Linear(num_ftrs, 100)  # For age class
        self.fc1 = nn.Linear(num_ftrs, 2)  # For gender class

    def forward(self, x):
        label1 = self.fc1(x)
        label2 = torch.sigmoid(self.fc2(x))

        return {'label1': label1, 'label2': label2}

if __name__ == '__main__':
    model = Densenet()