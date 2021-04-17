import torchvision.models as models
from torch import nn
import torch
from torch.nn import functional as F

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
        self.classifier = nn.Linear(num_ftrs, 100)  # For age class
        self.classifier2 = nn.Linear(num_ftrs, 2)  # For gender class

    def forward(self, x):
        label1 = self.classifier(x)
        label2 = torch.sigmoid(self.classifier2(x))

        return {'label1': label1, 'label2': label2}



class ResnetV1(nn.Module):
    def __init__(self):
        super(ResnetV1, self).__init__()
        original_model = models.resnet34(pretrained="True")
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        self.apply_log_soft = nn.LogSoftmax(dim=1)
        self.fc1 = nn.Linear(512, 1)  # For age class
        self.fc2 = nn.Linear(512, 2)  # For gender class

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        label1 = torch.sigmoid(self.fc1(x))
        label2 = self.apply_log_soft(self.fc2(x))
        return {'label1': label1, 'label2': label2}

class ResnetV2(nn.Module):
    def __init__(self):
        super(ResnetV2, self).__init__()
        original_model = models.resnet34(pretrained="True")
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        self.apply_log_soft = nn.LogSoftmax(dim=1)
        self.fc1 = nn.Linear(512, 100)  # For age class
        self.fc2 = nn.Linear(512, 2)  # For gender class

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        label1 = self.apply_log_soft(self.fc1(x))
        label2 = self.apply_log_soft(self.fc2(x))
        return {'label1': label1, 'label2': label2}

if __name__ == '__main__':
    from torchsummary import summary
    model = ResnetV1()
    if torch.cuda.is_available():
        model.cuda()
    summary(model, input_size=(3, 224, 224))
