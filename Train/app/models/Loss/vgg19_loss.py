import torch
import torch.nn as nn
import torchvision

class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer + 1)])

    def normalize(self, tensors, mean, std):
        if not torch.is_tensor(tensors):
            raise TypeError('tensor is not a torch image.')
        for tensor in tensors:
            for t, m, s in zip(tensor, mean, std):
                t.sub_(m).div_(s)
        return tensors

    def forward(self, x):
        # it image is gray scale then make it to 3 channel
        if x.size()[1] == 1:
            x = x.expand(-1, 3, -1, -1)
            
        # [-1: 1] image to  [0:1] image---------------------------------------------------(1)
        x = (x + 1) * 0.5
        
        # https://pytorch.org/docs/stable/torchvision/models.html
        x.data = self.normalize(x.data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return self.features(x)

# Feature extracting using vgg19
vgg19 = torchvision.models.vgg19(pretrained=True)
feature_extractor = FeatureExtractor(vgg19, feature_layer=35)
feature_extractor.eval()

class VGG19Loss(object):
    def __init__(self):
        global feature_extractor
        self.initialized = False
        self.feature_extractor = feature_extractor
        self.MSE = nn.MSELoss()

    def __call__(self, output, target, device):
        if self.initialized == False:
            self.feature_extractor = self.feature_extractor.to(device)
            self.MSE = self.MSE.to(device)
            self.initialized = True

        # [-1: 1] image to  [0:1] image---------------------------------------------------(2)
        output = (output + 1) * 0.5
        target = (target + 1) * 0.5

        output = self.feature_extractor(output)
        target = self.feature_extractor(target).data
        return self.MSE(output, target)