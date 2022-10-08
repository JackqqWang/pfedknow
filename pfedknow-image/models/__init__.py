# from .simsiam import SimSiam
from .byol import BYOL, global_net,global_net2
from torchvision.models import resnet50, resnet18
import torch
from .backbones import *

def get_backbone(backbone, castrate=True):
    backbone = eval(f"{backbone}()")

    # if castrate:
    #     backbone.output_dim = backbone.fc1.in_features
    #     backbone.fc1 = torch.nn.Identity()

    return backbone

def get_pruned_backbone(backbone, cfg=None, dataset=None, castrate=True):

    backbone = eval(f"{backbone}({cfg},dataset='{dataset}')")

    # if castrate:
    #     backbone.output_dim = backbone.fc1.in_features
    #     backbone.fc1 = torch.nn.Identity()

    return backbone

class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class CNN_Cifar_pruned(nn.Module):
    def __init__(self, cfg=None,dataset='cifar10', init_weights=True):
        super(CNN_Cifar_pruned, self).__init__()
        if cfg is None:
            cfg = [32, 'M', 128, 'M', 256]
        self.feature = self.make_layers(cfg, True)
        # self.output_dim = cfg[-1] ## TODO adhoc
        if dataset == 'cifar100':
            self.num_classes = 100
        elif dataset == 'cifar10':
            self.num_classes = 10
            self.num_feature=512
        else:
            self.num_classes = 10
        self.num_feature = 10
        self.classifier = nn.Linear(16*cfg[-1], self.num_feature)
        self.output_dim = self.num_feature
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False): # make layers M is max pooling
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

# class CNN_Cifar_pruned(nn.Module):
#     def __init__(self, cfg=None,dataset=None):
#         super(CNN_Cifar_pruned, self).__init__()
#         if(cfg==None):
#             cfg = [32, 'M', 128, 'M', 256]
#         self.conv_layer = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=cfg[0], kernel_size=3, padding=1),
#             nn.BatchNorm2d(cfg[0]),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=cfg[0], out_channels=64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(in_channels=64, out_channels=cfg[2], kernel_size=3, padding=1),
#             nn.BatchNorm2d(cfg[2]),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=cfg[2], out_channels=128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout2d(p=0.05),
#             nn.Conv2d(in_channels=128, out_channels=cfg[-1], kernel_size=3, padding=1),
#             nn.BatchNorm2d(cfg[-1]),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=cfg[-1], out_channels=256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             # nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#         self.fc_layer = nn.Sequential(
#             nn.Dropout(p=0.1),
#             nn.Linear(4096, 1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, 512),
#             nn.ReLU(inplace=True)
#             # nn.Dropout(p=0.1),
#             # nn.Linear(512, 10)
#         )
#         self.output_dim = 512
#         self.num_classes=10
#         self._initialize_weights()
#
#     def forward(self, x):
#         x = self.conv_layer(x)
#         x = nn.AvgPool2d(2)(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc_layer(x)
#         return x
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(0.5)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_()
class CNNCifar(nn.Module):
    def __init__(self):
        super(CNNCifar, self).__init__()
        self.num_classes=10
        self.output_dim=512
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True)
            # nn.Dropout(p=0.1),
            # nn.Linear(512, 10)
        )
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x




def get_model(name, backbone, cfg=None, dataset=None):
    if name == 'local':
        model = BYOL(get_backbone(backbone))
    elif name == 'global':
        if cfg!=None:
            model = global_net(get_pruned_backbone(backbone,dataset=dataset))
        else:
            if dataset == 'mnist':
                model = global_net2(CNN_BN_Mnist(dataset=None))
            elif dataset =='cifar':
                model = global_net2(CNN_Cifar_pruned(cfg=None,dataset=None))
            else:
                model = global_net2(get_pruned_backbone(backbone,dataset=dataset))
    elif name == 'fedfixmatch' and backbone == 'Mnist':
        model = CNNMnist().to('cuda')
    elif name == 'fedfixmatch' and backbone == 'Cifar':
        model = CNNCifar().to('cuda')
    elif name == 'fedfixmatch' and backbone == 'Svhn':
        model = CNNCifar().to('cuda')
    elif name == 'fedfixmatch' and backbone == 'Vgg_backbone':
        model = global_net2(get_backbone(backbone))
    else:
        raise NotImplementedError
    return model






