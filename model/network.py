import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F
from .resnext import resnext101_32x8d,resnext50_32x4d
from .resattnet import AttentionNet_IR_56,AttentionNet_IRSE_56,AttentionNet_IR_92,AttentionNet_IRSE_92
from .resnet import ResNet_18,ResNet_50,ResNet_101,ResNet_152
from .efficientnet import EfficientNet
from .mobilenet import MobileFaceNet,MobileNetV2,MobileNetV3,MobileNeXt
from .ghost import GhostNet
from .hrnet import HRNet_W18_small,HRNet_W18_small_v2,HRNet_W18,HRNet_W30,HRNet_W32,HRNet_W40,HRNet_W44,HRNet_W48,HRNet_W64
from .densnet import densenet
from .alexnet import AlexNet,AlexNetLite
from .vgg import VGG_16
import torchvision.models as models


class Identity(nn.Module):
      def __init__(self):
        super(Identity, self).__init__()
      def forward(self,x):
        return x


heaviside = lambda x: torch.heaviside(x,torch.zeros_like(x))


class LbpLayer(nn.Module):
    def __init__(self,in_planes, out_planes, stride=1, groups=1, dilation=1,act=heaviside):
        super(LbpLayer, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.act = act
        self.lrn = nn.LocalResponseNorm(size=3, alpha=0.0001, beta=0.75, k=2)
        self.w = nn.Parameter(torch.empty(out_planes,1,8,1,1))
        nn.init.normal_(self.w,std=0.1)
        self.zp = nn.ZeroPad2d(1)
        
    def forward(self,x):
        x_lst = []
        x_lst.append(self.act(x[:,:,1:-1,1:-1]-x[:,:,0:-2,0:-2]).unsqueeze(dim=2))
        x_lst.append(self.act(x[:,:,1:-1,1:-1]-x[:,:,0:-2,1:-1]).unsqueeze(dim=2))
        x_lst.append(self.act(x[:,:,1:-1,1:-1]-x[:,:,0:-2,2:]).unsqueeze(dim=2))
        x_lst.append(self.act(x[:,:,1:-1,1:-1]-x[:,:,1:-1,0:-2]).unsqueeze(dim=2))
        x_lst.append(self.act(x[:,:,1:-1,1:-1]-x[:,:,2:,0:-2]).unsqueeze(dim=2))
        x_lst.append(self.act(x[:,:,1:-1,1:-1]-x[:,:,2:,1:-1]).unsqueeze(dim=2))
        x_lst.append(self.act(x[:,:,1:-1,1:-1]-x[:,:,2:,2:]).unsqueeze(dim=2))
        x_lst.append(self.act(x[:,:,1:-1,1:-1]-x[:,:,1:-1,2:]).unsqueeze(dim=2))
        x_cat = torch.cat(x_lst,dim=2)
        
        y = F.conv3d(x_cat,torch.exp(self.w),
            stride=1, padding=0, 
            dilation=1, groups=self.in_planes).squeeze()
        
        return self.zp(y) #+ x #self.lrn(y)

class Model1(nn.Module):
    def __init__(self,cfg):
        super(Model1, self).__init__()
        self.use_lbp = cfg.use_lbp 
        self.emb_size = cfg.emb_size
        if self.use_lbp:
            self.lbp = LbpLayer(3,3)
            nb_ch = 3 # self.lbp.get_out_ch()

        else:
            self.lbp = Identity()
            nb_ch = 3

        self.backbone = get_backbone(cfg,3)
        # self.backbone =Inception_v1()# ResNetSe()
        # self.backbone = CDCNpp() #CDCNpp()
        # self.backbone = LbpNet()
        # self.backbone = models.resnet18(pretrained=True)
        # self.backbone = models.densenet121(pretrained=True)
        # self.backbone = models.resnet34(pretrained=True)
        # self.backbone.classifier = nn.Sequential()
        # self.backbone.fc = Identity()
        # self.backbone = ResNetCbam(BasicBlock, [2, 2, 2, 2], None, 1, 'CBAM')
        # self.backbone = models.resnext50_32x4d()
        # self.backbone.layer3 = nn.Sequential()
        # self.backbone.layer4 = nn.Sequential()
        # self.backbone.fc = nn.Sequential()
        # self.backbone =models.mobilenet_v2(pretrained=True)
        # self.backbone.classifier = nn.Sequential()

    #   self.lm = nn.LayerNorm(self.emb_size,eps=1e-12)
        
        self.head = nn.Linear(self.emb_size,1,bias = False)
        # nn.init.xavier_normal_(self.head.weight)
        nn.init.kaiming_normal_(self.head.weight)
        self.bn1d = nn.BatchNorm1d(cfg.emb_size, affine=False)
    def forward(self,x):       
        x = self.lbp(x)
        emb = self.backbone(x)

        emb = F.normalize(emb)
        self.head.weight = nn.Parameter(F.normalize(self.head.weight))
        # with torch.no_grad():
        #     self.head.weight = nn.Parameter(F.normalize(self.head.weight))
            # self.head.weight.div_(torch.norm(self.head.weight, dim=1, keepdim=True))

        return self.head(F.dropout(emb, p=0.5)),emb




# class Model(nn.Module):
#     def __init__(self,cfg):
#       super(Model, self).__init__()
#       self.use_lbp = cfg.use_lbp 
#       self.emb_size = cfg.emb_size
#       if self.use_lbp:
#         self.lbp = LbpBlock(cfg.lbp_ch)
#         nb_ch = self.lbp.get_out_ch()

#       else:
#         self.lbp = Identity()
#         nb_ch = 3

#       self.backbone = get_backbone(cfg,nb_ch)
#       self.lm = nn.LayerNorm(self.emb_size,eps=1e-12)
#       self.head = nn.Linear(self.emb_size,1,bias = False)

#     def forward(self,x):
#         # x = self.lbp(x)
#         emb = self.backbone(x)
#         emb = F.normalize(emb,eps=1e-6)
#         self.head.weight = nn.Parameter(F.normalize(self.head.weight))
#         # emb = self.lm(emb)
#         # emb = F.dropout(emb, p=0.2)
#         return 2.6*self.head(emb),emb       


def get_backbone(cfg,nb_ch):
  input_size = [cfg.input_size,cfg.input_size]
  if cfg.backbone == 'resnext50_32x4d':
    model = resnext50_32x4d(input_size,cfg.emb_size,nb_ch=nb_ch)
  elif cfg.backbone == 'resnext101_32x8d':
    model = resnext101_32x8d(input_size,cfg.emb_size,nb_ch=nb_ch)
  elif cfg.backbone == 'AttentionNet_IR_56':#TODO: fix for 224 size
    model = AttentionNet_IR_56(input_size,cfg.emb_size,nb_ch=nb_ch)
  elif cfg.backbone == 'AttentionNet_IRSE_56':#TODO: fix for 224 size
    model = AttentionNet_IRSE_56(input_size,cfg.emb_size,nb_ch=nb_ch)
  elif cfg.backbone == 'AttentionNet_IR_92':#TODO: fix for 224 size
    model = AttentionNet_IR_92(input_size,cfg.emb_size,nb_ch=nb_ch)
  elif cfg.backbone == 'AttentionNet_IRSE_92':#TODO: fix for 224 size
    model = AttentionNet_IRSE_92(input_size,cfg.emb_size,nb_ch=nb_ch)
  elif cfg.backbone == 'ResNet_50':
    model = ResNet_50(input_size,emb_size=cfg.emb_size,nb_ch=nb_ch)
  elif cfg.backbone == 'ResNet_188':
    model =resnet188()
    return model
  elif cfg.backbone == 'ResNet_18':
    model = models.resnet18(pretrained=False)
    model.fc = Identity()
  elif cfg.backbone == 'ResNet_101':
    model = ResNet_101(input_size,emb_size=cfg.emb_size,nb_ch=nb_ch)
  elif cfg.backbone == 'ResNet_152':
    model = ResNet_152(input_size,emb_size=cfg.emb_size,nb_ch=nb_ch)
  elif cfg.backbone == 'efficientnet':

    # model = EfficientNet.from_pretrained('efficientnet-b0')
    # model = EfficientNet.from_name('efficientnet-b0')
    # model._dropout = nn.Sequential()
    # model._fc = nn.Linear(1280,cfg.emb_size)
    # model._fc = nn.Sequential()
    # model = efficientnet(input_size,emb_size=cfg.emb_size,nb_ch=nb_ch)
    model =  models.efficientnet_b0(pretrained=True)
    # model.features[8] = nn.Sequential()
    model.classifier = Identity()
  elif cfg.backbone == 'MobileNetV2':
    model =models.mobilenet_v2(pretrained=False)
    model.classifier = nn.Sequential() #[18][2]
    # model.features[18] = nn.Sequential()
    # model.classifier[1] = nn.Linear(1280,cfg.emb_size)
    # model = MobileNetV2(input_size,emb_size=cfg.emb_size,nb_ch=nb_ch)
  elif cfg.backbone == 'MobileNetV3':
    model = MobileNetV3(input_size,emb_size=cfg.emb_size,nb_ch=nb_ch)
  elif cfg.backbone == 'MobileNeXt':
    model = MobileNeXt(input_size,emb_size=cfg.emb_size,nb_ch=nb_ch)
  elif cfg.backbone == 'MobileFaceNet':
    model = MobileFaceNet(input_size,emb_size=cfg.emb_size,nb_ch=nb_ch)
  elif cfg.backbone == 'GhostNet':
    model = GhostNet(input_size,emb_size=cfg.emb_size,nb_ch=nb_ch)
  elif cfg.backbone == 'HRNet_W18_small':
    model = HRNet_W18_small(input_size,emb_size=cfg.emb_size,nb_ch=nb_ch)
  elif cfg.backbone == 'HRNet_W18_small_v2':
    model = HRNet_W18_small_v2(input_size,emb_size=cfg.emb_size,nb_ch=nb_ch)
  elif cfg.backbone == 'HRNet_W18':
    model = HRNet_W18(input_size,emb_size=cfg.emb_size,nb_ch=nb_ch)
  elif cfg.backbone == 'HRNet_W30':
    model = HRNet_W30(input_size,emb_size=cfg.emb_size,nb_ch=nb_ch)
  elif cfg.backbone == 'HRNet_W32':
    model = HRNet_W32(input_size,emb_size=cfg.emb_size,nb_ch=nb_ch)
  elif cfg.backbone == 'HRNet_W40':
    model = HRNet_W40(input_size,emb_size=cfg.emb_size,nb_ch=nb_ch)
  elif cfg.backbone == 'HRNet_W44':
    model = HRNet_W44(input_size,emb_size=cfg.emb_size,nb_ch=nb_ch)
  elif cfg.backbone == 'HRNet_W48':
    model = HRNet_W48(input_size,emb_size=cfg.emb_size,nb_ch=nb_ch)    
  elif cfg.backbone == 'HRNet_W64':
    model = HRNet_W64(input_size,emb_size=cfg.emb_size,nb_ch=nb_ch)  
  elif cfg.backbone == 'densenet':
    model = densenet(input_size,emb_size=cfg.emb_size,nb_ch=nb_ch)    
  elif cfg.backbone == 'alexnet':
    model = AlexNetLite(emb_size=cfg.emb_size,nb_ch=nb_ch)
  elif cfg.backbone == 'vgg':
    model = VGG_16(emb_size=cfg.emb_size,nb_ch=nb_ch)
  return model #densenet

  