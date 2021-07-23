import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F
from .resnext import resnext101_32x8d,resnext50_32x4d
from .resattnet import AttentionNet_IR_56,AttentionNet_IRSE_56,AttentionNet_IR_92,AttentionNet_IRSE_92
from .resnet import ResNet_50,ResNet_101,ResNet_152
from .efficientnet import efficientnet
from .mobilenet import MobileFaceNet,MobileNetV2,MobileNetV3,MobileNeXt
from .ghost import GhostNet
from .hrnet import HRNet_W18_small,HRNet_W18_small_v2,HRNet_W18,HRNet_W30,HRNet_W32,HRNet_W40,HRNet_W44,HRNet_W48,HRNet_W64
from .densnet import densenet

class LbpBlock1(nn.Module):
    def __init__(self,out_channels,in_channels=3,res=True,scnd_der=False,act=torch.sign):
        super(LbpBlock, self).__init__()
        self.out_channels = out_channels
        self.act = act
        self.res = res
        self.scnd_der = scnd_der
        nb_param = 8 + 1*int(res)+4*int(scnd_der)
        lbp_ch = 5 # TODO: add this to parameters
        #self.w = nn.Parameter(torch.randn(out_channels,8))
        #self.ws = nn.Parameter(torch.randn(out_channels,4))
        self.w = nn.Parameter(torch.randn(in_channels,lbp_ch,8))
        self.wl = nn.Parameter(torch.randn(lbp_ch,out_channels))
        self.zp = nn.ZeroPad2d(1)
        # TODO: make weith for res x 
    def forward(self,x):
        x_o = []
        if self.res:
            x_o.append( x[:,:,1:-1,1:-1])  

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

        #x_o.append(torch.einsum('nilwh,ol->nowh',x_cat,torch.exp(self.w)))
        x_o.append(torch.einsum('nilwh,iml->nimwh',x_cat,torch.exp(self.w)))
        x_o[-1] = torch.einsum('nimwh,mo->nowh',x_o[-1],self.wl)

        if self.scnd_der:
            x_lst = []
            x_lst.append(self.act(x[:,:,0:-2,0:-2] + x[:,:,2:,2:] - 2*x[:,:,1:-1,1:-1]).unsqueeze(dim=2))
            x_lst.append(self.act(x[:,:,0:-2,1:-1] + x[:,:,2:,1:-1] - 2*x[:,:,1:-1,1:-1]).unsqueeze(dim=2))
            x_lst.append(self.act(x[:,:,0:-2,2:] + x[:,:,2:,0:-2] - 2*x[:,:,1:-1,1:-1]).unsqueeze(dim=2))
            x_lst.append(self.act(x[:,:,1:-1,0:-2] + x[:,:,1:-1,2:] - 2*x[:,:,1:-1,1:-1]).unsqueeze(dim=2))
            x_cat = torch.cat(x_lst,dim=2)
            x_o.append(torch.einsum('nilwh,ol->nowh',x_cat,torch.exp(self.ws)))
            
        out = torch.cat(x_o,dim=1)
        out = out.contiguous()
        return self.zp(out)
    def get_out_ch(self):
        return self.out_channels + 3*int(self.res)+self.out_channels*int(self.scnd_der)

heaviside = lambda x: torch.heaviside(x,torch.FloatTensor([0]))
class LbpBlock(nn.Module):# new version
    def __init__(self,out_channels,in_channels=3,res=True,scnd_der=False,act=heaviside):
        super(LbpBlock, self).__init__()
        self.out_channels = out_channels
        self.act = act
        self.res = res
        self.scnd_der = scnd_der
        self.w = nn.Parameter(torch.randn(out_channels,8))
        self.zp = nn.ZeroPad2d(1)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,bias=False)

    def forward(self,x):
        x_o = []
        if self.res:
            x_o.append(x)  
            
        x_lst = []
        x = self.conv1x1(x)
        x_lst.append(self.act(x[:,:,1:-1,1:-1]-x[:,:,0:-2,0:-2]).unsqueeze(dim=2))
        x_lst.append(self.act(x[:,:,1:-1,1:-1]-x[:,:,0:-2,1:-1]).unsqueeze(dim=2))
        x_lst.append(self.act(x[:,:,1:-1,1:-1]-x[:,:,0:-2,2:]).unsqueeze(dim=2))
        x_lst.append(self.act(x[:,:,1:-1,1:-1]-x[:,:,1:-1,0:-2]).unsqueeze(dim=2))
        x_lst.append(self.act(x[:,:,1:-1,1:-1]-x[:,:,2:,0:-2]).unsqueeze(dim=2))
        x_lst.append(self.act(x[:,:,1:-1,1:-1]-x[:,:,2:,1:-1]).unsqueeze(dim=2))
        x_lst.append(self.act(x[:,:,1:-1,1:-1]-x[:,:,2:,2:]).unsqueeze(dim=2))
        x_lst.append(self.act(x[:,:,1:-1,1:-1]-x[:,:,1:-1,2:]).unsqueeze(dim=2))
        x_cat = torch.cat(x_lst,dim=2)
        x_cat = torch.einsum('nolwh,ol->nowh',x_cat,torch.exp(self.w))
        x_o.append(self.zp(x_cat))

        out = torch.cat(x_o,dim=1)
        return out.contiguous()
    def get_out_ch(self):
        return self.out_channels + 3*int(self.res)+self.out_channels*int(self.scnd_der)

class Identity(nn.Module):
      def __init__(self):
        super(Identity, self).__init__()
      def forward(self,x):
        return x



class Model(nn.Module):
    def __init__(self,cfg):
      super(Model, self).__init__()
      self.use_lbp = cfg.use_lbp 
      self.emb_size = cfg.emb_size
      if cfg.use_lbp:
        self.lbp = LbpBlock(cfg.lbp_ch)
        nb_ch = self.lbp.get_out_ch()

      else:
        self.lbp = Identity()
        nb_ch = 3

      self.backbone = get_backbone(cfg,nb_ch)
      self.head = nn.Linear(self.emb_size,1,bias = False)

    def forward(self,x):
        x = self.lbp(x)
        emb = self.backbone(x)
        return self.head(emb),emb       


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
  elif cfg.backbone == 'ResNet_101':
    model = ResNet_101(input_size,emb_size=cfg.emb_size,nb_ch=nb_ch)
  elif cfg.backbone == 'ResNet_152':
    model = ResNet_152(input_size,emb_size=cfg.emb_size,nb_ch=nb_ch)
  elif cfg.backbone == 'efficientnet':
    model = efficientnet(input_size,emb_size=cfg.emb_size,nb_ch=nb_ch)
  elif cfg.backbone == 'MobileNetV2':
    model = MobileNetV2(input_size,emb_size=cfg.emb_size,nb_ch=nb_ch)
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
  return model #densenet