import torch.nn as nn
from models.MLP import MLP as mlp
from models.LeNet1d import LeNet as lenet
from models.CNN_1d import CNN as cnn1d
from models.wdcnn import WDCNN as wdcnn
from models.rsnet import RSNet as rsnet
# from models.TFCNN import TFCNN as tfcnn
# from models.TFCNNV5 import TFCNN as tfcnn
from models.TFCNNV6 import TFCNN as tfcnn
# from models.TFCNNV7 import TFCNN as tfcnn
from models.byol import BYOL_Model as byol_model
from models.moco import MoCo_Model as moco_model
from models.simclr import SimCLR_Model as simclr_model
from models.normal_contrast_moco import MoCo_Model as nc_moco_model
from models.normal_contrast_simclr import SimCLR_Model as nc_simclr_model
from models.physical_hint_simclr import SimCLR_Model as ph_simclr_model
from models.physical_hint_with_simclr import SimCLR_Model as phw_simclr_model
# from models.resnet import resnet18_1d, resnet50_1d # kernel_size=64, stride=16, padding=24
from models.Resnet1d import resnet18_1d, resnet50_1d # kernel_size=7, stride=2, padding=3
from models.mlp_head import MLPHead as mlphead 
from models.mlp_head import MLPHeadV2 as mlpheadv2
from models.finetune_ae import Finetine_ae as finetune_ae
import models.rwkdcae as rwkdcae
import models.dcae as dcae
import models.Ae1d as ae1d
import models.Dae1d as dae1d
import models.Sae1d as sae1d
import models.decoders as decoders
import models.DCGAN as dcgan
import models.DCGAN_SN as dcgan_sn


class Void(nn.Module):
    def __init__(self):
        super(Void, self).__init__()

    def forward(self, x):
        return x
