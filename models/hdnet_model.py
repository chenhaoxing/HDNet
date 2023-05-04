
import torch
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable
from .fMSE import MaskWeightedMSE

class HDNetModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L1']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['comp', 'real', 'output', 'mask', 'real_f', 'fake_f', 'bg', 'attentioned']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G']
        else:
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.relu = nn.ReLU()
        if self.isTrain:
            # define loss functions
            self.criterionL1 = MaskWeightedMSE(100)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr*opt.g_lr_ratio, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.comp = input['comp'].to(self.device)
        self.real = input['real'].to(self.device)
        self.mask = input['mask'].to(self.device)
        self.inputs = self.comp
        if self.opt.input_nc == 4:
            self.inputs = torch.cat([self.inputs, self.mask], 1)  # channel-wise concatenation
        self.real_f = self.real * self.mask
        self.bg = self.real * (1 - self.mask)

    def forward(self):
        self.output = self.netG(self.inputs, self.mask)
        self.fake_f = self.output * self.mask
        self.attentioned = self.output * self.mask + self.inputs[:,:3,:,:] * (1 - self.mask)
        self.harmonized = self.attentioned

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G_L1 = self.criterionL1(self.attentioned, self.real, self.mask) * self.opt.lambda_L1
        self.loss_G = self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
         # update G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights

