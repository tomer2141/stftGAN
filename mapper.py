import numpy as np
import torch
import torch.nn as nn
from args import cfg


# Input: (BatchSize,256,128,1)
class Mapper0(nn.Module):
    def __init__(self):
        super(Mapper0, self).__init__()
        out_dim = cfg.mapper_fc_out_dim
        self.numOfParams = cfg.num_params
        self.fc24 = nn.Linear(4, out_dim)
        self.fc26 = nn.Linear(4, out_dim)
        self.fc32 = nn.Linear(2, out_dim)
        self.fc30 = nn.Linear(16, out_dim)
        self.fc28 = nn.Linear(16, out_dim)
        self.fc3 = nn.Linear(16, out_dim)
        self.fc4 = nn.Linear(16, out_dim)
        self.concat2latent = nn.Linear(self.numOfParams * out_dim, cfg.gen_latent_dim)

    def forward(self, dict):
        out24 = self.fc24(dict['24.osc2waveform'])
        out26 = self.fc26(dict['26.lfo1waveform'])
        out32 = self.fc32(dict['32.lfo1destination'])
        out30 = self.fc30(dict['30.lfo1amount'])
        out28 = self.fc28(dict['28.lfo1rate'])
        out3 = self.fc3(dict['3.cutoff'])
        out4 = self.fc4(dict['4.resonance'])
        concatenated = torch.cat((out24, out26, out32, out30, out28, out3, out4), 1)
        latent = self.concat2latent(concatenated)

        return latent
        

class Mapper(nn.Module):
    def __init__(self):
        super(Mapper, self).__init__()
        out_dim = cfg.mapper_fc_out_dim
        before_relu_dim = 50
        self.numOfParams = cfg.num_params
        self.fc24 = nn.Linear(4, out_dim)
        self.fc26 = nn.Linear(4, out_dim)
        self.fc32 = nn.Linear(2, out_dim)
        self.fc30 = nn.Linear(16, out_dim)
        self.fc28 = nn.Linear(16, out_dim)
        self.fc3 = nn.Linear(16, out_dim)
        self.fc4 = nn.Linear(16, out_dim)
        self.concat2relu = nn.Linear(self.numOfParams * out_dim, before_relu_dim)
        self.concat2latent = nn.Linear(before_relu_dim, cfg.gen_latent_dim)
        self.relu = nn.ReLU()

    def forward(self, dict):
        out24 = self.fc24(dict['24.osc2waveform'])
        out26 = self.fc26(dict['26.lfo1waveform'])
        out32 = self.fc32(dict['32.lfo1destination'])
        out30 = self.fc30(dict['30.lfo1amount'])
        out28 = self.fc28(dict['28.lfo1rate'])
        out3 = self.fc3(dict['3.cutoff'])
        out4 = self.fc4(dict['4.resonance'])
        concatenated = torch.cat((out24, out26, out32, out30, out28, out3, out4), 1)
        outA = self.concat2relu(concatenated)
        relu_outA = self.relu(outA)
        latent = self.concat2latent(relu_outA)

        return latent


# Input: (BatchSize,256,128,1)
class MapperBoxMuller(nn.Module):
    def __init__(self):
        super(MapperBoxMuller, self).__init__()
        out_dim = cfg.mapper_fc_out_dim
        self.numOfParams = cfg.num_params
        self.fc24 = nn.Linear(4, out_dim)
        # self.fc24_2 = nn.Linear(4, out_dim)
        self.fc26 = nn.Linear(4, out_dim)
        # self.fc26_2 = nn.Linear(4, out_dim)
        self.fc32 = nn.Linear(2, out_dim)
        # self.fc32_2 = nn.Linear(2, out_dim)
        self.fc30 = nn.Linear(16, out_dim)
        # self.fc30_2 = nn.Linear(16, out_dim)
        self.fc28 = nn.Linear(16, out_dim)
        # self.fc28_2 = nn.Linear(16, out_dim)
        self.fc3 = nn.Linear(16, out_dim)
        # self.fc3_2 = nn.Linear(16, out_dim)
        self.fc4 = nn.Linear(16, out_dim)
        # self.fc4_2 = nn.Linear(16, out_dim)
        self.concat2latent_1 = nn.Linear(self.numOfParams * out_dim, cfg.gen_latent_dim)
        self.concat2latent_2 = nn.Linear(self.numOfParams * out_dim, cfg.gen_latent_dim)

    def forward(self, dict):
        out24 = self.fc24(dict['24.osc2waveform'])
        out26 = self.fc26(dict['26.lfo1waveform'])
        out32 = self.fc32(dict['32.lfo1destination'])
        out30 = self.fc30(dict['30.lfo1amount'])
        out28 = self.fc28(dict['28.lfo1rate'])
        out3 = self.fc3(dict['3.cutoff'])
        out4 = self.fc4(dict['4.resonance'])
        concatenated = torch.cat((out24, out26, out32, out30, out28, out3, out4), 1)
        #U1 = 1/(1+torch.exp(-self.concat2latent_1(concatenated)))
        #U2 = 1/(1+torch.exp(-self.concat2latent_2(concatenated)))
        U1 = nn.Softmax(dim=1)(self.concat2latent_1(concatenated))
        U2 = nn.Softmax(dim=1)(self.concat2latent_2(concatenated))
        R = torch.sqrt(-2*torch.log(U1+torch.finfo(torch.float32).eps))
        Theta = 2*torch.pi*U2
        latent = R * torch.cos(Theta)
        return latent


