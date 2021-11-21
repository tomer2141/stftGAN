from generator import Generator
from discriminator import Discriminator
from mapper import Mapper
import torch
import pickle
import io
from utils import csv2dic, CPU_Unpickler, compute_gp_mapper, compute_gp
from args import cfg
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss

weights_best_geneerator_path = 'checkpoints/200500.pkl'
weights_best_mapper_path = 'checkpoints/mapper_new_data/4_0'

spectrograms_path = 'data/logspec_tal_tensor.pt'
train_len = 150000
batch = 64
learning_rate = cfg.learning_rate
betas = (0.9, 0.99)
epochs = 10000

netG = Generator()
netG.to(cfg.device)
netD = Discriminator()
netD.to(cfg.device)

# data = torch.load(weights_best_geneerator_path, map_location=cfg.device)
with open(weights_best_geneerator_path, 'rb') as f:
    data = CPU_Unpickler(f).load()

netG.load_state_dict(data['G'])
netG.eval()
netD.load_state_dict(data['D'])
netD.eval()

netM = Mapper()
netM.to(cfg.device)
netM.load_state_dict(torch.load(weights_best_mapper_path))
netM.eval()

dic = csv2dic(cfg.csv_path)
dic_train = {}
dic_test = {}
for key, value in dic.items():
    dic_train[key] = dic[key][:train_len].type(torch.FloatTensor)
    dic_test[key] = dic[key][train_len:].type(torch.FloatTensor)

spectrograms = torch.load(spectrograms_path)
spectrograms_train = spectrograms[:train_len]
spectrograms_test = spectrograms[train_len:]

train_data = torch.utils.data.TensorDataset(
    dic_train['24.osc2waveform'],
    dic_train['26.lfo1waveform'],
    dic_train['32.lfo1destination'],
    dic_train['30.lfo1amount'],
    dic_train['28.lfo1rate'],
    dic_train['3.cutoff'],
    dic_train['4.resonance'],
    spectrograms_train
)

# test_data = torch.utils.data.TensorDataset(
#     dic_test['24.osc2waveform'],
#     dic_test['26.lfo1waveform'],
#     dic_test['32.lfo1destination'],
#     dic_test['30.lfo1amount'],
#     dic_test['28.lfo1rate'],
#     dic_test['3.cutoff'],
#     dic_test['4.resonance'],
#     spectrograms_test
# )

train_loader = DataLoader(train_data, batch_size=batch, shuffle=True)
# test_loader = DataLoader(test_data, batch_size=batch, shuffle=True)

optimizer_mapper = Adam(netM.parameters(), lr=learning_rate, betas=betas)
optimizer_mapper.zero_grad()

for epoch in range(epochs):

    for idx, (hot1, hot2, hot3, hot4, hot5, hot6, hot7, specs) in enumerate(train_loader):
        d = {'24.osc2waveform': hot1, '26.lfo1waveform': hot2, '32.lfo1destination': hot3, '30.lfo1amount': hot4,
             '28.lfo1rate': hot5, '3.cutoff': hot6, '4.resonance': hot7}

        for param in cfg.param_buckets_tal:
            d[param] = d[param].to(cfg.device)
        specs = specs.to(cfg.device)

        out_mapper = netM(d)
        spec = netG(out_mapper).to(cfg.device)
        loss = torch.mean(torch.abs(spec - specs), (1, 2, 3)).mean()
        # + 10.0*compute_gp(netD, specs, spec).type(torch.FloatTensor)
        # cfg.gamma_gp * compute_gp_mapper(spec, specs)

        # if idx % 100 == 0:
        #     torch.save(netM.state_dict(), "checkpoints/mapper/%d_%d" % (epoch, idx))

        # loss.backward()
        # optimizer_mapper.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [MAE loss: %f]"
            % (epoch, epochs, idx, len(train_loader), loss.item()))

