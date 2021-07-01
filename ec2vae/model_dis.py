# EC2VAE
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

class RECVAE(nn.Module):
    def __init__(self, input_dims, hidden_dims,rhythm_dims,
    z1_dims,z2_dims,seq_len, decay = 1000):
        super(RECVAE, self).__init__()
        # encoder
        self.encoder_gru = nn.GRU(input_dims,hidden_dims,batch_first=True,bidirectional=True)
        self.linear_mu = nn.Linear(hidden_dims * 2, z1_dims + z2_dims)
        self.linear_var = nn.Linear(hidden_dims * 2, z1_dims + z2_dims)
        # rhythm decoder
        self.rdecoder_0 = nn.GRUCell(z2_dims + rhythm_dims,hidden_dims)
        self.rdecoder_hidden_init = nn.Linear(z2_dims, hidden_dims)
        self.rdecoder_out = nn.Linear(hidden_dims, rhythm_dims)
        # whole decoder
        self.decoder_0 = nn.GRUCell(z1_dims + input_dims + rhythm_dims, hidden_dims)
        self.decoder_1 = nn.GRUCell(hidden_dims,hidden_dims)
        self.decoder_hidden_init = nn.Linear(z1_dims, hidden_dims)
        self.decoder_out = nn.Linear(hidden_dims, input_dims)
        # parameter initialization
        self.z1_dims = z1_dims
        self.z2_dims = z2_dims
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.rhythm_dims = rhythm_dims
        self.seq_len = seq_len    
        # input
        self.x = None
        self.rx = None
        # teacher forcing hyperparameters
        self.iteration = 0
        self.eps = 1.0
        self.decay = torch.FloatTensor([decay])

    def _findmax(self, x):
        argx = x.argmax(1)
        x = torch.zeros_like(x)
        line = torch.arange(x.size(0)).long()
        if torch.cuda.is_available():
            line = line.cuda()
        x[line, argx] = 1.0
        return x
    def encoder(self,x):
        x = self.encoder_gru(x)[-1]
        # print(x.size())
        x = x.transpose_(0,1).contiguous()
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        var = self.linear_var(x).exp_()
        dis1 = Normal(mu[:,:self.z1_dims], var[:,:self.z1_dims])
        dis2 = Normal(mu[:,self.z1_dims:], var[:,self.z1_dims:])
        return dis1,dis2
    def rhythm_deocder(self, z):
        y = torch.zeros((z.size(0), self.rhythm_dims))
        # for y_-1, it is rest
        y[:, -1] = 1
        ys = []
        h0 = torch.tanh(self.rdecoder_hidden_init(z))
        hx = h0
        if torch.cuda.is_available():
            y = y.cuda()
        for i in range(self.seq_len):
            y = torch.cat([y, z], 1)
            hx = self.rdecoder_0(y, hx)
            y = F.log_softmax(self.rdecoder_out(hx), 1)
            ys.append(y)
            if self.training:
                p = torch.rand(1).item()
                if p < self.eps:
                    y = self.rx[:,i,:]
                else:
                    y = self._findmax(y)
            else:
                y = self._findmax(y)
        return torch.stack(ys,1)
    def decoder(self, z, rhythm):
        y = torch.zeros((z.size(0),self.input_dims))
        # for y_-1, it is rest
        y[:, -1] = 1
        ys = []
        cors = []
        h0 = torch.tanh(self.decoder_hidden_init(z))
        hx = [None, None]
        hx[0] = h0
        if torch.cuda.is_available():
            y = y.cuda()
        for i in range(self.seq_len):
            # print(rhythm.size())
            y = torch.cat([y, rhythm[:,i,:],z], 1)
            hx[0] = self.decoder_0(y, hx[0])
            if i == 0:
                # next hidden state first input if the first output of last state
                hx[1] = hx[0]
            hx[1] = self.decoder_1(hx[0],hx[1])
            y = F.log_softmax(self.decoder_out(hx[1]), 1)
            cors.append(self._findmax(y))
            ys.append(y)
            if self.training:
                p = torch.rand(1).item()
                if p < self.eps:
                    y = self.x[:,i,:]
                else:
                    y = self._findmax(y)
                # update the eps after one batch
                self.eps = self.decay / (self.decay + torch.exp(self.iteration / self.decay))
            else:
                y = self._findmax(y)
        return torch.stack(ys, 1), torch.stack(cors, 1)    
    def output_decoder(self, z1, z2):
#         print("vae output decoder", self.training)
        recon_rhythm = self.rhythm_deocder(z2)
        _, recon_x = self.decoder(z1, recon_rhythm)
        return recon_x
    def output_decoder_none(self, z1, z2):
#         print("vae output decoder none", self.training)
        recon_rhythm = self.rhythm_deocder(z2)
        recon_x,_ = self.decoder(z1, recon_rhythm)
        return recon_x
    def forward(self,x):
#         print("vae forward", self.training)
        if self.training:
            self.x = x
            self.rx = x[:,:,:-2].sum(-1).unsqueeze(-1)
            self.rx = torch.cat((self.rx,x[:,:,-2:]),-1)
            self.iteration += 1
        dis1, dis2 = self.encoder(x)
        z1 = dis1.rsample()
        z2 = dis2.rsample()
        recon_rhythm = self.rhythm_deocder(z2)
        recon_x, _ = self.decoder(z1, recon_rhythm)
        output = (recon_x, recon_rhythm, dis1, dis2)
        return output


