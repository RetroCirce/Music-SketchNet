# music inpaintNet rnn
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

class InpaintingNet(nn.Module):
    def __init__(self, input_dims, 
        pf_hidden_dims, g_h_dims, pf_num, inpaint_len, vae_model = None, train_mse = True, decay = 1000,teacher_forcing = True):
        super(InpaintingNet, self).__init__()
        # Past RNN
        self.past_gru = nn.GRU(input_dims, pf_hidden_dims, 
            pf_num, batch_first = True, bidirectional = True, dropout = 0.5)
        # Future RNN
        self.future_gru = nn.GRU(input_dims, pf_hidden_dims, 
            pf_num, batch_first = True, bidirectional = True, dropout = 0.5)
        # generation RNN
        self.generation_gru = nn.GRU(input_dims, g_h_dims, pf_num, batch_first = True, bidirectional = True, dropout = 0.5)
        self.linear_out = nn.Linear(g_h_dims * pf_num, input_dims)
        # parameter initialization
        self.input_dims = input_dims
        self.pf_hidden_dims = pf_hidden_dims
        self.g_h_dims = g_h_dims
        self.pf_num = pf_num
        self.inpaint_len = inpaint_len
        self.z_dims = 256
        self.train_mse = train_mse
        self.teacher_forcing = teacher_forcing
        # vae model freeze
        self.vae_model = vae_model
        if self.vae_model is not None:
            for param in self.vae_model.parameters():
                param.requires_grad = False
        # input
        self.past_x = None
        self.future_x = None
        self.c_x = None
        self.init_gc = None
        self.inpaint_x = None
        # teacher forcing hyperparameters
        self.iteration = 0
        self.eps = 0.5 #1.0
        self.decay = torch.FloatTensor([decay])
        self.xavier_initialization()
        
    def pf_encoder(self, past_x, future_x):
        _, h_past_x = self.past_gru(past_x)
        _,h_future_x = self.future_gru(future_x)
        c_x = torch.cat((h_past_x, h_future_x), 2)
        return c_x
    
    def gen_decoder(self, c_x, init_gc):
        y = init_gc.unsqueeze(1)
        ys = []
        ms = []
        hxx = c_x
        h0 = c_x
        hx = [None, None]
        hx[0] = h0
        # try using one layer
        # try to use a vanilla teacher forcing
        
        p = torch.rand(1).item()
        if self.training and (p < self.eps or not self.teacher_forcing):
            gf_input = torch.cat((y, self.inpaint_x[:,:-1,:]),1)
#             print("gf_input", gf_input.size())
            gf_out,_ = self.generation_gru(gf_input, hxx)
#             print("gf_output",gf_out.size())
            zs = self.linear_out(gf_out)
            dummy = torch.zeros((zs.size(0), 24)).cuda()
#             print("zs",zs.size())
            if not self.train_mse:
                for i in range(self.inpaint_len):
                    m,s = self.vae_model.decoder(z = zs[:,i,:], score_tensor = dummy, train = False)
                    ms.append(m)
            for i in range(self.inpaint_len):
                ys.append(zs[:,i,:])     
        else:
            for i in range(self.inpaint_len):
                y, hxx = self.generation_gru(y, hxx)
                y = y.contiguous().view(y.size(0), -1)
                y = self.linear_out(y)
                dummy = torch.zeros((y.size(0), 24)).cuda()
                ys.append(y)
                if not self.train_mse:
                    m, s = self.vae_model.decoder(z = y, score_tensor = dummy, train = False)
                    s = s.view(s.size(0),-1)
                    ms.append(m)
                    z_dist = self.vae_model.encoder(s)
                    y = z_dist.rsample()
    #             print("m:",m.size())
    #                 self.eps = self.decay / (self.decay + torch.exp(self.iteration / self.decay))
                y = y.unsqueeze(1)
        if self.train_mse:
            return torch.stack(ys, 1)
        else:
            return torch.stack(ms, 1)
        return torch.stack(ys, 1)                


    def forward(self, past_x, future_x, inpaint_x):
#         print("inpainting Net",self.training)
        if self.training:
            self.iteration += 1
        past_x = self.get_z_seq(past_x)
        future_x = self.get_z_seq(future_x)
        inpaint_x = self.get_z_seq(inpaint_x)
        self.past_x = past_x
        self.future_x = future_x
        self.inpaint_x = inpaint_x
        self.init_gc = past_x[:,-1,:]
        self.c_x = self.pf_encoder(past_x, future_x)
        # print("c_x size:",self.c_x.size())
        gen_x = self.gen_decoder(self.c_x, self.init_gc)
        # print("gen_x size:", gen_x.size())
        return gen_x, self.iteration
    
    def xavier_initialization(self):
        for name, param in self.past_gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.future_gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.generation_gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.linear_out.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
    def get_z_seq(self, x):
        batch_size = x.size(0)
        x = x.view(-1, 24)
        z_dist = self.vae_model.encoder(x)
        z = z_dist.rsample()
        z = z.view(batch_size, -1, self.z_dims);
        return z
        


        