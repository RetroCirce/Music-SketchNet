# inpaintrnn for sketchvae
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
        self.use_teacher = False
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
            self.use_teacher = True
            gf_input = torch.cat((y, self.inpaint_x[:,:-1,:]),1)
#             print("gf_input", gf_input.size())
            gf_out,_ = self.generation_gru(gf_input, hxx)
#             print("gf_output",gf_out.size())
            zs = self.linear_out(gf_out)
            dummy = torch.zeros((zs.size(0), 24)).long().cuda()
#             print("zs",zs.size())
            if not self.train_mse:
                for i in range(self.inpaint_len):
                    m = self.vae_model.final_decoder(zs[:, i, :], dummy ,is_train = False)
                    ms.append(m)
            for i in range(self.inpaint_len):
                ys.append(zs[:,i,:])     
        else:
            self.use_teacher = False
            for i in range(self.inpaint_len):
                y, hxx = self.generation_gru(y, hxx)
                y = y.contiguous().view(y.size(0), -1)
                y = self.linear_out(y)
                dummy = torch.zeros((y.size(0), 24)).long().cuda()
                ys.append(y)
                if not self.train_mse:
                    m = self.vae_model.final_decoder(y, dummy, is_train = False)
                    ms.append(m)
                y = y.unsqueeze(1)
        if self.train_mse:
            return torch.stack(ys, 1), self.inpaint_x
        else:
            return torch.stack(ms, 1)
        return torch.stack(ys, 1)                


    def forward(self, past_x, future_x, inpaint_x):
        # order: px, rx, len_x, nrx, gd 
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
        return gen_x, self.iteration, self.use_teacher
    
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
        # order: px, rx, len_x, nrx, gd
        px, _ , len_x, nrx, gd = x
        batch_size = px.size(0)
        px = px.view(-1, 24)
        nrx = nrx.view(-1, 24, 3)
        len_x = len_x.view(-1) 
        p_dis = self.vae_model.pitch_encoder(px, len_x)
        r_dis = self.vae_model.rhythm_encoder(nrx)
        if self.train_mse:
            zp = p_dis.mean
            zr = r_dis.mean
        else:
            zp = p_dis.rsample()
            zr = r_dis.rsample()
        z = torch.cat((zp,zr), -1)
        z = z.view(batch_size, -1, self.z_dims);
        return z
        


        
