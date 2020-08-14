# SketchNet
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from .attention_layer import CombineLayer, PositionalEncoding

class SketchNet(nn.Module):
    def __init__(
        self, zp_dims, zr_dims, pf_dims, gen_dims, combine_dims, pf_num, combine_num, combine_head, inpaint_len, total_len, vae_model = None, teacher_forcing = True):
        super(SketchNet, self).__init__()
        # set stage
        self.stage = "inpainting"
        # Past RNN
        self.past_p_gru = nn.GRU(
            zp_dims, pf_dims, pf_num, 
            batch_first = True, bidirectional = True, dropout = 0.5
        )
        self.past_r_gru = nn.GRU(
            zr_dims, pf_dims, pf_num, 
            batch_first = True, bidirectional = True, dropout = 0.5
        )
        # Future RNN
        self.future_p_gru = nn.GRU(
            zp_dims, pf_dims, pf_num, 
            batch_first = True, bidirectional = True, dropout = 0.5
        )
        self.future_r_gru = nn.GRU(
            zr_dims, pf_dims, pf_num, 
            batch_first = True, bidirectional = True, dropout = 0.5
        )
        # generation RNN
        self.gen_p_gru = nn.GRU(
            zp_dims, gen_dims, pf_num, 
            batch_first = True, bidirectional = True, dropout = 0.5
        )
        self.gen_r_gru = nn.GRU(
            zr_dims, gen_dims, pf_num, 
            batch_first = True, bidirectional = True, dropout = 0.5
        )
        self.gen_p_out = nn.Linear(gen_dims * 2, zp_dims)
        self.gen_r_out = nn.Linear(gen_dims * 2, zr_dims)
        # combine attention
        self.combine_in = nn.Linear(zp_dims + zr_dims, combine_dims)
        self.combine_posenc = PositionalEncoding(combine_dims, n_position=total_len)
        self.combine_dropout = nn.Dropout(p = 0.1)
        self.combine_nn = nn.ModuleList(
            [
                CombineLayer(
                    combine_dims, combine_dims * 4, combine_head, 
                    combine_dims // combine_head, combine_dims // combine_head,
                    dropout = 0.1
                ) for _ in range(combine_num) 
            ]
        )
        self.combine_norm = nn.LayerNorm(combine_dims, eps = 1e-6) 
        self.combine_out = nn.Linear(combine_dims, zp_dims + zr_dims)
        # parameter initialization
        self.zr_dims = zr_dims
        self.zp_dims = zp_dims
        self.pf_dims = pf_dims
        self.gen_dims = gen_dims
        self.pf_num = pf_num
        self.combine_num = combine_num
        self.combine_dims = combine_dims
        self.combine_head = combine_head
        self.total_len = total_len
        self.inpaint_len = inpaint_len
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
        self.xavier_initialization()
        
    def pf_pitch_encoder(self, past_pz, future_pz):
        _, h_past_px = self.past_p_gru(past_pz)
        _, h_future_px = self.future_p_gru(future_pz)
        c_x = torch.cat((h_past_px, h_future_px), 2)
        return c_x
    def pf_rhythm_encoder(self, past_rz, future_rz):
        _, h_past_rx = self.past_r_gru(past_rz)
        _, h_future_rx = self.future_r_gru(future_rz)
        c_x = torch.cat((h_past_rx, h_future_rx), 2)
        return c_x
    def gen_pitch_decoder(self, c_x, init_gc, is_teacher = True):
        y = init_gc.unsqueeze(1)
        ys = []
        hxx = c_x
        if self.training and (is_teacher or not self.teacher_forcing):
            self.use_teacher = True
            gf_input = torch.cat((y, self.inpaint_x[:,:-1,:self.zp_dims]),1)
            gf_out,_ = self.gen_p_gru(gf_input, hxx)
            zs = self.gen_p_out(gf_out)
            for i in range(self.inpaint_len):
                ys.append(zs[:,i,:])     
        else:
            self.use_teacher = False
            for i in range(self.inpaint_len):
                y, hxx = self.gen_p_gru(y, hxx)
                y = y.contiguous().view(y.size(0), -1)
                y = self.gen_p_out(y)
                ys.append(y)
                y = y.unsqueeze(1)
        return torch.stack(ys, 1)                
    def gen_rhythm_decoder(self, c_x, init_gc, is_teacher = True):
        y = init_gc.unsqueeze(1)
        ys = []
        hxx = c_x
        if self.training and (is_teacher or not self.teacher_forcing):
            self.use_teacher = True
            gf_input = torch.cat((y, self.inpaint_x[:,:-1,self.zp_dims:]),1)
            gf_out,_ = self.gen_r_gru(gf_input, hxx)
            zs = self.gen_r_out(gf_out)
            for i in range(self.inpaint_len):
                ys.append(zs[:,i,:])     
        else:
            self.use_teacher = False
            for i in range(self.inpaint_len):
                y, hxx = self.gen_r_gru(y, hxx)
                y = y.contiguous().view(y.size(0), -1)
                y = self.gen_r_out(y)
                ys.append(y)
                y = y.unsqueeze(1)
        return torch.stack(ys, 1)
    
    def combine_decoder(self, past_x, inpaint_x, future_x, c_x, is_train = True):
        inpaint_sta = past_x.size(1)
        p = torch.rand(1).item()
        self.use_teacher = p < self.eps 
        zs = torch.cat((past_x, c_x, future_x), 1)
        if is_train and self.training and self.use_teacher:
            for i in range(self.inpaint_len):
                p = torch.rand(1).item()
                if p < 0.3:
                    zs[:, i + inpaint_sta, :self.zp_dims] = inpaint_x[:, i, :self.zp_dims]
                p = torch.rand(1).item()
                if p < 0.3:
                    zs[:, i + inpaint_sta, self.zp_dims:] = inpaint_x[:, i, self.zp_dims:]
        ys = self.combine_dropout(self.combine_posenc(self.combine_in(zs)))
        weights = []
        for enc_layer in self.combine_nn:
            ys, weight = enc_layer(ys, slf_attn_mask = None)
            weights += [weight]
        ys = self.combine_norm(ys)
        ys = self.combine_out(ys)
        return ys[:, inpaint_sta:inpaint_sta + self.inpaint_len,:], weights

    def sketch_generation(self, past_x, future_x, inpaint_x, sketch_index, sketch_cond):
        past_x = self.get_z_seq(past_x)
        future_x = self.get_z_seq(future_x)
        inpaint_x = self.get_z_seq(inpaint_x)
        self.past_x = past_x
        self.future_x = future_x
        self.inpaint_x = inpaint_x
        init_p_gc = past_x[:, -1, :self.zp_dims]
        init_r_gc = past_x[:, -1, self.zp_dims:]
        is_teacher = False
        c_p_x = self.pf_pitch_encoder(past_x[:,:,:self.zp_dims], future_x[:,:,:self.zp_dims])
        c_r_x = self.pf_rhythm_encoder(past_x[:,:,self.zp_dims:], future_x[:,:,self.zp_dims:]) 
        gen_pz = self.gen_pitch_decoder(c_p_x, init_p_gc, is_teacher)
        gen_rz = self.gen_rhythm_decoder(c_r_x, init_r_gc, is_teacher)
        c_x = torch.cat((gen_pz, gen_rz), -1)
        
        inpaint_sta = past_x.size(1)
        zs = torch.cat((past_x, c_x, future_x), 1)
        cond_output = []
        for i,d in enumerate(sketch_index):
            if d < self.inpaint_len:
                p_dis = self.vae_model.pitch_encoder(sketch_cond[i][0], sketch_cond[i][1])
                zp = p_dis.rsample()
                zs[:, d + inpaint_sta, :self.zp_dims] = zp
                cond_output.append(zp)
            if d >= self.inpaint_len:
                r_dis = self.vae_model.rhythm_encoder(sketch_cond[i])
                zr = r_dis.rsample()
                zs[:, d - self.inpaint_len + inpaint_sta, self.zp_dims:] = zr
                cond_output.append(zr)

        ys = self.combine_dropout(self.combine_posenc(self.combine_in(zs)))
        weights = []
        for enc_layer in self.combine_nn:
            ys, weight = enc_layer(ys, slf_attn_mask = None)
            weights += [weight]
        ys = self.combine_norm(ys)
        ys = self.combine_out(ys)

        ys = ys[:, inpaint_sta:inpaint_sta + self.inpaint_len,:]
        for i,d in enumerate(sketch_index):
            if d < self.inpaint_len:
                ys[:, d, :self.zp_dims] = cond_output[i]
            if d >= self.inpaint_len:
                ys[:, d - self.inpaint_len, self.zp_dims:] = cond_output[i]
        gen_m = self.get_measure(ys)
        return gen_m
    
    def forward(self, past_x, future_x, inpaint_x):
        # order: px, rx, len_x, nrx, gd 
        if self.training:
            self.iteration += 1
        past_x = self.get_z_seq(past_x)
        future_x = self.get_z_seq(future_x)
        inpaint_x = self.get_z_seq(inpaint_x)
        self.past_x = past_x
        self.future_x = future_x
        self.inpaint_x = inpaint_x
        if self.stage == "inpainting":
            init_p_gc = past_x[:, -1, :self.zp_dims]
            init_r_gc = past_x[:, -1, self.zp_dims:]
            p = torch.rand(1).item()
            is_teacher = p < self.eps
            c_p_x = self.pf_pitch_encoder(past_x[:,:,:self.zp_dims], future_x[:,:,:self.zp_dims])
            c_r_x = self.pf_rhythm_encoder(past_x[:,:,self.zp_dims:], future_x[:,:,self.zp_dims:]) 
            gen_pz = self.gen_pitch_decoder(c_p_x, init_p_gc, is_teacher)
            gen_rz = self.gen_rhythm_decoder(c_r_x, init_r_gc, is_teacher)
            gen_z = torch.cat((gen_pz, gen_rz), -1)
            gen_m = self.get_measure(gen_z)
        if self.stage == "sketch":
            init_p_gc = past_x[:, -1, :self.zp_dims]
            init_r_gc = past_x[:, -1, self.zp_dims:]
            is_teacher = False
            c_p_x = self.pf_pitch_encoder(past_x[:,:,:self.zp_dims], future_x[:,:,:self.zp_dims])
            c_r_x = self.pf_rhythm_encoder(past_x[:,:,self.zp_dims:], future_x[:,:,self.zp_dims:]) 
            gen_pz = self.gen_pitch_decoder(c_p_x, init_p_gc, is_teacher)
            gen_rz = self.gen_rhythm_decoder(c_r_x, init_r_gc, is_teacher)
            gen_z = torch.cat((gen_pz, gen_rz), -1)
            final_z, _ = self.combine_decoder(past_x, inpaint_x, future_x, gen_z, is_train = True)
            gen_m = self.get_measure(final_z)
        return gen_m, self.iteration, self.use_teacher, self.stage
    
    def xavier_initialization(self):
        for name, param in self.past_p_gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.past_r_gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.future_p_gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.future_r_gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.gen_p_gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.gen_r_gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.gen_p_out.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.gen_r_out.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
    def get_z_seq(self, x):
        # order: px, rx, len_x, nrx, gd
        px, _ , len_x, nrx, gd = x
        batch_size = px.size(0)
        px = px.view(-1, self.vae_model.seq_len)
        nrx = nrx.view(-1, self.vae_model.seq_len, 3)
        len_x = len_x.view(-1) 
        p_dis = self.vae_model.pitch_encoder(px, len_x)
        r_dis = self.vae_model.rhythm_encoder(nrx)
        zp = p_dis.rsample()
        zr = r_dis.rsample()
        z = torch.cat((zp,zr), -1)
        z = z.view(batch_size, -1, self.zr_dims + self.zp_dims)
        return z
    def get_measure(self, z):
        dummy = torch.zeros((z.size(0), self.vae_model.seq_len)).long().cuda()
        ms = []
        for i in range(self.inpaint_len):
            m = self.vae_model.final_decoder(z[:, i, :], dummy, is_train = False)
            ms.append(m)
        return torch.stack(ms, 1)
    def set_stage(self, stage):
        self.stage = stage
        if self.stage == "sketch":
            for param in self.past_p_gru.parameters():
                param.requires_grad = False
            for param in self.past_r_gru.parameters():
                param.requires_grad = False
            for param in self.future_p_gru.parameters():
                param.requires_grad = False
            for param in self.future_r_gru.parameters():
                param.requires_grad = False
            for param in self.gen_p_gru.parameters():
                param.requires_grad = False
            for param in self.gen_r_gru.parameters():
                param.requires_grad = False
            for param in self.gen_p_out.parameters():
                param.requires_grad = False
            for param in self.gen_r_out.parameters():
                param.requires_grad = False
            
        


        
