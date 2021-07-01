# SketchVAE
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter
from torch.distributions import Normal

class SketchVAE(nn.Module):
    def __init__(
        self, input_dims = 130, p_input_dims = 129, r_input_dims = 3, 
        hidden_dims = 512, zp_dims = 128, zr_dims = 128,
        seq_len = 24, beat_num = 4, tick_num = 6, 
        decay = 1000):
        super(SketchVAE, self).__init__()
        # pitch_encoder
        self.p_vocab_dims = 10
        self.p_layer_num = 2
        self.p_embedding = nn.Embedding(input_dims, self.p_vocab_dims)
        self.p_encoder_gru = nn.GRU(
            self.p_vocab_dims, 
            hidden_dims, 
            num_layers = self.p_layer_num, 
            batch_first = True, 
            bidirectional = True, 
            dropout = 0.2
        )
        self.p_linear_mu = nn.Linear(hidden_dims * 2 * self.p_layer_num, zp_dims)
        self.p_linear_var = nn.Linear(hidden_dims * 2 * self.p_layer_num, zp_dims)
        # rhythm_encoder
        self.r_layer_num = 2
        self.r_encoder_gru = nn.GRU(
            r_input_dims, 
            hidden_dims, 
            num_layers = self.r_layer_num, 
            batch_first = True, 
            bidirectional = True, 
            dropout = 0.2
        )
        self.r_linear_mu = nn.Linear(hidden_dims * 2 * self.r_layer_num, zr_dims)
        self.r_linear_var = nn.Linear(hidden_dims * 2 * self.r_layer_num, zr_dims)
        # hierarchical_decoder
        self.beat_layer_num = 2
        self.tick_layer_num = 2
        self.z_to_beat_hidden = nn.Sequential(
            nn.Linear(zr_dims + zp_dims, hidden_dims * self.beat_layer_num),
            nn.SELU()
        )
        self.beat_0 = Parameter(data = torch.zeros(1))
        self.beat_gru = nn.GRU(
            1, hidden_dims, num_layers = self.beat_layer_num, 
            dropout = 0.2, batch_first = True)
        self.beat_to_tick_hidden = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims * self.tick_layer_num),
            nn.SELU()
        )
        self.beat_to_tick_input = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims),
            nn.SELU()
        )
        self.tick_0 = Parameter(data = torch.zeros(self.p_vocab_dims))
        self.d_embedding = nn.Embedding(input_dims, self.p_vocab_dims)
        self.tick_gru = nn.GRU(
            self.p_vocab_dims + hidden_dims, hidden_dims, num_layers = self.tick_layer_num, 
            dropout = 0.2, batch_first = True)
        self.tick_to_note = nn.Sequential(
            nn.Linear(hidden_dims, input_dims),
            nn.ReLU()
        )
        # parameter initialization
        self.input_dims = input_dims
        self.p_input_dims = p_input_dims
        self.r_input_dims = r_input_dims
        self.zr_dims = zr_dims
        self.zp_dims = zp_dims
        self.hidden_dims = hidden_dims
        self.seq_len = seq_len    
        self.beat_num = beat_num
        self.tick_num = tick_num
        # teacher forcing hyperparameters
        self.iteration = 0
        self.eps = 1.0
        self.decay = torch.FloatTensor([decay])

    def pitch_encoder(self, px, len_x):
        px = self.p_embedding(px)
        s_len_x, s_len_idx = torch.sort(len_x, descending = True)
        _, re_len_idx = torch.sort(s_len_idx)
        s_px = px.index_select(0, s_len_idx)
        padding_px = torch.nn.utils.rnn.pack_padded_sequence(s_px, s_len_x, batch_first = True)
        padding_px = self.p_encoder_gru(padding_px)[-1]
        padding_px = padding_px.transpose(0,1).contiguous()
        padding_px = padding_px.view(padding_px.size(0), -1)
        n_px = padding_px.index_select(0, re_len_idx)
        p_mu = self.p_linear_mu(n_px)
        p_var = self.p_linear_var(n_px).exp_()
        p_dis = Normal(p_mu, p_var)
        return p_dis
    def rhythm_encoder(self, rx):
        rx = self.r_encoder_gru(rx)[-1]
        rx = rx.transpose(0,1).contiguous()
        rx = rx.view(rx.size(0), -1)
        r_mu = self.r_linear_mu(rx)
        r_var = self.r_linear_var(rx).exp_()
        r_dis = Normal(r_mu, r_var)
        return r_dis
    def final_decoder(self, z, gd, is_train = True):
        gd = self.d_embedding(gd)
        beat_out = self.forward_beat(z)
        recon = self.forward_tick(beat_out, gd, is_train)
        return recon
    def forward_beat(self, z):
        batch_size = z.size(0)
        h_beat = self.z_to_beat_hidden(z)
        h_beat = h_beat.view(batch_size, self.beat_layer_num, -1)
        h_beat = h_beat.transpose(0,1).contiguous()
        beat_input = self.beat_0.unsqueeze(0).expand(
            batch_size, self.beat_num, 1
        )
        beat_out, _ = self.beat_gru(beat_input, h_beat)
        return beat_out
    def forward_tick(self, beat_out, gd, is_train = True):
        ys = []
        batch_size = beat_out.size(0)
        tick_input = self.tick_0.unsqueeze(0).expand(batch_size, self.p_vocab_dims)
        tick_input = tick_input.unsqueeze(1)
        y = tick_input
        for i in range(self.beat_num):
            h_tick = self.beat_to_tick_hidden(beat_out[:, i, :])
            h_tick = h_tick.view(batch_size, self.tick_layer_num, -1)
            h_tick = h_tick.transpose(0,1).contiguous()
            c_tick = self.beat_to_tick_input(beat_out[:, i, :]).unsqueeze(1)
            for j in range(self.tick_num):
                y = torch.cat((y, c_tick), -1)
                y, h_tick = self.tick_gru(y, h_tick)
                y = y.contiguous().view(y.size(0), -1)
                y = self.tick_to_note(y)
                ys.append(y)
                y = y.argmax(-1)
                y = self.d_embedding(y)
                if self.training and is_train:
                    p = torch.rand(1).item()
                    if p < self.eps:
                        y = gd[:, i * self.tick_num + j, :]
                    # update the eps after one batch
                    self.eps = self.decay / (self.decay + torch.exp(self.iteration / self.decay))
                y = y.unsqueeze(1)
        return torch.stack(ys, 1)
    def forward(self, px, rx, len_x, gd):
        """
        px: [batch, seq_len, 1] with p_input number range
        rx: [batch, seq_len, r_input]
        len_x: [batch, 1] the efficient length of each pitch sequence
        gd: [batch, seq_len, 1] groundtruth of the melody sequence
        """
        if self.training:
            self.iteration += 1
        p_dis = self.pitch_encoder(px, len_x)
        r_dis = self.rhythm_encoder(rx)
        zp = p_dis.rsample()
        zr = r_dis.rsample()
        z = torch.cat((zp,zr), -1)
        recon = self.final_decoder(z, gd)
        return recon, p_dis, r_dis, self.iteration


