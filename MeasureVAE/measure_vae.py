import os
from torch import distributions

from MeasureVAE.encoder import *
from MeasureVAE.decoder import *
from utils.model import *


class MeasureVAE(Model):
    def __init__(
            self, 
            num_notes = 130,
            note_embedding_dim=10,
            metadata_embedding_dim=2,
            num_encoder_layers=2,
            encoder_hidden_size=512,
            encoder_dropout_prob=0.5,
            latent_space_dim=256,
            num_decoder_layers=2,
            decoder_hidden_size=512,
            decoder_dropout_prob=0.5,
            has_metadata=False
    ):
        """
        Initializes the Measure VAE class object
        :param dataset: FolkMeasureDataset object
        :param note_embedding_dim: int,
        :param metadata_embedding_dim: int,
        :param num_encoder_layers: int,
        :param encoder_hidden_size: int,
        :param encoder_dropout_prob: float, from 0. to 1.
        :param latent_space_dim: int,
        :param num_decoder_layers: int,
        :param decoder_hidden_size: int,
        :param decoder_dropout_prob: float, from 0. to 1.
        :param has_metadata: bool
        """
        super(MeasureVAE, self).__init__()
        # define constants
        self.num_beats_per_measure = 4
        self.num_ticks_per_measure = 24
        self.num_ticks_per_beat = int(self.num_ticks_per_measure / self.num_beats_per_measure)

        # initialize members
        self.note_embedding_dim = note_embedding_dim
        self.metadata_embedding_dim = metadata_embedding_dim
        self.num_encoder_layers = num_encoder_layers
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_dropout_prob = encoder_dropout_prob
        self.latent_space_dim = latent_space_dim
        self.num_decoder_layers = num_decoder_layers
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_dropout_prob = decoder_dropout_prob
        self.has_metadata = has_metadata
        self.num_notes = num_notes
        print("NUMBER OF NOTES: ", self.num_notes)
        # Encoder
        self.encoder = Encoder(
            note_embedding_dim = self.note_embedding_dim,
            rnn_hidden_size=self.encoder_hidden_size,
            num_layers=self.num_encoder_layers,
            num_notes=self.num_notes,
            dropout=self.encoder_dropout_prob,
            bidirectional=True,
            z_dim=self.latent_space_dim,
            rnn_class=torch.nn.GRU
        )

        # Decoder
        self.decoder = HierarchicalDecoder(
            note_embedding_dim=self.note_embedding_dim,
            num_notes=self.num_notes,
            z_dim=self.latent_space_dim,
            num_layers=self.num_decoder_layers,
            rnn_hidden_size=self.decoder_hidden_size,
            dropout=self.decoder_dropout_prob,
            rnn_class=torch.nn.GRU
        )

        # location to save model
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        self.filepath = os.path.join(cur_dir, 'models/',
                                     self.__repr__())

    def __repr__(self):
        """
        String Representation of class
        :return: string, class representation
        """
        return f'MeasureVAE(' \
               f'{self.encoder.__repr__()},' \
               f'{self.decoder.__repr__()},' \
               f')'

    def forward(
            self,
            measure_score_tensor: Variable,
            train=True
    ):
        """
        Implements the forward pass of the VAE
        :param measure_score_tensor: torch Variable,
                (batch_size, measure_seq_length)
        :param measure_metadata_tensor: torch Variable,
                (batch_size, measure_seq_length, num_metadata)
        :param train: bool,
        :return: torch Variable,
                (batch_size, measure_seq_length, self.num_notes)
        """
        # check input
        seq_len = measure_score_tensor.size(1)
        assert(seq_len == self.num_ticks_per_measure)
        # compute output of encoding layer
        z_dist = self.encoder(measure_score_tensor)

        # sample from distribution
        z_tilde = z_dist.rsample()

        # compute prior
        prior_dist = distributions.Normal(
            loc=torch.zeros_like(z_dist.loc),
            scale=torch.ones_like(z_dist.scale)
        )
        z_prior = prior_dist.sample()

        # compute output of decoding layer
        weights, samples = self.decoder(
            z=z_tilde,
            score_tensor=measure_score_tensor,
            train=train
        )
        return weights, samples, z_dist, prior_dist, z_tilde, z_prior

    def forward_test(self, measure_score_tensor: Variable):
        """
        Implements the forward pass of the VAE
        :param measure_score_tensor: torch Variable,
                (batch_size, num_measures, measure_seq_length)
        :return: torch Variable,
                (batch_size, measure_seq_length, self.num_notes)
        """
        # check input
        batch_size, num_measures, seq_len = measure_score_tensor.size()
        assert(seq_len == self.num_ticks_per_measure)

        # compute output of encoding layer
        z = []
        for i in range(num_measures):
            z_dist = self.encoder(measure_score_tensor[:, i, :])
            z.append(z_dist.rsample().unsqueeze(1))
        z_tilde = torch.cat(z, 1)

        # compute output of decoding layer
        weights = []
        samples = []
        dummy_measure_tensor = to_cuda_variable(torch.zeros(batch_size, seq_len))
        for i in range(num_measures):
            w, s = self.decoder(
                z=z_tilde[:, i, :],
                score_tensor=dummy_measure_tensor,
                train=False
            )
            samples.append(s)
            weights.append(w.unsqueeze(1))
        samples = torch.cat(samples, 2)
        weights = torch.cat(weights, 1)
        return weights, samples
