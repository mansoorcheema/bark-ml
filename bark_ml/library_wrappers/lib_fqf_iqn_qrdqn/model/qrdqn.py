from torch import nn

from .base_model import BaseModel
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.network import DQNBase, NoisyLinear


class QRDQN(BaseModel):

    def __init__(self, num_channels, num_actions, N, params,
                 dueling_net=False, noisy_net=False):
        super(QRDQN, self).__init__()

        self.N = N
        self.num_channels = num_channels
        self.num_actions = num_actions
        self.dueling_net = dueling_net
        self.noisy_net = noisy_net
        self.embedding_dim = params["ML"]["QRDQN"]["EmbeddingDims", "", 512]

        linear = NoisyLinear if noisy_net else nn.Linear

        # Feature extractor of DQN.
        self.dqn_net = DQNBase(num_channels=num_channels, embedding_dim= self.embedding_dim,
                               hidden=params["ML"]["QRDQN"]["HiddenDims", "", 512])
        # Quantile network.
        if not dueling_net:
            self.q_net = nn.Sequential(
                linear(self.embedding_dim, 512),
                nn.ReLU(),
                linear(512, num_actions * N),
            )
        else:
            self.advantage_net = nn.Sequential(
                linear(self.embedding_dim, 512),
                nn.ReLU(),
                linear(512, num_actions * N),
            )
            self.baseline_net = nn.Sequential(
                linear(self.embedding_dim, 512),
                nn.ReLU(),
                linear(512, N),
            )

        

    def forward(self, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None
        batch_size = states.shape[0] if states is not None\
            else state_embeddings.shape[0]

        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)

        if not self.dueling_net:
            quantiles = self.q_net(
                state_embeddings).view(batch_size, self.N, self.num_actions)
        else:
            advantages = self.advantage_net(
                state_embeddings).view(batch_size, self.N, self.num_actions)
            baselines = self.baseline_net(
                state_embeddings).view(batch_size, self.N, 1)
            quantiles = baselines + advantages\
                - advantages.mean(dim=2, keepdim=True)

        assert quantiles.shape == (batch_size, self.N, self.num_actions)

        return quantiles

    def calculate_q(self, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None
        batch_size = states.shape[0] if states is not None\
            else state_embeddings.shape[0]

        # Calculate quantiles.
        quantiles = self(states=states, state_embeddings=state_embeddings)

        # Calculate expectations of value distributions.
        q = quantiles.mean(dim=1)
        assert q.shape == (batch_size, self.num_actions)

        return q
