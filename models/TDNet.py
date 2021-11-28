import math

import torch
from torch import nn
import torch.nn.functional as F


class BiGRU(nn.Module):
    def __init__(self, vocs_size, hidden_size, hidden_size_2):
        super(BiGRU, self).__init__()
        self.embed = nn.Embedding(vocs_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size_2, bidirectional=True)

    def forward(self, input):
        embed = self.embed(input)
        out, _ = self.gru(embed)
        return out


class BiLinear(nn.Module):
    def __init__(self, hidden_size_2):
        super(BiLinear, self).__init__()
        self.W = nn.Parameter(torch.Tensor(hidden_size_2 * 2, hidden_size_2 * 2))
        self.reset_parameters()

    def forward(self, input1, input2):
        batch = input2.shape[0]
        out = torch.bmm(input1, self.W.repeat(batch, 1, 1))
        out = torch.bmm(out, input2.transpose(1, 2))
        return out

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)


class TDNet(nn.Module):
    def __init__(self, voc_size_tweet=18000, voc_size_topic=18000, hidden_size=768, hidden_size_2=768):
        """
        :param input_dim:
        :param hidden_size:
        :param out_size:
        :param n_layers:
        :param batch_size:
        """
        super(TDNet, self).__init__()
        self.gru_tweet = BiGRU(voc_size_tweet, hidden_size, hidden_size_2)
        self.gru_topic = BiGRU(voc_size_topic, hidden_size, hidden_size_2)
        self.bl = BiLinear(hidden_size_2)
        self.detect = nn.Linear(4*hidden_size_2, 2)

    def forward(self, tweet, topic, beta=None):

        batch = tweet.shape[0]

        tweet = self.gru_tweet(tweet)
        topic = self.gru_tweet(topic)

        if beta is None:
            beta = torch.ones(topic.shape[1]) / topic.shape[1]

        S = self.bl(topic, tweet)
        R = torch.zeros((batch, topic.shape[2]*2))
        for i in range(batch):
            mean_r = torch.zeros((tweet.shape[-1],))
            for j in range(topic.shape[1]):

                s_i = nn.Softmax()(S[i][j])
                r_i = torch.mm(tweet[i].T, s_i.reshape((-1, 1)))
                r_i = r_i.reshape((-1,))

                if j == topic.shape[1] - 1:
                    continue
                mean_r += r_i * beta[j]
            R[i] = torch.cat([r_i, mean_r])
        out = self.detect(R)
        out = nn.Softmax(out)
        return out, R



from torchsummary import summary
model = TDNet()
summary(model, [(200,), (10,)], device='cpu')
for name, i in model.named_parameters():
    print(name, i)