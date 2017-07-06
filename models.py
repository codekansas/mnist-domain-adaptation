"""Holds all the model classes."""

import pkg_resources

pkg_resources.require('torch==0.1.12.post2')
import torch
from torch import nn
from torch.nn import init


class Encoder(nn.Module):
    """Defines the encoder model.

    This model takes an MNIST digit and encodes it into a `num_embed_dims`
    vector, which is used as input by the `PredictEven` and `PredictDigits`
    models. The encoder model is a simple stacked 2D CNN.
    """

    def __init__(self, num_embed_dims: int):
        super(Encoder, self).__init__()
        self.conv_model = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(3, 3),
            ),
            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True,
            ),
            nn.MaxPool2d(
                kernel_size=(2, 2),
            ),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3),
            ),
            nn.BatchNorm2d(
                num_features=64,
            ),
            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True,
            ),
            nn.Conv2d(
                in_channels=64,
                out_channels=8,
                kernel_size=(1, 1),
                groups=4,
            ),
            nn.BatchNorm2d(
                num_features=8,
            ),
            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True,
            ),
        )
        self.embed_model = nn.Sequential(
            nn.Linear(
                in_features=8 * 11 * 11,
                out_features=num_embed_dims,
            ),
            nn.Softplus(),
        )

    def forward(self, x):
        x = self.conv_model(x)
        x = x.view(-1, 8 * 11 * 11)
        x = self.embed_model(x)
        return x


class Predictor(nn.Module):
    """Defines a model for predicting something from the embedding vector.

    This model takes as input the encoded vector and outputs a prediction
    with `num_classes` classes. In this example, two predictor networks are
    trained to predict separate things, and the encoded vector is
    restricted so that one of the networks works poorly and the other
    works well.
    """

    def __init__(self, num_embed_dims: int, num_classes: int):
        super(Predictor, self).__init__()
        self.pred_model = nn.Sequential(
            nn.Linear(
                in_features=num_embed_dims,
                out_features=num_classes,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.pred_model(x)


def weights_init(m):
    """Function for custom initialization of a model's weights.

    This initialization is *probably* good. A model can be initialized as:
        `encoder = Encoder(...).apply(weights_init)`
    """
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        init.xavier_uniform(m.weight.data)
        m.bias.data.fill_(0)
