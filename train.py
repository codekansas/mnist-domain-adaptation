#!/usr/bin/env python3

import argparse
import os
from collections import defaultdict
import pkg_resources

pkg_resources.require('torch==0.1.12.post2')
import torch
from torch.nn import init
from torch.autograd import Variable
from torch import (
    utils as tutils,
    nn,
    optim,
)

pkg_resources.require('torchvision==0.1.8')
from torchvision import (
    datasets,
    transforms,
    utils as vutils,
)

import matplotlib.pyplot as plt
import numpy as np

DESCRIPTION = '''
    Trains the image classifier with or without the
'''


def get_smoothed(x_data, smoothing=50):
    samples = [x_data[i:i+smoothing] for i in range(len(x_data) - smoothing)]
    lower_bound = np.asarray([np.percentile(x, 25) for x in samples])
    upper_bound = np.asarray([np.percentile(x, 75) for x in samples])
    means = np.asarray([np.mean(x) for x in samples])
    return means, lower_bound, upper_bound


class Encoder(nn.Module):
    def __init__(self, num_embed_dims):
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
    def __init__(self, num_embed_dims):
        super(Predictor, self).__init__()
        self.pred_model = nn.Sequential(
            nn.Linear(
                in_features=num_embed_dims,
                out_features=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.pred_model(x)


class Adaptor(nn.Module):
    def __init__(self, num_embed_dims):
        super(Adaptor, self).__init__()
        self.adapt_model = nn.Sequential(
            nn.Linear(
                in_features=num_embed_dims,
                out_features=10,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.adapt_model(x)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        init.xavier_uniform(m.weight.data)
        m.bias.data.fill_(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('-b', '--batch-size', type=int, default=100)
    parser.add_argument('-r', '--data-root', type=str, default='data/')
    parser.add_argument('-w', '--num-workers', type=int, default=2)
    parser.add_argument('-e', '--num-epoch', type=int, default=10)
    parser.add_argument('-i', '--image-root', type=str, default='images/')
    parser.add_argument('-d', '--domain-adapt', action='store_true')
    args = parser.parse_args()

    # Loads the MNIST dataset.
    train_dataset = datasets.MNIST(
        root=args.data_root,
        download=True,
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5), (0.5, 0.5)),
        ]),
    )
    train_dataloader = tutils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # Builds the models.
    num_embed_dims = 128
    encoder = Encoder(num_embed_dims).apply(weights_init)
    predictor = Predictor(num_embed_dims).apply(weights_init)
    adaptor = Adaptor(num_embed_dims).apply(weights_init)

    # Loss function: Minimize binary crossentropy.
    bce_loss_func = nn.BCELoss()
    xent_loss_func = nn.CrossEntropyLoss()

    # Creates model optimizers.
    encoder_optim = optim.Adam(encoder.parameters())
    adaptor_optim = optim.Adam(adaptor.parameters())
    predictor_optim = optim.Adam(predictor.parameters())

    # Keeps track of training progress.
    metrics = defaultdict(list)

    # Trains the model.
    for epoch in range(1, args.num_epoch + 1):
        for i, (x_data, y_data) in enumerate(train_dataloader, 1):
            encoder.zero_grad()
            predictor.zero_grad()

            # Converts Y data to odd-even labels.
            is_odd = y_data % 2

            # Converts the input data to autograd variables.
            input_variable = Variable(x_data)
            label_variable = Variable(y_data)
            is_odd_variable = Variable(is_odd).float()

            # Updates the predictor network.
            encoded = encoder(input_variable)
            is_odd_prob = predictor(encoded)
            is_odd_err = bce_loss_func(is_odd_prob, is_odd_variable)
            is_odd_err.backward(retain_variables=args.domain_adapt)

            # Computes accuracy.
            is_odd_acc = torch.sum(is_odd_prob.round() == is_odd_variable)
            is_odd_acc = is_odd_acc.data.numpy()[0] / args.batch_size

            # Updates the adaptor network.
            label_prob = adaptor(encoded.detach())
            label_err = xent_loss_func(label_prob, label_variable)
            label_err.backward(retain_variables=args.domain_adapt)

            encoder_optim.step()
            predictor_optim.step()
            adaptor_optim.step()

            # Performs domain adaptation, if the user specified it.
            if args.domain_adapt:
                adapt_err = xent_loss_func(1 - label_prob, label_variable)
                adapt_err.backward()
                encoder_optim.step()

            # Computes accuracy.
            _, label_pred = label_prob.max(-1)
            label_acc = torch.sum(label_pred == label_variable)
            label_acc = label_acc.data.numpy()[0] / args.batch_size

            print(
                '\rEpoch {epoch}:{iteration} '
                'is odd acc={is_odd_acc:.2f} '
                'label acc={label_acc:.2f}    '
                .format(
                    epoch=epoch,
                    iteration=i,
                    is_odd_acc=is_odd_acc,
                    label_acc=label_acc,
                ),
                end='',
            )

            metrics['Is Odd Accuracy'].append(is_odd_acc)
            metrics['Label Accuracy'].append(label_acc)

        if not os.path.isdir(args.image_root):
            os.mkdir(args.image_root)
        save_name = 'metrics_epoch_{}.png'.format(epoch)
        save_path = os.path.join(args.image_root, save_name)

        plt.figure()
        for name, value in metrics.items():
            v, lo, hi = get_smoothed(value)
            plt.plot(v, label=name)
            plt.fill_between(
                np.arange(len(v)),
                lo,
                hi,
                alpha=0.2,
                linewidth=0,
                antialiased=True,
            )
        plt.grid()
        plt.xlabel('Batches')
        plt.xlim([0, len(v)])
        plt.ylim([0, 1])
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode='expand', borderaxespad=0.)
        plt.savefig(save_path)
        plt.close()

    print(dataloader)
