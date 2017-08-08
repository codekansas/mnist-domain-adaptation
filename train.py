#!/usr/bin/env python3

import argparse
import os
from collections import defaultdict
import pkg_resources

pkg_resources.require('torch==0.1.12.post2')
import torch
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

from models import (
    Encoder,
    Predictor,
    weights_init,
)

DESCRIPTION = '''
    Trains the MNIST image classifier with constraints.

    This script can be run with the default parameters, in which case it will
    train a vanilla CNN for classifying whether an MNIST digit is positive or
    negative, as well as an auxiliary classifier that will try to predict the
    actual digit from a latent vector in the classifier.

    The \033[1m--domain-restrict\033[0m argument causes the encoder model to
    adapt itself so that the auxiliary classifier's performance decreases.

    The \033[1m--swap-predictors\033[0m argument swaps the two tasks, so that
    the model classifies MNIST digits and the auxiliary classifier tries to
    predict if the digit is odd or even from the latent representation.
'''


def get_smoothed(x_data: list, smoothing: int=50):
    samples = [x_data[i:i+smoothing] for i in range(len(x_data) - smoothing)]
    lower_bound = np.asarray([np.percentile(x, 25) for x in samples])
    upper_bound = np.asarray([np.percentile(x, 75) for x in samples])
    means = np.asarray([np.mean(x) for x in samples])
    return means, lower_bound, upper_bound


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('-d', '--domain-restrict', action='store_true')
    parser.add_argument('-s', '--swap-predictors', action='store_true')
    parser.add_argument('-b', '--batch-size', type=int, default=100)
    parser.add_argument('-e', '--num-epoch', type=int, default=10)
    parser.add_argument('-r', '--data-root', type=str, default='data/')
    parser.add_argument('-i', '--image-root', type=str, default='images/')
    parser.add_argument('-w', '--num-data-workers', type=int, default=2)
    args = parser.parse_args()

    # Creates objects for loading the the MNIST dataset.
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
        num_workers=args.num_data_workers,
    )

    # Builds the models.
    num_embed_dims = 128
    encoder = Encoder(num_embed_dims).apply(weights_init)
    predictor = Predictor(num_embed_dims, 1).apply(weights_init)
    adaptor = Predictor(num_embed_dims, 10).apply(weights_init)

    # Defines the loss functions.
    adapt_loss_func = nn.CrossEntropyLoss()
    predict_loss_func = nn.BCELoss()

    # Swaps the models, if the user specified as such.
    if args.swap_predictors:
        predictor, adaptor = adaptor, predictor
        predict_loss_func, adapt_loss_func = adapt_loss_func, predict_loss_func

    # Creates model optimizers.
    encoder_optim = optim.Adam(encoder.parameters())
    adaptor_optim = optim.Adam(adaptor.parameters())
    predictor_optim = optim.Adam(predictor.parameters())

    # Keeps track of training progress.
    metrics = defaultdict(list)

    # Trains the model.
    for epoch in range(1, args.num_epoch + 1):
        for i, (x_data, a_data) in enumerate(train_dataloader, 1):
            encoder.zero_grad()
            predictor.zero_grad()
            adaptor.zero_grad()

            # Converts Y data to odd-even labels.
            y_data = (a_data % 2).float()

            # Swaps the predict and adapt data, if needed.
            if args.swap_predictors:
                y_data, a_data = a_data, y_data

            # Converts the input data to autograd variables.
            input_variable = Variable(x_data)
            predict_variable = Variable(y_data)
            adapt_variable = Variable(a_data)

            # Updates the adaptor network.
            encoded = encoder(input_variable)
            adapt_prob = adaptor(encoded)
            adapt_err = adapt_loss_func(adapt_prob, adapt_variable)
            adapt_err.backward(retain_variables=True)  # Retain encoder parts.
            adaptor_optim.step()

            # Weight clipping (Wasserstein GAN intuition).
            for param in adaptor.parameters():
                param.data.clamp_(-0.01, 0.01)

            # Updates the predictor and encoder networks parameters.
            encoder.zero_grad()
            predict_prob = predictor(encoded)
            predict_err = predict_loss_func(predict_prob, predict_variable)
            if args.domain_restrict:
                deadapt_err = adapt_loss_func(adapt_prob, adapt_variable)
                predict_err -= deadapt_err
            predict_err.backward()
            predictor_optim.step()
            encoder_optim.step()

            # Helper function for computing accuracy.
            def torch_accuracy(y_true, y_pred):
                acc = torch.sum(y_true == y_pred)
                acc = acc.data.numpy()[0] / args.batch_size
                return acc

            # Gets the predictions from the probabilities.
            if args.swap_predictors:
                adapt_pred = adapt_prob.round()
                _, predict_pred = predict_prob.max(-1)
            else:
                predict_pred = predict_prob.round()
                _, adapt_pred = adapt_prob.max(-1)

            # Computes accuracy and logs it.
            adapt_acc = torch_accuracy(adapt_pred, adapt_variable)
            predict_acc = torch_accuracy(predict_pred, predict_variable)
            metrics['Adapt Accuracy'].append(adapt_acc)
            metrics['Predict Accuracy'].append(predict_acc)

            print(
                '\rEpoch {epoch} ({iteration})'
                .format(epoch=epoch, iteration=i)
                + ''.join(
                    ' {}: {:.2f}'.format(k, v[-1])
                    for k, v in sorted(metrics.items())
                ),
                end='',
            )

        if not os.path.isdir(args.image_root):
            os.mkdir(args.image_root)
        save_name = 'metrics_epoch_{}.png'.format(epoch)
        save_path = os.path.join(args.image_root, save_name)

        plt.figure()
        for name, value in sorted(metrics.items()):
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
