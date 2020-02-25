'''
Execute by:
python add_noise.py --dataset=cifar10 --gauss_noise=0.05 --salt_pep=0.5 --speckle_noise=0.05
'''

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from skimage.util import random_noise

import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import argparse

ap = argparse.ArgumentParser()
ap.add_argument(
    '-d', '--dataset', type=str, 
    help='dataset to use'
)
ap.add_argument(
    '-gn', '--gauss_noise', type=float, default=0.0, 
    help='amount of gaussian noise to add'
)
ap.add_argument(
    '-svp', '--salt_pep', type=float, default=0.0, 
    help='amount of salt vs pepper noise to add'
)
ap.add_argument(
    '-sn', '--speckle_noise', type=float, default=0.0, 
    help='amount of speckle noise noise to add'
)
args = vars(ap.parse_args())

BATCH_SIZE = 4

if args['dataset'] == 'mnist' or args['dataset'] == 'fashionmnist':  
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)), 
    ])
    if args['dataset'] == 'mnist':
        trainset = datasets.MNIST(
            root='../input/data',
            train=True,
            download=True, 
            transform=transform
        )
        testset = datasets.MNIST(
            root='../input/data',
            train=False,
            download=True,
            transform=transform
        )
    elif args['dataset'] == 'fashionmnist':
        trainset = datasets.FashionMNIST(
            root='../input/data',
            train=True,
            download=True, 
            transform=transform
        )
        testset = datasets.FashionMNIST(
            root='../input/data',
            train=False,
            download=True,
            transform=transform
        )
if args['dataset'] == 'cifar10':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
    ])
    trainset = datasets.CIFAR10(
        root='../input/data',
        train=True,
        download=True, 
        transform=transform
    )
    testset = datasets.CIFAR10(
        root='../input/data',
        train=False,
        download=True,
        transform=transform
    )
trainloader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=BATCH_SIZE,
    shuffle=True
)
testloader = torch.utils.data.DataLoader(
    testset, 
    batch_size=BATCH_SIZE,
    shuffle=False
)

def save_noisy_image(img, name):
    if img.size(1) == 3:
        img = img.view(img.size(0), 3, 32, 32)
        save_image(img, name)
    else:
        img = img.view(img.size(0), 1, 28, 28)
        save_image(img, name)


def gaussian_noise():
    for data in trainloader:
        img, _ = data[0], data[1]
        # make var=0.0 to add 0 noise
        gauss_img = torch.tensor(random_noise(img, mode='gaussian', mean=0, var=args['gauss_noise'], clip=True))
        save_noisy_image(gauss_img, f"outputs/plots/{args['dataset']}_{args['gauss_noise']}_gaussian.png")
        break

def salt_pepper_noise():
    for data in trainloader:
        img, _ = data[0], data[1]
        s_and_p = torch.tensor(random_noise(img, mode='s&p', salt_vs_pepper=args['salt_pep'], clip=True))
        save_noisy_image(s_and_p, f"outputs/plots/{args['dataset']}_{args['salt_pep']}_s&p.png")
        break

def speckle_noise():
    for data in trainloader:
        img, _ = data[0], data[1]
        # make var=0.0 to add 0 noise
        speckle_noise = torch.tensor(random_noise(img, mode='speckle', mean=0, var=args['speckle_noise'], clip=True))
        save_noisy_image(speckle_noise, f"outputs/plots/{args['dataset']}_{args['speckle_noise']}_speckle.png")
        break

gaussian_noise()
salt_pepper_noise()
speckle_noise()

'''
The following are the custom implementations of the adding of noise. Use as per your requirements

def gaussian_noise():
    for data in trainloader:
        img, _ = data[0], data[1]
        # gaussian_random_noise(mean, sigma, (batch_size, row, col, ch))
        gauss_noise = np.random.normal(0, 0.5, img.shape)
        # convert the noise to pytorch tensor
        gauss_noise = torch.tensor(gauss_noise)
        gauss_noise = gauss_noise.reshape(img.shape[0], img.shape[1], img.shape[2], img.shape[3])
        gauss_img = img + gauss_noise
        # clip to make the values fall between 0 and 1
        gauss_img = np.clip(gauss_img, 0., 1.)
        save_noisy_image(gauss_img, f"Images/{args['dataset']}_gaussian.png")
        break


def salt_pepper_noise():
    for data in trainloader:
        img, _ = data[0], data[1]
        s_vs_p = 0.2
        amount = 0.002
        s_and_p = img + (s_vs_p*amount) + torch.randn(img.shape)
        # clip to make the values fall between 0 and 1
        s_and_p = np.clip(s_and_p, 0., 1.)
        save_noisy_image(s_and_p, f"Images/{args['dataset']}_s&p.png")
        break

def speckle_noise():
    for data in trainloader:
        img, _ = data[0], data[1]
        gauss_noise = np.random.normal(0, 0.5, img.size())
        gauss_noise = torch.tensor(gauss_noise)
        gauss_noise = gauss_noise.reshape(img.shape[0], img.shape[1], img.shape[2], img.shape[3])
        speckle_noise = img + (img * gauss_noise)
        # clip to make the values fall between 0 and 1
        speckle_noise = np.clip(speckle_noise, 0., 1.)
        save_noisy_image(speckle_noise, f"Images/{args['dataset']}_speckle.png")
        break
'''