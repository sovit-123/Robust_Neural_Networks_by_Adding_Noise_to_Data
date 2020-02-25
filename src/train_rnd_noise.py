# file to train neural network with random noisy image data
# argument parsers for epochs, train noise, test noise
'''
python src/train_rnd_noise.py --epochs=20 --train_noise=no --test_noise=no
'''

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import matplotlib
plt.style.use('ggplot')

from torchvision import datasets
from torchvision.utils import save_image
from skimage.util import random_noise

ap = argparse.ArgumentParser()
ap.add_argument('-e', '--epochs', type=int, default=10,
	help='number of epochs to train the model')
ap.add_argument('-n_tr', '--train_noise', default='no', type=str,
    help='whether to add noise to training data or not')  
ap.add_argument('-n_te', '--test_noise', default='no', type=str,
    help='whether to add noise to training data or not')  
args = vars(ap.parse_args())

# get computation device
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
device = get_device()

# define transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

# parameters
batch_size = 32
num_classes = 10
pretrained = False
requires_grad = True
train_noise = args['train_noise']
test_noise = args['test_noise']
epochs = args['epochs']

# get the data
trainset = datasets.CIFAR10(
    root='D:\Data_Science\Datasets\PyTorch_Datasets',
    train=True,
    download=True, 
    transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=batch_size,
    shuffle=True
)
testset = datasets.CIFAR10(
    root='D:\Data_Science\Datasets\PyTorch_Datasets',
    train=False,
    download=True,
    transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, 
    batch_size=batch_size,
    shuffle=False
)

def model():
    # all pytorch models expect image to be 224x224
    model = models.resnet18(progress=True, pretrained=pretrained)
    # freeze hidden layers
    if requires_grad == False:
        for param in model.parameters():
            param.requires_grad = False
    # train the hidden layers
    elif requires_grad == True:
        for param in model.parameters():
            param.requires_grad = True
    # make the classification layers learnable
    model.fc = nn.Linear(512, num_classes)
    
    model = model.to(device)
    
    return model
model = model()

# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)     

# the training function
print(f"Training with noise: {train_noise}")
print(f"Validating with noise: {test_noise}")
print(f"Training for {epochs} epochs")
print(f"pretrained = {pretrained}, requires_grad = {requires_grad}")
if pretrained and requires_grad:
    print('Training with ImageNet weights and updating hidden layer weights')
elif pretrained and not requires_grad:
    print('Training with ImageNet weights and freezing hidden layers weights')
elif not pretrained and requires_grad:
    print('Training with random weights and updating hidden layers weights')
elif not pretrained and not requires_grad:
    print('Training with random weights and freezing hidden layers weights')
def train(NUM_EPOCHS, epoch, model, dataloader):
    model.train()
    loss = 0
    acc = 0
    running_loss = 0.0
    running_correct = 0
    for i, data in enumerate(trainloader):
        img, labels = data[0].to(device), data[1].to(device)
        # add noise to the image data
        if train_noise == 'yes':
            noise = torch.randn(img.shape).to(device)
            new_img = img + noise
        elif train_noise == 'no':
            new_img = img
        new_img = new_img.to(device)
        optimizer.zero_grad()
        outputs = model(new_img)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs.data, 1)
        running_loss += loss.item()
        running_correct += (preds == labels).sum().item()
        # backpropagation
        loss.backward()
        # update the parameters
        optimizer.step()
               
    loss = running_loss / len(trainset)
    acc = 100. * running_correct / len(trainset)
    print(f"Epoch {epoch+1} of {NUM_EPOCHS}, train loss: {loss:.3f}, train acc: {acc:.3f}")
    return loss, acc

# the validation function
def validate(NUM_EPOCHS, epoch, model, testloader):
    model.eval()
    loss = 0.0
    acc = 0
    running_loss = 0.0
    running_correct = 0
    with torch.no_grad():
        for i, data in enumerate(testloader):
            img, labels = data[0].to(device), data[1].to(device)
            # add noise to the image data
            if test_noise == 'yes':
                noise = torch.randn(img.shape).to(device)
                new_img = img + noise
            elif test_noise == 'no':
                new_img = img
            new_img = new_img.to(device)
            outputs = model(new_img)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs.data, 1)
            running_loss += loss.item()
            running_correct += (preds == labels).sum().item()

    loss = running_loss / len(testset)
    acc = 100. * running_correct / len(testset)
    print(f"Epoch {epoch+1} of {NUM_EPOCHS}, val loss: {loss:.3f}, val acc: {acc:.3f}")
    return loss, acc   

train_loss, train_acc = [], []
val_loss, val_acc = [], []
start = time.time()
for epoch in range(epochs):
    e_start = time.time()
    train_epoch_loss, train_epoch_acc = train(epochs, epoch, model, trainloader)
    train_loss.append(train_epoch_loss)
    train_acc.append(train_epoch_acc)
    val_epoch_loss, val_epoch_acc = validate(epochs, epoch, model, testloader)
    val_loss.append(val_epoch_loss)
    val_acc.append(val_epoch_acc)
    e_end = time.time()
    print(f"Took {(e_end-e_start)/60:.3f} minutes for epoch {epoch+1}")
end = time.time()
print(f"Took {(end-start)/60:.3f} minutes to train")
torch.save(model, f"outputs/models/rnd_{train_noise}_{test_noise}.pth")
plt.figure()
plt.plot(train_acc, label='training accuracy')
plt.plot(val_acc, label='validation accuracy')
plt.title('Accuracy Plots')
plt.xlabel('Accuracy')
plt.ylabel('Epochs')
plt.legend()
plt.savefig(f"outputs/plots/rnd_{train_noise}_{test_noise}_acc.png")
plt.figure()
plt.plot(train_loss, label='training loss')
plt.plot(val_loss, label='validation loss')
plt.title('Loss Plots')
plt.xlabel('Loss')
plt.ylabel('Epochs')
plt.legend()
plt.savefig(f"outputs/plots/rnd_{train_noise}_{test_noise}_loss.png")