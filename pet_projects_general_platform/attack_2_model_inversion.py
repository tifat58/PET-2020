from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Subset, Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
import random
import torchvision.transforms as transforms

plt.ion()   # interactive mode
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

""" LeNet architecture implementation
    This model is used to train mnist and fashion mnist classifier 
"""
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self,  input_dim=1, output_dim=10):
        super(LeNet, self,).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(self.input_dim, 6, (5,5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, (5,5))
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, self.output_dim)
        
    def forward(self, x):

        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# LeNet for Cifar10
class LeNet_cifar10(nn.Module):
    def __init__(self,  input_dim=1, output_dim=10):
        super(LeNet_cifar10, self,).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(self.input_dim, 6, (5,5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, (5,5))
        self.fc1   = nn.Linear(16*6*6, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, self.output_dim)
        
    def forward(self, x):

        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# function to initialize weights with xavier normal distribution
def weight_inits(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
    
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)


# Custom Model dataset class
# input args: name: dataset name {mnist, fashionmnist, cifar10}, root: directory to download data
# returns: train and test loader
from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
import torchvision.transforms as transforms

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class ModelDataset(Dataset):

    def __init__(self, name: str, root='./data'):
        super().__init__()
        self.root = root
        self.name = name

        if name == 'cifar10':

          self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
          transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
          # target_transform = transforms.Lambda(lambda x: convert_label(x))
          
          self.train_dataset = torchvision.datasets.CIFAR10(root=self.root, train=True, download=True, transform=transform)
          self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=64, shuffle=True, num_workers=2)

          self.test_dataset = torchvision.datasets.CIFAR10(root=self.root, train=False, download=True, transform=transform)
          self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=64, shuffle=False, num_workers=2)

          self.train_size = (len(self.train_dataset))
          self.test_size = (len(self.test_dataset))


        elif name == 'mnist':
          self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
          transform = transforms.Compose([transforms.ToTensor(),
                                        ])
          # target_transform = transforms.Lambda(lambda x: convert_label(x))
          
          self.train_dataset = torchvision.datasets.MNIST(root=self.root, train=True, download=True, transform=transform)
          self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=64, shuffle=True, num_workers=2)

          self.test_dataset = torchvision.datasets.MNIST(root=self.root, train=False, download=True, transform=transform)
          self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=64, shuffle=False, num_workers=2)

          self.train_size = (len(self.train_dataset))
          self.test_size = (len(self.test_dataset))


        elif name == 'FashionMNIST':

          self.classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
          transform = transforms.Compose([transforms.ToTensor(),
                                        ])
          # target_transform = transforms.Lambda(lambda x: convert_label(x))
          
          self.train_dataset = torchvision.datasets.FashionMNIST(root=self.root, train=True, download=True, transform=transform)
          self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=64, shuffle=True, num_workers=2)

          self.test_dataset = torchvision.datasets.FashionMNIST(root=self.root, train=False, download=True, transform=transform)
          self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=64, shuffle=False, num_workers=2)

          self.train_size = (len(self.train_dataset))
          self.test_size = (len(self.test_dataset))



def train_model(model, criterion, optimizer, data_loader, scheduler, train_size, num_epochs=20):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            
            if phase == 'train':
                model.train()  # Set model to training mode
                
                running_loss = 0.0
                running_corrects = 0
                
                for data in data_loader:
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(True):
                        outputs = model(inputs)
                        
                        _, preds = torch.max(outputs, 1)
#                         print(preds, labels)
                        loss = criterion(outputs, labels)
                        
                        loss.backward()
                        optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                scheduler.step()
                
                epoch_loss = running_loss / train_size
                epoch_acc = running_corrects.double() / train_size

                print('{} Loss: {:.4f} Acc: {:.4f}'.format('Train', epoch_loss, epoch_acc))
                    
  

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # # load best model weights
    # model.load_state_dict(best_model_wts)
    # return model, best_acc
    
def test_model(model, criterion, optimizer, data_loader, test_size):
    model.eval()   # Set model to evaluate mode
                
    running_loss = 0.0
    running_corrects = 0

    for data in data_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / test_size
    epoch_acc = running_corrects.double() / test_size

    print('{} Loss: {:.4f} Acc: {:.4f}'.format('Val', epoch_loss, epoch_acc))

# process function is kept as identity function
def process(x):
  return x


#Model Inversion Algorithm 1

# sent x with noisy initialization with input shape
# y is the label to predict
# model: trained classifier model
# alpha: total number of iteration/epoch
# beta: early stopping criteria
# gamma: minimum loss value to stop iteration
# lamda: optimizer learning rate

def MI_face(x, y, model, alpha=1500, beta=50, gamma=0.001, lamda=0.001, optim_name='SGD'):
    model.eval()
    # x = x.to(device)
    y= y.to(device)
    if optim_name == 'Adam':
      optimizer = optim.Adam([x], lr=lamda)
      
    else:
      optimizer = optim.SGD([x], lr=lamda, momentum=0.9)

    loss_list = torch.zeros([alpha])
    for i in range(alpha):
        optimizer.zero_grad()
        logits = model(x)
        # x = process(x)

        prob = torch.softmax(logits, -1)
        loss = y * prob.log()
        loss = - loss.sum(-1).mean()
        
        loss.backward()
        optimizer.step()

        loss_list[i] = loss.item()
        # print(loss.item())
        if i > beta:
          marker = True
          for j in range((i - beta), i):
            if loss_list[j] >= loss.item():
              marker = False
              break
          if marker:
            break
        
        if round(loss.item(),4) <= gamma:
          break

        if i % 100 == 0:
            print('loss', loss.item())
            
    print("Image training finished...")
    print("total iteration:", i)
    
    x = torch.tanh(x)
    return x,y


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_image(images):
    pic = images.detach().numpy()
    # print(pic.shape)
    pic_1 = pic[0,0,:,:]
    plt.imshow(pic_1, cmap='Greys', interpolation='nearest')


def save_model(model, model_save_name):
    path = F"/content/gdrive/My Drive/{model_save_name}"
    torch.save(model.state_dict(), path)
    print("Model Saved...")


#Training Fashionmnist
def main_fmnist(train_num_epoch=20, alpha=1500, beta=50, gamma=0.001, lamda=0.001, class_label=6):

    fashion_data = ModelDataset(name="FashionMNIST")
    train_size = fashion_data.train_size
    test_size = fashion_data.test_size

    print("########################################\n")
    print("Selected Attack: Model Inversion (Attack 2)")
    print("Selected Dataset: ", fashion_data.name)
    print("Selected Class: ", class_label)
    print("Train epoch for target model: ", train_num_epoch)
    print("Hyperparametes for MI_Face function: alpha={}, beta={}, gamma={}, lamda={}".format(alpha, beta, gamma, lamda))
    print("\n########################################\n")

    fashion_net = LeNet(input_dim=1).to(device)
    fashion_net.apply(weight_inits)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(fashion_net.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)

    # get some random training images
    # dataiter = iter(fashion_data.train_loader)
    # images, labels = dataiter.next()
    # imshow(torchvision.utils.make_grid(images[0:8]))
    # print(' '.join('%5s' % fashion_data.classes[labels[j]] for j in range(8)))

    print("Training Target Model: \n")
    train_model(fashion_net, criterion, optimizer_ft, fashion_data.train_loader, exp_lr_scheduler,
                train_size=train_size, num_epochs=train_num_epoch)
    print("Validation loss and accuracy of Classifier Model in Fashion Mnist Data: ")
    test_model(fashion_net, criterion, optimizer_ft,fashion_data.test_loader, test_size)

    y = torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y[class_label] = 1.0  # assigning the expected class label probability to 1
    x = torch.randn((1, 1, 28, 28), requires_grad=True, device=device)
    print("\ncalling MI_face algorithm to optimize x for class label {} \n".format(class_label))
    images, labels = MI_face(x, y, fashion_net, alpha=alpha, beta=beta, gamma=gamma,
                             lamda=lamda)

    # test_net = LeNet()
    # test_net.load_state_dict(torch.load(F"/content/gdrive/My Drive/fashion_net.pt" ))
    # test_model(test_net, criterion, optimizer_ft, fashion_data.test_loader)
    print("\n------ Final Output:-------")
    print("Trained Image True Label: {} ({})".format(fashion_data.classes[torch.argmax(labels).item()], torch.argmax(labels).item()))
    y_pred = fashion_net(images)
    prob = torch.softmax(y_pred, -1)
    print("Trained Image Predicted Label: {} ({}) ".format(fashion_data.classes[prob.argmax().item()], prob.argmax().item()))
    print("Predicted probability for the class: ", prob.max().item())
    pic = images.cpu().detach().numpy()
    pic_1 = pic[0,0,:,:]
    plt.imshow(pic_1, cmap='Greys', interpolation='nearest')
    plt.show()


#Training mnist
def main_mnist(train_num_epoch=20, alpha=1500, beta=50, gamma=0.001, lamda=0.001, class_label=6):

    mnist_data = ModelDataset("mnist")
    train_size = mnist_data.train_size
    test_size = mnist_data.test_size

    print("########################################\n")
    print("Selected Attack: Model Inversion (Attack 2)")
    print("Selected Dataset: ", mnist_data.name)
    print("Selected Class: ", class_label)
    print("Train epoch for target model: ", train_num_epoch)
    print("Hyperparametes for MI_Face function: alpha={}, beta={}, gamma={}, lamda={}".format(alpha, beta, gamma, lamda))
    print("\n########################################\n")

    mnist_net = LeNet(input_dim=1).to(device)
    mnist_net.apply(weight_inits)
    criterion = nn.CrossEntropyLoss()
    optimizer_mnist = optim.SGD(mnist_net.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_mnist, step_size=15, gamma=0.1)

    print("Training Target Model: \n")
    train_model(mnist_net, criterion, optimizer_mnist, mnist_data.train_loader,
                exp_lr_scheduler,train_size=train_size, num_epochs=train_num_epoch)
    print("Validation loss and accuracy of Classifier Model in Mnist Data:")
    test_model(mnist_net, criterion, optimizer_mnist, mnist_data.test_loader,
               test_size=test_size)

    y = torch.FloatTensor([0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0])
    y[class_label] = 1.0 # assigning the expected class label probability to 1
    x = torch.randn((1, 1, 28, 28), requires_grad=True, device=device)
    print("\n calling MI_face algorithm to optimize x for class label {} \n".format(class_label))
    images, labels = MI_face(x, y, mnist_net, alpha=alpha, beta=beta, gamma=gamma, lamda=lamda)

    # test_net = LeNet()
    # test_net.load_state_dict(torch.load(F"/content/gdrive/My Drive/mnist_net.pt" ))
    # test_model(test_net, criterion, optimizer_ft, fashion_data.test_loader)
    print("\n------ Final Output:-------")
    print("Trained Image True Label: {} ({})".format(mnist_data.classes[torch.argmax(labels).item()], torch.argmax(labels).item()))
    y_pred = mnist_net(images)
    prob = torch.softmax(y_pred, -1)
    print("Trained Image Predicted Label: {} ({}) ".format(mnist_data.classes[prob.argmax().item()], prob.argmax().item()))
    print("Predicted probability for the class: ", prob.max().item())
    pic = images.cpu().detach().numpy()
    pic_1 = pic[0,0,:,:]
    plt.imshow(pic_1, cmap='Greys', interpolation='nearest')
    plt.show()


#Training CIFAR10
def main_cifar10(train_num_epoch=20, alpha=1500, beta=50, gamma=0.001, lamda=0.001, class_label=6):

    cifar10_data = ModelDataset(name="cifar10")
    train_size = cifar10_data.train_size
    test_size = cifar10_data.test_size

    print("########################################\n")
    print("Selected Attack: Model Inversion (Attack 2)")
    print("Selected Dataset: ", cifar10_data.name)
    print("Selected Class: ", class_label)
    print("Train epoch for target model: ", train_num_epoch)
    print("Hyperparametes for MI_Face function: alpha={}, beta={}, gamma={}, lamda={}".format(alpha, beta, gamma, lamda))
    print("\n########################################\n")

    cifar_net = LeNet_cifar10(input_dim=3).to(device)
    cifar_net.apply(weight_inits)
    criterion = nn.CrossEntropyLoss()
    optimizer_cifar = optim.SGD(cifar_net.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_cifar, step_size=35, gamma=0.1)

    print("Training Target Model: \n")
    train_model(cifar_net, criterion, optimizer_cifar, cifar10_data.train_loader, exp_lr_scheduler,
                       train_size=train_size, num_epochs=train_num_epoch)
    print("Validation loss and accuracy of Classifier Model in CIFAR10 Data: ")
    test_model(cifar_net, criterion, optimizer_cifar, cifar10_data.test_loader, test_size)

    y = torch.FloatTensor([0.0, 0, 0.0, 0, 0, 0, 0, 0, 0, 0])
    y[class_label] = 1.0  # assigning the expected class label probability to 1
    x = torch.randn((1, 3, 32, 32), requires_grad=True, device=device)
    images, labels = MI_face(x, y, cifar_net, alpha=alpha, beta=beta, gamma=gamma, lamda=lamda)

    # cifar_net = LeNet_cifar10()
    # cifar_net.load_state_dict(torch.load(F"/content/gdrive/My Drive/cifar_net.pt" ))
    # test_model(cifar_net, criterion, optimizer_ft, cifar10_data.test_loader)
    print("\n------ Final Output:-------")
    print("Trained Image True Label: {} ({})".format(cifar10_data.classes[torch.argmax(labels).item()], torch.argmax(labels).item()))
    y_pred = cifar_net(images)
    prob = torch.softmax(y_pred, -1)
    print("Trained Image Predicted Label: {} ({}) ".format(cifar10_data.classes[prob.argmax().item()], prob.argmax().item()))
    print("Predicted probability for the class: ", prob.max().item())
    pic = images.cpu().detach().numpy()
    pic_1 = pic[0,0,:,:]
    plt.imshow(pic_1, cmap='Greys', interpolation='nearest')
    plt.show()