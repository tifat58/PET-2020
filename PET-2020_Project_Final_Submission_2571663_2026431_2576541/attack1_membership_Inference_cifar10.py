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
from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
plt.ion()   # interactive mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(device)

def get_target_label_idx(labels, targets, shots=5, test=False):
    """
    Get the indices of labels that are included in targets.
    :param labels: array of labels
    :param targets: list/tuple of target labels
    :return: list with indices of target labels
    """
    final_list = []
    
    for t in targets:
        if test:
            final_list += np.argwhere(np.isin(labels, t)).flatten().tolist()
        else:
            final_list += np.argwhere(np.isin(labels, t)).flatten().tolist()
    
    return final_list


def convert_label(x):

    if x >= 5:
        return x-5
    else:
        return x


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class CIFAR10_Dataset(Dataset):

    def __init__(self, root: str, normal_class=[0,1,2,3,4,5,6,7,8,9], shots=5):
        super().__init__()

        self.root = root
        self.normal_classes = tuple(normal_class)

        target_transform = transforms.Lambda(lambda x: convert_label(x))
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        train_set = MyCIFAR10(root=self.root, train=True, download=True,
                            transform=transform)      
        train_index = get_target_label_idx(train_set.targets, self.normal_classes, shots=shots)
        random.shuffle(train_index) #shuffling before split

        # splitting to target and shadowset
        train_index_half_len = int(len(train_index)/2)
        shadow_set = Subset(train_set, train_index[0:train_index_half_len])
        target_set = Subset(train_set, train_index[train_index_half_len:])

        shadow_half_len = int(len(shadow_set)/2)
        # print("shadow half len: ", shadow_half_len)
        self.shadow_train = Subset(shadow_set, list(range(0, shadow_half_len)))
        self.shadow_test = Subset (shadow_set, list(range(shadow_half_len, len(shadow_set))))

        target_half_len = int(len(target_set)/2)
        # print("Target half len: ", target_half_len)
        self.target_train = Subset(target_set, list(range(0, target_half_len)))
        self.target_unknown = Subset(target_set, list(range(target_half_len, len(target_set))))

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        # test_set = MyMNIST(root=self.root, train=False, download=True,
        #                         transform=transform)


class MyCIFAR10(CIFAR10):
    """Torchvision CIFAR10 class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, **kwargs):
        super(MyCIFAR10, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """Override the original method of the CIFAR10 class"""
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.test_data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # print(type(img))
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index  # only line changed

		
""" This is the classifier model as mentioned in the paper
"""
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, (5,5), padding=2)
        self.conv2 = nn.Conv2d(32, 32, (5,5))
        self.fc1   = nn.Linear(32*6*6, 128)
        self.fc2   = nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = x.view(-1, self.num_flat_features(x))
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        # print(size)
        num_features = 1
        for s in size:
            num_features *= s
        # print(num_features)
        return num_features


""" Attack Model Implementation """
class AttackModel(nn.Module):
  def __init__(self, input_size, hidden_size):
      super(AttackModel, self).__init__()
      self.input_size = input_size
      self.hidden_size  = hidden_size

      self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
      self.fc2 = torch.nn.Linear(self.hidden_size, 2)
      # self.sigmoid = torch.nn.Softmax()

  def forward(self, x):
      x = self.fc1(x)
      x = F.softmax(self.fc2(x))
      output = x
      return output
	  
	  
def train_model(model, train_loader, test_loader, train_size, test_size, criterion, optimizer, scheduler, num_epochs=20, attack=False):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            
            if phase == 'train':
                model.train()  # Set model to training mode
                
                running_loss = 0.0
                running_corrects = 0
                
                for data in train_loader:
                    if attack:
                      inputs, labels = data
                    else:
                      inputs, labels, idx = data

                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(True):
                        outputs = model(inputs)
                        
                        _, preds = torch.max(outputs, 1)
                        # print(preds, labels)
                        loss = criterion(outputs, labels)
                        
                        loss.backward()
                        optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                    
                scheduler.step()
                # print(running_corrects)
                epoch_loss = running_loss / train_size
                epoch_acc = running_corrects.double() / train_size

                print('{} Loss: {:.4f} Acc: {:.4f}'.format('Train', epoch_loss, epoch_acc))
                    
            else:
                
                model.eval()   # Set model to evaluate mode
                
                running_loss = 0.0
                running_corrects = 0
                
                for data in test_loader:
                    if attack:
                      inputs, labels = data
                    else:
                      inputs, labels, idx = data
                    
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
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

                last_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc, last_model_wts



""" Returns top3 prediction probability given x"""
def test(model, test_loader, test_size, criterion):
    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0
    # prediction_list = []
    i = 0
    for data in test_loader:
        inputs, labels, idx = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            sm = torch.nn.Softmax()
            pred_probs = sm(outputs)
            pred_probs, indices = torch.sort(pred_probs, descending=True)
            # print(pred_probs)
            if i == 0:
                prediction_list = pred_probs[:,0:3]
            else:
                prediction_list = torch.cat((prediction_list, pred_probs[:,0:3]))
            i += 1
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / test_size
    epoch_acc = running_corrects.double() / test_size

    print('{} Loss: {:.4f} Acc: {:.4f}'.format('Val', epoch_loss, epoch_acc))

    return prediction_list


""" Attack model test function """
def attack_test(model, test_loader, test_size, criterion):
    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0
    # prediction_list = []
    i = 0
    test_true_label = []
    test_pred_label = []
    for data in test_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            sm = torch.nn.Softmax()
            pred_probs = sm(outputs)
            pred_probs, indices = torch.sort(pred_probs, descending=True)
            # # print(pred_probs)
            # if i == 0:
            #   prediction_list = pred_probs[:,0:3]
            # else:
            #   prediction_list = torch.cat((prediction_list, pred_probs[:,0:3]))
            # i += 1
            loss = criterion(outputs, labels)

            test_true_label.append(labels.data)
            test_pred_label.append(preds.data)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / test_size
    epoch_acc = running_corrects.double() / test_size

    print('{} Loss: {:.4f} Acc: {:.4f}'.format('Val', epoch_loss, epoch_acc))

    return test_true_label, test_pred_label

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# # get some random training images
# dataiter = iter(shadow_train_loader)
# images, labels, idx = dataiter.next()
#
# # show images
# imshow(torchvision.utils.make_grid(images[0:8]))
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(8)))

def attack1_main_cifar10(shadow_num_epoch=40, target_num_epoch=40, attack_num_epoch=40, batch_size=64):
    shadow_epoch = shadow_num_epoch
    target_epoch = target_num_epoch
    attack_epoch = attack_num_epoch
    dataset = CIFAR10_Dataset(root='data/')
    shadow_train_loader = DataLoader(dataset.shadow_train, batch_size=batch_size, shuffle=True, num_workers=0)
    shadow_test_loader = DataLoader(dataset.shadow_test, batch_size=batch_size, shuffle=True, num_workers=0)
    train_size = len(dataset.shadow_train)
    test_size = len(dataset.shadow_test)

    print("########################################\n")
    print("Selected Attack: Membership Inference (Attack 1)")
    print("Selected Dataset: Cifar10")
    print("Train epoch for Target model: ", target_epoch)
    print("Train epoch for Shadow model: ", shadow_epoch)
    print("Train epoch for Attack model: ", attack_epoch)
    print("Batch Size: ", batch_size)
    print("\n########################################\n")

    net = LeNet().to(device)
    # loading pretrained model
    #     net = torch.load('LeNet_5_class_pretrained_model.tar')
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-07)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)

    print("\nTraining Shadow Model.........\n")
    best_net, best_acc, last_net = train_model(net, shadow_train_loader, shadow_test_loader,
                                               train_size, test_size, criterion, optimizer_ft,
                                               exp_lr_scheduler, num_epochs=shadow_epoch)

    s_net = LeNet().to(device)
    s_net.load_state_dict(last_net)

    print("Shadow model Test set predictions: ")
    shadow_test_prediction = test(s_net, shadow_test_loader, test_size, criterion)
    print("Shadow model Train set predictions: ")
    shadow_train_prediction = test(s_net, shadow_train_loader, train_size, criterion)

    d1 = torch.utils.data.TensorDataset(shadow_train_prediction, torch.ones(train_size, dtype=torch.long))
    d2 = torch.utils.data.TensorDataset(shadow_test_prediction, torch.zeros(test_size, dtype=torch.long))
    shadow_trained_dataset = torch.utils.data.ConcatDataset([d1, d2])


    """ Training Target model """
    target_train_loader = DataLoader(dataset.target_train, batch_size=batch_size, shuffle=True, num_workers=0)
    target_unk_loader = DataLoader(dataset.target_unknown, batch_size=batch_size, shuffle=True, num_workers=0)

    target_train_size = len(dataset.target_train)
    target_test_size = len(dataset.target_unknown)
    target_net = LeNet().to(device)
    # loading pretrained model
    #     net = torch.load('LeNet_5_class_pretrained_model.tar')
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(target_net.parameters(), lr=0.001, weight_decay=1e-07)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

    print("\nTraining Target Model.........\n")
    target_best_net, target_best_acc, target_last_net = train_model(target_net, target_train_loader, target_unk_loader,
                                                                    target_train_size, target_test_size, criterion,
                                                                    optimizer_ft, exp_lr_scheduler, num_epochs=target_epoch)

    t_net = LeNet().to(device)
    t_net.load_state_dict(target_last_net)

    print("Target model unknown set predictions: ")
    target_test_prediction = test(t_net, target_unk_loader, target_test_size, criterion)
    print("Target model Train set predictions: ")
    target_train_prediction = test(t_net, target_train_loader, target_train_size, criterion)

    d1 = torch.utils.data.TensorDataset(target_train_prediction, torch.ones(train_size, dtype=torch.long))
    d2 = torch.utils.data.TensorDataset(target_test_prediction, torch.zeros(test_size, dtype=torch.long))
    target_trained_dataset = torch.utils.data.ConcatDataset([d1, d2])

    s_trained_train_size = len(shadow_trained_dataset)
    s_trained_test_size = len(target_trained_dataset)

    attack_model = AttackModel(3, 64).to(device)
    atk_train_loader = DataLoader(shadow_trained_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    atk_test_loader = DataLoader(target_trained_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer_ft = optim.Adam(attack_model.parameters(), lr=0.01, weight_decay=1e-6)

    print("\nTraining Attack Model.........\n")
    attack_best_net, attack_best_acc, attack_last_net = train_model(attack_model, atk_train_loader, atk_test_loader,
                                                                    s_trained_train_size, s_trained_test_size, criterion,
                                                                    optimizer_ft, exp_lr_scheduler, num_epochs=attack_epoch, attack=True)

    a_net = AttackModel(3, 64).to(device)
    a_net.load_state_dict(attack_last_net)
    t_l, t_p = attack_test(attack_best_net, atk_test_loader, s_trained_test_size, criterion)

    l_true = []
    l_pred = []
    for d in t_l:
      l_list = d.cpu().numpy()
      for l in l_list:
          l_true.append(l)
    # print(l_true)
    for d in t_p:
      l_list = d.cpu().numpy()
      for l in l_list:
          l_pred.append(l)
    # print(l_pred)

    target_names = ['0','1']
    print(classification_report(l_true, l_pred))
    cm1 = confusion_matrix(l_true, l_pred, labels=[1, 0])
    print("Confusion Matrix:")
    print(cm1)

    print('\nStatistical Report:')
    total1=sum(sum(cm1))
    #####from confusion matrix calculate accuracy
    accuracy1=(cm1[0,0]+cm1[1,1])/total1
    print ('Acc: ', round(accuracy1, 2))

    prec = cm1[0,0]/(cm1[0,0]+cm1[1,0])
    print('Precision: ', round(prec,2) )

    rec = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    print('Recall: ', round(rec,2))

    index = ['Cifar10']
    df = pd.DataFrame({'Precision': prec,'Recall': rec}, index=index)
    ax = df.plot.bar(rot=0)