import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import FakeData

input_size = 784
hidden_sizes = [128]
output_size = 10
epochs = 5
batch_size = 4

attack_train_set_size = 1000

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=0)

class CustomFakeDataset(Dataset):

    def __init__(self, source_model, size=1000, image_size=(3, 224, 224), num_classes=10,
                 transform=None, target_transform=None, random_offset=0):
        self.size = size
        self.num_classes = num_classes
        self.image_size = image_size
        self.random_offset = random_offset
        self.source_model = source_model
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # create random image that is consistent with the index id
        if index >= len(self):
            raise IndexError("{} index out of range".format(self.__class__.__name__))
        rng_state = torch.get_rng_state()
        torch.manual_seed(index + self.random_offset)
        img = torch.randn(*self.image_size)
        target = torch.randint(0, self.num_classes, size=(1,), dtype=torch.long)[0]
        torch.set_rng_state(rng_state)

        # convert to PIL Image
        img = transforms.ToPILImage()(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(self.source_model, img)

        return img, target

    def __len__(self):
        return self.size

def train_model(train_model, train_loader, epochs):
    train_model.train()
    optimizer = optim.SGD(train_model.parameters(), lr=0.01, momentum=0.5)
    #optim.Adadelta(train_model.parameters(), lr=1.0)
    # create a loss function
    criterion = nn.NLLLoss()

    # run the main training loop
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data, target = Variable(data), Variable(target)
            # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
            data = data.view(-1, input_size)
            train_model_out = train_model(data)
            loss = criterion(train_model_out, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 5000 == 0:
                print(loss.data)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.data.item()))
    return train_model

def evaluate_model(eval_model, eval_loader):

    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    eval_model.eval()

    with torch.no_grad():
        for data in eval_loader:
            images, labels = data
            images = Variable(images)
            images = images.view(-1, input_size)
            outputs = eval_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    print('Accuracy: %d %%' % (100 * correct / total))

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (i, 100 * class_correct[i] / class_total[i]))


def query_model(model, image):
    image = Variable(image)
    image = image.view(-1, input_size)
    with torch.no_grad():
        outputs = model(image)
        print(outputs)
    _, predicted = torch.max(outputs.data, 1)
    print("Predicted: %s" % predicted)
    return predicted

transform = torchvision.transforms.Compose([
       torchvision.transforms.ToTensor(),
       torchvision.transforms.Normalize(
             (0.5,), (0.5,))
     ])

DATASET_PATH = './data'
TARGET_MODEL_PATH = './model_stealing_target_model.pth'

train_dataset = torchvision.datasets.MNIST(root = './data', train = True, transform = transform, download=True)
verify_dataset = torchvision.datasets.MNIST(root = './data', train = False, transform = transform, download=True)

#train_dataset, verify_dataset = torch.utils.data.random_split(complete_dataset, [50000, 10000])
#print(len(train_dataset))


target_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
target_verify_loader = torch.utils.data.DataLoader(verify_dataset, batch_size=batch_size, shuffle=True)


target_model = Net()

#target_model = train_model(target_model, target_train_loader, 4)
#torch.save(target_model.state_dict(), TARGET_MODEL_PATH)


target_model.load_state_dict(torch.load(TARGET_MODEL_PATH))
#print(target_model.fc1.weight.data)

def target_transform (model, image):
    return query_model(model, image)

attack_train_dataset = CustomFakeDataset(size=1000, image_size = (28, 28), source_model=target_model, transform = transform, target_transform = target_transform)
attack_train_loader = torch.utils.data.DataLoader(attack_train_dataset, batch_size=batch_size, shuffle=True)


evaluate_model(target_model, target_verify_loader)

exit(1)


for batch_idx, (data, target) in enumerate(target_train_loader):
    data, target = Variable(data), Variable(target)
    #print(data)
    #print(data.shape)
    #print(target)
    break



#for batch_idx, (data, target) in enumerate(target_train_loader):
    #data, target = Variable(data), Variable(target)
    #query_model(target_model, data)
    #print(data)
    #print(data.shape)
    #print(target)





exit(1)


for batch_idx, (data, target) in enumerate(target_train_loader):
    #query_model(target_model, data)
    break
    #print(data.shape)
    #print(target)

exit(1)
attack_images = []
attack_labels = []

for i in range(attack_train_set_size):
    attack_images.append(np.random.random((28,28)))


print(target_model)


