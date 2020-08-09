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
batch_size = 4

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
    """
    This dataset generates random images and labels them with a class label queried from a source model (in this case the target model)
    """
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
            # set the target to the class label predicted by the target model
            target = self.target_transform(self.source_model, img)

        return img, target

    def __len__(self):
        return self.size

def train_model(train_model, train_loader, epochs):
    """

    :param train_model: the model to be trained
    :param train_loader: the loader accessing the training dataset
    :param epochs: number of epochs during training
    :return:
    """
    # set model to training mode
    train_model.train()

    # define optimizer and loss function
    optimizer = optim.SGD(train_model.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.NLLLoss()

    # train for the specified number of epochs
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data, target = Variable(data), Variable(target)
            data = data.view(-1, input_size)
            train_model_out = train_model(data)
            loss = criterion(train_model_out, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 5000 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.data.item()))
    return train_model

def evaluate_model(eval_model, eval_loader):
    """

    :param eval_model: model to be evaluated
    :param eval_loader: loader accessing the evaluation data set
    :return:
    """
    # initialize evaluation metrics
    correct = 0
    total = 0

    # set model to evaluation mode
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
    print('Accuracy: %d %%' % (100 * correct / total))


def query_model(model, image):
    """
    Query the given model with an image
    :param model: the model to be queried
    :param image: the image to query the model with
    :return: the predicted class
    """
    image = Variable(image)
    image = image.view(-1, input_size)
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs.data, 1)
    predicted = predicted[0]
    # return the class label
    return predicted

def target_transform (model, image):
    """
    Transform function used in the random dataset to generate class labels by querying the target model
    :param model: the model to be queried to create the class label
    :param image: the image to create a class label for
    :return:
    """
    return query_model(model, image)

# Main flow ------------------------------------------------------------------------------------------------------------

# Initialize transform function for data sets
transform = torchvision.transforms.Compose([
       torchvision.transforms.ToTensor(),
       torchvision.transforms.Normalize(
             (0.5,), (0.5,))
     ])

# Define constants used to store / load datasets and model weights
DATASET_PATH = './data'
TARGET_MODEL_PATH = './model_stealing_target_model.pth'
ATTACK_MODEL_PATH = './model_stealing_attack_model.pth'

# build the training and evaluation dataset for the target model
train_dataset = torchvision.datasets.MNIST(root = './data', train = True, transform = transform, download=True)
verify_dataset = torchvision.datasets.MNIST(root = './data', train = False, transform = transform, download=True)

# Initialize the loaders for both datasets
target_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
target_verify_loader = torch.utils.data.DataLoader(verify_dataset, batch_size=batch_size, shuffle=True)

# Initialize the target model
target_model = Net()

# Train the target model (uncomment to enable training instead of loading pretrained model data)
#target_model = train_model(target_model, target_train_loader, 4)
#torch.save(target_model.state_dict(), TARGET_MODEL_PATH)

# Load the pretrained target model (comment out to enable training instead of loading pretrained model data)
target_model.load_state_dict(torch.load(TARGET_MODEL_PATH))


# build the training dataset and loader for the attack model
attack_train_dataset = CustomFakeDataset(size=10000, image_size = (28, 28), source_model=target_model, transform = transform, target_transform = target_transform)
attack_train_loader = torch.utils.data.DataLoader(attack_train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the attack model
attack_model = Net()

# Train the attack model (uncomment to enable training instead of loading pretrained model data)
#attack_model = train_model(attack_model, attack_train_loader, 40)
#torch.save(attack_model.state_dict(), ATTACK_MODEL_PATH)

# Load the pretrained attack model (comment out to enable training instead of loading pretrained model data)
attack_model.load_state_dict(torch.load(ATTACK_MODEL_PATH))

# Evaluate model accuracy
#evaluate_model(target_model, target_verify_loader)
#evaluate_model(attack_model, target_verify_loader)

# Calculate the average distance between the weights of the hidden layers of both models
layer_size = target_model.fc2.weight.data.size(0)
distance = target_model.fc2.weight.data - attack_model.fc2.weight.data

print("Average distance between weights in layer fc2: ", abs(torch.sum(distance).item() / layer_size))
