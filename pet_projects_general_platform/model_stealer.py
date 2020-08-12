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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#input_size = 784
# batch_size = 128

class Net(nn.Module):
    def __init__(self, dataset_name):
        super(Net, self).__init__()
        # model configuration depends on used dataset
        if (dataset_name == "cifar10"):
            self.conv1 = nn.Conv2d(3, 32, (5, 5), padding=2)
            self.conv2 = nn.Conv2d(32, 32, (5, 5))
            self.fc1 = nn.Linear(32 * 6 * 6, 128)
            self.fc2 = nn.Linear(128, 10)
        else:
            self.conv1 = nn.Conv2d(1, 6, (5, 5), padding=2)
            self.conv2 = nn.Conv2d(6, 16, (5, 5))
            self.fc1 = nn.Linear(16 * 5 * 5, 128)
            self.fc2 = nn.Linear(128, 10)

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


class CustomFakeDataset(Dataset):
    """
    This dataset generates random images and labels them with a class label queried from a source model (in this case the target model)
    """
    def __init__(self, source_model, size=1000, image_size=(3, 224, 224), num_classes=10,
                 transform=None, target_transform=None, random_offset=0, target_model_input_size=28*28):
        self.size = size
        self.num_classes = num_classes
        self.image_size = image_size
        self.random_offset = random_offset
        self.source_model = source_model
        self.transform = transform
        self.target_transform = target_transform
        self.target_model_input_size = target_model_input_size

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
            target = self.target_transform(self.source_model, img, self.target_model_input_size)

        return img, target

    def __len__(self):
        return self.size


def train_model(train_model, train_loader, epochs, input_size):
    """
    :param train_model: the model to be trained
    :param train_loader: the loader accessing the training dataset
    :param epochs: number of epochs during training
    :return:
    """
    # set model to training mode
    train_model.train()

    # define optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(train_model.parameters(), lr=0.001, momentum=0.9)

    # train for the specified number of epochs
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data, target = Variable(data), Variable(target)
            #data = data.view(-1, input_size)
            data = data.to(device)
            target = target.to(device)
            train_model_out = train_model(data)
            loss = criterion(train_model_out, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 200 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.data.item()))
    return train_model

def evaluate_model(eval_model, eval_loader, input_size):
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
            images = images.to(device)
            labels = labels.to(device)
            #images = images.view(-1, input_size)
            outputs = eval_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            for pred in predicted:
                predictions[pred.item()] = predictions[pred.item()] +1
            correct += (predicted == labels).sum().item()
    print('Accuracy: %d %%' % (100 * correct / total))

predictions = [0,0,0,0,0,0,0,0,0,0]


def query_model(model, image, input_size):
    """
    Query the given model with an image
    :param model: the model to be queried
    :param image: the image to query the model with
    :return: the predicted class
    """
    model.to(device)
    image = Variable(image)
    # add a dimenstion to emulate batch size signifier when querying the source model later
    image = image[None, :, :, :]
    image = image.to(device)
    #image = image.view(-1, input_size)
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs.data, 1)
    #print(outputs.data)
    #print(predicted)
    #exit(1)
    predicted = predicted[0]
    predictions[predicted] = predictions[predicted] +1
    # return the class label
    return predicted

def target_transform (model, image, input_size):
    """
    Transform function used in the random dataset to generate class labels by querying the target model
    :param model: the model to be queried to create the class label
    :param image: the image to create a class label for
    :return:
    """
    return query_model(model, image, input_size)

# Main flow ------------------------------------------------------------------------------------------------------------
def main(dataset_name, target_num_epoch=20, attack_num_epoch=20, batch_size=64):

    if dataset_name not in ["mnist", "fashion-mnist", "cifar10"]:
        print("unknown dataset given: ", dataset_name)
        return

    target_epoch = target_num_epoch
    attack_epoch = attack_num_epoch


    # Initialize transform function for data sets
    transform = torchvision.transforms.Compose([
           torchvision.transforms.ToTensor(),
           torchvision.transforms.Normalize(
                 (0.5,), (0.5,))
         ])

    # Define constants used to store / load datasets and model weights
    DATASET_PATH = './data'

    # build the training and evaluation dataset for the target model
    if dataset_name == "mnist":
        train_dataset = torchvision.datasets.MNIST(root = './data', train = True, transform = transform, download=True)
        verify_dataset = torchvision.datasets.MNIST(root = './data', train = False, transform = transform, download=True)
        # mnist dataset contains of 28 * 28 pixel images
        TARGET_MODEL_PATH = './model_stealing_target_model_mnist.pth'
        ATTACK_MODEL_PATH = './model_stealing_attack_model_mnist.pth'
        model_input_size = 28 * 28
        model_input_dimensions = (28, 28)
        target_model = Net(dataset_name).to(device)

    elif dataset_name == "fashion-mnist":
        train_dataset = torchvision.datasets.FashionMNIST(root = './data', train = True, transform = transform, download=True)
        verify_dataset = torchvision.datasets.FashionMNIST(root = './data', train = False, transform = transform, download=True)
        TARGET_MODEL_PATH = './model_stealing_target_model_fmnist.pth'
        ATTACK_MODEL_PATH = './model_stealing_attack_model_fmnist.pth'
        # fashion-mnist dataset contains of 28 * 28 pixel images
        model_input_size = 28 * 28
        model_input_dimensions = (28, 28)
        target_model = Net(dataset_name).to(device)


    # elif dataset_name == "cifar10":
    else:
        train_dataset = torchvision.datasets.CIFAR10(root = './data', train = True, transform = transform, download=True)
        verify_dataset = torchvision.datasets.CIFAR10(root = './data', train = False, transform = transform, download=True)
        TARGET_MODEL_PATH = './model_stealing_target_model_cifar10.pth'
        ATTACK_MODEL_PATH = './model_stealing_attack_model_cifar10.pth'
        # cifar10 dataset contains of 32 * 32 pixel images
        model_input_size = 32 * 32 * 3
        model_input_dimensions = (3, 32, 32)
        target_model = Net(dataset_name).to(device)

    # Initialize the loaders for both datasets
    target_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    target_verify_loader = torch.utils.data.DataLoader(verify_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the target model
    #target_model = Net(model_input_size)


    print("########################################\n")
    print("Selected Attack: Model Stealing (Attack 3)")
    print("Selected Dataset: ", dataset_name)
    print("Train epoch for target model: ", target_epoch)
    print("Train epoch for Attack model: ", attack_epoch)
    print("Batch Size: ", batch_size)
    print("\n########################################\n")

    print("\nTraining Target Model....")
    # Train the target model (uncomment to enable training instead of loading pretrained model data)
    target_model = train_model(target_model, target_train_loader, target_epoch, model_input_size)
    torch.save(target_model.state_dict(), TARGET_MODEL_PATH)
    print("Training Model Validation Accuracy: ")
    evaluate_model(target_model, target_verify_loader, model_input_size)
    # Load the pretrained target model (comment out to enable training instead of loading pretrained model data)
    # target_model.load_state_dict(torch.load(TARGET_MODEL_PATH))


    # build the training dataset and loader for the attack model
    attack_train_dataset = CustomFakeDataset(size=60000, image_size = model_input_dimensions, source_model=target_model, transform = transform, target_transform = target_transform, target_model_input_size=model_input_size)
    attack_train_loader = torch.utils.data.DataLoader(attack_train_dataset, batch_size=batch_size, shuffle=True)
    attack_verify_loader = torch.utils.data.DataLoader(attack_train_dataset, batch_size=batch_size, shuffle=True)

    lbl_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for img, target in attack_train_loader:
        # print(target)
        for tg in target:
            lbl_list[tg] = lbl_list[tg] + 1

    print("Random images prediction class distribution by target Model: ")
    print(lbl_list)
    # exit()
    # Initialize the attack model
    attack_model = Net(model_input_size).to(device)
    print("Training Attack Model.... ")
    # Train the attack model (uncomment to enable training instead of loading pretrained model data)
    attack_model = train_model(attack_model, attack_train_loader, attack_epoch, model_input_size)
    torch.save(attack_model.state_dict(), ATTACK_MODEL_PATH)

    # Load the pretrained attack model (comment out to enable training instead of loading pretrained model data)
    #attack_model.load_state_dict(torch.load(ATTACK_MODEL_PATH))

    # Evaluate model accuracy
    print("\n############## Results ########\n")

    print("Dataset: ", dataset_name)
    print("Random images prediction class distribution by target Model: ", lbl_list)
    print("Target Model Test Accuracy: ")
    evaluate_model(target_model, target_verify_loader, model_input_size)
    print("Attack Model Test Accuracy: ")
    evaluate_model(attack_model, target_verify_loader, model_input_size)
    # print(predictions)
    # Calculate the average distance between the weights of the hidden layers of both models
    layer_size = target_model.fc2.weight.data.size(0)
    distance = target_model.fc2.weight.data - attack_model.fc2.weight.data

    print("Average distance between weights in layer fc2: ", abs(torch.sum(distance).item() / layer_size))


# called by main.py
#main("mnist")
#main("fashion-mnist")
# main("cifar10")
