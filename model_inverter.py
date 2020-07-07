import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# input size for target model (MNIST)
input_size = 784
# sizes of hidden layers
hidden_sizes = [128, 64]
# one output for every MNIST class
output_size = 10

# trains a MNIST model
def train_model(model, train_loader):
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()

    epochs = 2
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)

            # Training pass
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)

            #This is where the model learns by backpropagating
            loss.backward()

            #optimizes its weights here
            optimizer.step()

            running_loss += loss.item()
        else:
            print("Epoch {} - Training loss: {}".format(e, running_loss/len(train_loader)))

    return model

# verify training impact
def verify_model(model, verify_loader):
    correct_count, all_count = 0, 0
    for images,labels in verify_loader:
        for i in range(len(labels)):
            img = images[i].view(1, 784)
            with torch.no_grad():
                logps = model(img)


            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if(true_label == pred_label):
              correct_count += 1
            all_count += 1

    print("Number Of Images Tested =", all_count)
    print("\nModel Accuracy =", (correct_count/all_count))


# query the model with an image (query_tensor) to obtain a list of confidence values
# also returns the image tensor again (experimental, can probably be removed)
def query_model(model, verify_loader, query_tensor):
    print("Querying model")

    #plt.imshow(image.numpy().squeeze(), cmap='gray_r')
    #plt.show()
    image_vector = query_tensor.view(1, 784)
    image_vector.retain_grad()
    confidence_intervals_raw = model(image_vector)
    confidence_intervals_exp = torch.exp(confidence_intervals_raw)
    confidence_intervals_list = list(confidence_intervals_exp.tolist()[0])
    return (image_vector, confidence_intervals_list)
    #predicted_label = confidence_intervals_list.index(max(confidence_intervals_list))
    #print(predicted_label)
    #return predicted_label



# returns probability that image at index $x_index has label $label
def f(model, verify_loader, label, x_tensor):
    result_tensor, confidence_intervals_list = query_model(model, verify_loader, x_tensor)
    #print("return of query")
    #print(result_tensor)
    #print(confidence_intervals_list)
    return (result_tensor, confidence_intervals_list[label])

# cost function for gradient descent
def c(model, verify_loader, label, x_tensor):
    result_tensor, confidence = f(model, verify_loader, label, x_tensor)
    #print("return of f")
    #print(result_tensor)
    #print(confidence)
    return (result_tensor, torch.tensor([1 - confidence], requires_grad=True))

# placeholder function. Can later be used to process / sharpen image between descent steps
def process(value):
    return value

# placeholder function. Should return the gradient of the given function
def gradient(value):
    return 1

# implement MI-FACE algorithm from the paper
def mi_face(label, alpha, beta, gamma, step_size, model, verify_loader):

    #x_current = torch.zeros(input_size, requires_grad=True)

    # start out with a zeroed vector
    x_current = torch.autograd.Variable(torch.zeros(input_size), requires_grad=True)

    for i in range(1, alpha):
        # save x from the previous step
        x_prev = x_current

        # query model for previous step costs
        x_prev, x_previous_cost = c(model, verify_loader, label, x_prev)

        # the actual descent step
        x_current = process(x_prev - step_size * gradient(x_previous_cost))

        # query model for cost of the new x
        x_current, x_current_cost = c(model, verify_loader, label, x_current)


        # TODO: save previous costs in list $previous_values
        # x_previous_cost = x_current_cost
        # previous_values[previous_index] = x_current_cost
        # previous_index = (previous_index + 1) % beta

        # TODO: additional end conditions
        #if (x_current_cost >= max(previous_values)):
        #    break
        #if (x_current_cost <= gamma):
        #    break

    return x_current

transform = torchvision.transforms.Compose([
       torchvision.transforms.ToTensor(),
       torchvision.transforms.Normalize(
             (0.5,), (0.5,))
     ])

DATASET_PATH = './data'
TARGET_MODEL_PATH = './model_inversion_target_model.pth'

# download datasets if neccessary
train_dataset = torchvision.datasets.MNIST(root = './data', train = True, transform = transform, download=True)
verify_dataset = torchvision.datasets.MNIST(root = './data', train = False, transform = transform, download=True)

# init dataloaders
target_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
target_verify_loader = torch.utils.data.DataLoader(verify_dataset, batch_size=1, shuffle=True)

# init target model
target_model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))

# Disable the next two lines to skip training after saving the model once
target_model = train_model(target_model, target_train_loader)
torch.save(target_model.state_dict(), TARGET_MODEL_PATH)

target_model.load_state_dict(torch.load(TARGET_MODEL_PATH))

print('Finished Training')

# TODO: make this work
result_tensor = mi_face(5, 10, 5, 2, 3, target_model, target_verify_loader)

exit(1)
