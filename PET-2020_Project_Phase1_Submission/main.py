import model_stealer

print("Choose the attack type:")
print("1 - membership inference")
print("2 - model inversion")
print("3 - model stealing")

try:
    attack_type =(int)(input(">"))
except ValueError:
    print("Not a number")
    exit(1)

print("Choose the data:")
print("1 - mnist")
print("2 - fashion-mnist")
print("3 - cifar10")

try:
    dataset = (int)(input(">"))
except ValueError:
    print("Not a number")
    exit(1)

if (attack_type == 1):
    if (dataset == 1):
        print("TODO: integrate membership inference on MNIST dataset")
    if (dataset == 2):
        print("TODO: integrate membership inference on FASHION-MNIST dataset")
    if (dataset == 3):
        print("TODO: integrate membership inference on CIFAR10 dataset")

if (attack_type == 2):
    if (dataset == 1):
        print("TODO: integrate model inversion on MNIST dataset")
    if (dataset == 2):
        print("TODO: integrate model inversion on FASHION-MNIST dataset")
    if (dataset == 3):
        print("TODO: integrate model inversion on CIFAR10 dataset")

if (attack_type == 3):
    if (dataset == 1):
        model_stealer.main("mnist")
    if (dataset == 2):
        model_stealer.main("fashion-mnist")
    if (dataset == 3):
        model_stealer.main("cifar10")




