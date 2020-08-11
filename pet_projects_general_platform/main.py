import model_stealer
import attack1_membership_Inference_cifar10
import attack_1_membership_inference_mnist_fmnist
import attack_2_model_inversion


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
        attack_1_membership_inference_mnist_fmnist.attack1_main_mnist()
    if (dataset == 2):
        attack_1_membership_inference_mnist_fmnist.attack1_main_fmnist()
    if (dataset == 3):
        attack1_membership_Inference_cifar10.attack1_main_cifar10()

if (attack_type == 2):
    if (dataset == 1):
        attack_2_model_inversion.main_mnist()
    if (dataset == 2):
        attack_2_model_inversion.main_fmnist()
    if (dataset == 3):
        attack_2_model_inversion.main_cifar10("cifar10")

if (attack_type == 3):
    if (dataset == 1):
        model_stealer.main("mnist")
    if (dataset == 2):
        model_stealer.main("fashion-mnist")
    if (dataset == 3):
        model_stealer.main("cifar10")




