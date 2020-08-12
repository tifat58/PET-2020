import model_stealer
import attack1_membership_Inference_cifar10
import attack_1_membership_inference_mnist_fmnist
import attack_2_model_inversion

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


print("Choose the attack type:")
print("1 - membership inference")
print("2 - model inversion")
print("3 - model stealing")

try:
    attack_type =(int)(input(">"))
except ValueError:
    print("Not a Choice")
    exit(1)

print("Choose the data:")
print("1 - mnist")
print("2 - fashion-mnist")
print("3 - cifar10")

try:
    dataset = (int)(input(">"))
except ValueError:
    print("Not a Choice")
    exit(1)

if (attack_type == 1):
    if (dataset == 1):
        attack_1_membership_inference_mnist_fmnist.attack1_main_mnist(shadow_num_epoch=40, target_num_epoch=40, attack_num_epoch=40, batch_size=64)
    if (dataset == 2):
        attack_1_membership_inference_mnist_fmnist.attack1_main_fmnist(shadow_num_epoch=40, target_num_epoch=40, attack_num_epoch=40, batch_size=64)
    if (dataset == 3):
        attack1_membership_Inference_cifar10.attack1_main_cifar10(shadow_num_epoch=40, target_num_epoch=40, attack_num_epoch=40, batch_size=64)

if (attack_type == 2):
    """
    change the values of the hyperparameters during calling the function
    class_label is the expected class [0-9] of which the x value to be optimized
    """
    if (dataset == 1):
        attack_2_model_inversion.main_mnist(train_num_epoch=40, alpha=2000, beta=50, gamma=0.001, lamda=0.001, class_label=6)
    if (dataset == 2):
        attack_2_model_inversion.main_fmnist(train_num_epoch=40, alpha=2000, beta=50, gamma=0.001, lamda=0.001, class_label=2)
    if (dataset == 3):
        attack_2_model_inversion.main_cifar10(train_num_epoch=40, alpha=2000, beta=50, gamma=0.001, lamda=0.001, class_label=4)

if (attack_type == 3):
    if (dataset == 1):
        model_stealer.main( dataset_name="mnist", target_num_epoch=50, attack_num_epoch=40, batch_size=128)
    if (dataset == 2):
        model_stealer.main(dataset_name="fashion-mnist", target_num_epoch=50, attack_num_epoch=40, batch_size=128)
    if (dataset == 3):
        model_stealer.main(dataset_name="cifar10", target_num_epoch=50, attack_num_epoch=40, batch_size=128)

if dataset == 1:
    dataset_name = 'mnist'
elif dataset == 2:
    dataset_name = 'fashion_mnist'
elif dataset == 3:
    dataset_name = 'cifar10'
else:
    dataset_name = ''

print("Results are for Attack {}, dataset {}".format(attack_type, dataset_name))

