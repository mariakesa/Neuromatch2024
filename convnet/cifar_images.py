import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Define a function to load the CIFAR-10 dataset


def load_cifar10():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=1, shuffle=True, num_workers=2)
    return trainloader

# Define a function to find images from a specific class


def find_images_from_class(loader, target_class, num_images):
    images = []
    for data in loader:
        inputs, labels = data
        if labels.item() == target_class:
            # Squeeze to remove the singleton dimension
            images.append(inputs.squeeze())
            if len(images) == num_images:
                break
    return images


# Load CIFAR-10 dataset
trainloader = load_cifar10()

# Find images for the specified classes
# Let's assume class 0 (airplane) for the first and last image
# and find different classes for the middle three images

class_0_images = find_images_from_class(trainloader, 0, 2)  # Airplane
class_1_image = find_images_from_class(trainloader, 1, 1)   # Automobile
class_2_image = find_images_from_class(trainloader, 2, 1)   # Bird
class_3_image = find_images_from_class(trainloader, 3, 1)   # Cat

# Prepare the sequence of images
images = [class_0_images[0], class_1_image[0],
          class_2_image[0], class_3_image[0], class_0_images[1]]

# Convert images to numpy format for plotting
images_np = [img.numpy().transpose((1, 2, 0)) for img in images]

# Plot the images
plt.figure(figsize=(15, 3))
for i, img in enumerate(images_np):
    plt.subplot(1, 5, i + 1)
    plt.imshow(img)
    plt.axis('off')  # Remove the axis numbers
plt.show()
