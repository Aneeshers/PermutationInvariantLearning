from torchvision.datasets import CIFAR100, MNIST
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from PIL import Image
import random
import os
import torch
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

class dataUtils:
    mnist_superclasses = ['0 vs 1', '2 vs 3', '4 vs 5', '6 vs 7', '8 vs 9']
    digit_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    cifar_mapping_coarse_fine = {
            'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
            'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
            'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
            'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
            'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear',
                                    'sweet_pepper'],
            'household electrical device': ['clock', 'computer_keyboard', 'lamp',
                                            'telephone', 'television'],
            'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
            'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
            'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
            'large man-made outdoor things': ['bridge', 'castle', 'house', 'road',
                                            'skyscraper'],
            'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain',
                                            'sea'],
            'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee',
                                            'elephant', 'kangaroo'],
            'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
            'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
            'people': ['baby', 'boy', 'girl', 'man', 'woman'],
            'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
            'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
            'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree',
                    'willow_tree'],
            'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
            'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
        }
    fine_labels = [
            'apple',
            'aquarium_fish',
            'baby',
            'bear',
            'beaver',
            'bed',
            'bee',
            'beetle',
            'bicycle',
            'bottle',
            'bowl',
            'boy',
            'bridge',
            'bus',
            'butterfly',
            'camel',
            'can',
            'castle',
            'caterpillar',
            'cattle',
            'chair',
            'chimpanzee',
            'clock',
            'cloud',
            'cockroach',
            'couch',
            'crab',
            'crocodile',
            'cup',
            'dinosaur',
            'dolphin',
            'elephant',
            'flatfish',
            'forest',
            'fox',
            'girl',
            'hamster',
            'house',
            'kangaroo',
            'computer_keyboard',
            'lamp',
            'lawn_mower',
            'leopard',
            'lion',
            'lizard',
            'lobster',
            'man',
            'maple_tree',
            'motorcycle',
            'mountain',
            'mouse',
            'mushroom',
            'oak_tree',
            'orange',
            'orchid',
            'otter',
            'palm_tree',
            'pear',
            'pickup_truck',
            'pine_tree',
            'plain',
            'plate',
            'poppy',
            'porcupine',
            'possum',
            'rabbit',
            'raccoon',
            'ray',
            'road',
            'rocket',
            'rose',
            'sea',
            'seal',
            'shark',
            'shrew',
            'skunk',
            'skyscraper',
            'snail',
            'snake',
            'spider',
            'squirrel',
            'streetcar',
            'sunflower',
            'sweet_pepper',
            'table',
            'tank',
            'telephone',
            'television',
            'tiger',
            'tractor',
            'train',
            'trout',
            'tulip',
            'turtle',
            'wardrobe',
            'whale',
            'willow_tree',
            'wolf',
            'woman',
            'worm',
        ]
    def get_mnist_superclass_dataloader(epoch, batch_size=8, train=True):
        filename = f'./data/mnist_superclass_epoch{epoch}_{"train" if train else "test"}.pt'
        
        if os.path.exists(filename):
            dataset = torch.load(filename)
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            mnist_dataset = datasets.MNIST(root='./data', train=train, download=True, transform=transform)
            # Filter the dataset for the current epoch
            indices = [i for i, (image, label) in enumerate(mnist_dataset) if label in dataUtils.digit_pairs[epoch]]
            dataset = Subset(mnist_dataset, indices)
            # Save the dataset
            torch.save(dataset, filename)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2 if not train else 0)
        return loader
        
    def get_cifar_superclass_dataloader(superclass_index, batch_size=8, train=True):
        filename = f'/home/aneeshmu/WeightedParticles/Gradient-based-Particle-Filter/data/cifar_superclass_{superclass_index}_{"train" if train else "test"}.pt'
        #filename = f'./data/cifar_superclass_{superclass_index}_{"train" if train else "test"}.pt'
        print(superclass_index)
        if os.path.exists(filename):
            # Load dataset if it exists
            dataset = torch.load(filename)
        else:
            # Define the transform
            data_transforms = {
                'train': transforms.Compose([
                    RandomHorizontalFlip(0.5),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'test': transforms.Compose([
                    transforms.Resize(32),
                    transforms.CenterCrop(32),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            }
            transform = data_transforms["train" if train else "test"]
            cifar_dataset = datasets.CIFAR100(root='./data/CIFAR100', train=train, download=True, transform=transform)
            
            # Get indices for the desired superclass
            mapping_coarse_fine = dataUtils.cifar_mapping_coarse_fine
            fine_label_to_idx = {label: idx for idx, label in enumerate(dataUtils.fine_labels)}
            mapping_coarse_fine_indices = {
                coarse: [fine_label_to_idx[label] for label in fine_labels]
                for coarse, fine_labels in mapping_coarse_fine.items()
            }
            superclass_names = list(dataUtils.cifar_mapping_coarse_fine.keys())
            target_indices = mapping_coarse_fine_indices[superclass_names[superclass_index]]
            indices = [i for i, (_, label) in enumerate(cifar_dataset) if label in target_indices]
            dataset = Subset(cifar_dataset, indices)
            torch.save(dataset, filename)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2 if not train else 0)
        return loader

    
    def getSplitMNIST(permute=False):
        trainloaders = []
        testloaders = []
        NUM_EPOCHS = 5
        for epoch in range(NUM_EPOCHS):
            trainloaders.append(dataUtils.get_mnist_superclass_dataloader(epoch, train=True))
            testloaders.append(dataUtils.get_mnist_superclass_dataloader(epoch, train=False))
        return trainloaders, testloaders
    def getSplitCIFAR():
        trainloaders = []
        testloaders = []
        NUM_SUPERCLASSES = 20
        for superclass in range(NUM_SUPERCLASSES):
            trainloaders.append(dataUtils.get_cifar_superclass_dataloader(superclass, train=True))
            testloaders.append(dataUtils.get_cifar_superclass_dataloader(superclass, train=False))
        return trainloaders, testloaders
        