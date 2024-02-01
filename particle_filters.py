import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from tqdm import tqdm
import numpy as np
from dataUtils import dataUtils


class particleUtils:
    def parameters_to_particle(net):
        return [nn.Parameter(p.data.clone()) for p in net.parameters()]
    def particleToNet(particle, net):
        for p_particle, p_net in zip(particle, net.parameters()):
            p_net.data = p_particle.data.clone()
    def print_top_particle_weights(weights, losses, top_n=100):
        indexed_weights_losses = list(enumerate(zip(weights, losses)))
        sorted_particles = sorted(indexed_weights_losses, key=lambda x: x[1][0], reverse=True)
        top_particles = sorted_particles[:top_n]
        particles_str = ', '.join(f"Particle {i+1}: Weight={w:.4f}, Loss={l:.4f}" for i, (w, l) in top_particles)
        print(f"Top {top_n} particles: [{particles_str}]")
    def injectNoise(particles):
        for particle in particles:
            for param in range(len(particle)):
                r = random.uniform(0.001, 0.04)
                particle[param].data = particle[param] + r * torch.randn_like(particle[param])
    def initParticles(net, num_particles):
        return [particleUtils.parameters_to_particle(net) for _ in range(num_particles)]

class GB_particle_filter:
    def __init__(self, model, train, test, num_particles, lr, setting, permute):
        self.model = model
        self.train = train
        self.test = test
        self.num_particles = num_particles
        self.lr = lr
        self.setting = setting
        self.permute = permute
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        #init particles:
        self.particles = particleUtils.initParticles(self.model.to(self.device), self.num_particles)
        particleUtils.injectNoise(self.particles)
        if setting == "pure_domain_mnist" or setting == "selective_domain_mnist":
            self.superclass_names = dataUtils.mnist_superclasses
        else:
            self.superclass_names = list(dataUtils.cifar_mapping_coarse_fine.keys())
        self.all_epochs_particle_accuracies = [[[0.0 for _ in range(len(self.train))] for _ in range(len(self.superclass_names))] for _ in range(self.num_particles)]

        self.mapping_coarse_fine = dataUtils.cifar_mapping_coarse_fine
        fine_label_to_idx = {label: idx for idx, label in enumerate(dataUtils.fine_labels)}
        self.mapping_coarse_fine_indices = {
            coarse: [fine_label_to_idx[label] for label in fine_labels]
            for coarse, fine_labels in self.mapping_coarse_fine.items()
        }
        self.digit_pairs = dataUtils.digit_pairs
        if self.permute:
            if self.setting == "pure_domain_mnist" or self.setting == "selective_domain_mnist":
                combined_list = list(zip(self.digit_pairs, self.superclass_names, self.train, self.test))

                # Shuffle combined list
                random.shuffle(combined_list)

                self.digit_pairs, self.superclass_names, self.train, self.test = zip(*combined_list)

                self.digit_pairs = list(self.digit_pairs)
                self.superclass_names = list(self.superclass_names)
                self.train = list(self.train)
                self.test = list(self.test)
                print("PERMUTATION: ")
                print(self.digit_pairs)
                print(self.superclass_names)
            else:
                # Shuffle mapping_coarse_fine and self.superclass_names together
                superclass_indices = {name: i for i, name in enumerate(self.superclass_names)}
                items = list(self.mapping_coarse_fine.items())
                random.shuffle(items)
                self.mapping_coarse_fine = dict(items)

                self.superclass_names = list(self.mapping_coarse_fine.keys())
                self.train = [self.train[superclass_indices[name]] for name in self.superclass_names]
                self.test = [self.test[superclass_indices[name]] for name in self.superclass_names]

                self.mapping_coarse_fine_indices = {
                    coarse: [fine_label_to_idx[label] for label in fine_labels]
                    for coarse, fine_labels in self.mapping_coarse_fine.items()
                }
                print("PERMUTATION: ")
                print(self.superclass_names)

            
    def train_model(self):
        weights = [1.0 / self.num_particles] * self.num_particles
        criterion = nn.CrossEntropyLoss()
        net = self.model.to(self.device)

        # Training loop
        for epoch in range(len(self.train)):
            losses = [0 for _ in self.particles]
            updated_losses = [0 for _ in self.particles]
            trainloader = self.train[epoch]
            print(len(trainloader))
            for t, data in enumerate(trainloader, 0):
                inputs, targets = data
                if self.setting == "pure_domain_mnist":
                    current_pair = dataUtils.digit_pairs[epoch]
                    targets = torch.tensor([0 if target == current_pair[0] else 1 for target in targets])
                if self.setting == "pure_domain_cifar":
                    targets = torch.tensor([self.mapping_coarse_fine_indices[self.superclass_names[epoch]].index(target) for target in targets])
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                for i, particle_i in enumerate(self.particles):
                    particleUtils.particleToNet(particle=particle_i, net=net)  # Set the network parameters to those of the current particle
                    net.train()
                    
                    for param in particle_i:
                        if param.grad is not None:
                            param.grad.zero_()
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)
                    losses[i] = loss  
                    loss.backward() 
                    
                    # Manual Gradient Descent -- an optimizer step could be used here instead.
                    with torch.no_grad():
                        for param, p in zip(particle_i, net.parameters()):
                            update_term_param = -self.lr * p.grad  
                            param.data += update_term_param
                    particleUtils.particleToNet(particle_i, net)
                    updated_loss = criterion(net(inputs), targets)
                    updated_losses[i] = updated_loss
                    
                    # Weighting
                    weights[i] += -0.5 * (losses[i] + updated_losses[i]).item()            
                    net.zero_grad()
                
                weightSum = np.log(sum([np.exp(w) for w in weights]))
                for l in range(len(weights)):
                    weights[l] -= weightSum

            self.eval(epoch, weights)
        
        return weights
    def eval(self, epoch, weights):
        
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        ORANGE = '\033[93m'
        YELLOW = '\033[95m'
        RESET = '\033[0m'
        
        net = self.model.to(self.device)
        particle_accuracies = [[0.0 for _ in range(self.num_particles)] for _ in range(len(self.superclass_names))]

        for superclass in range(epoch + 1):
            testloader = self.test[superclass]
            total = 0
            weighted_correct = 0
            random_particle_index = random.randint(0, len(self.particles) - 1)

            if self.setting == "pure_domain_mnist" or self.setting == "selective_domain_mnist":
                current_pair = self.digit_pairs[superclass]

            if (self.setting == "pure_domain_mnist") or (self.setting == "pure_domain_cifar"):
                for data in testloader:
                    images, labels = data                    
                    if self.setting == "pure_domain_mnist":
                        labels = torch.tensor([0 if label == current_pair[0] else 1 for label in labels])
                    if self.setting == "pure_domain_cifar":
                        labels = torch.tensor([self.mapping_coarse_fine_indices[self.superclass_names[superclass]].index(label) for label in labels])
                    images, labels = images.to(self.device), labels.to(self.device)
                    # Initialize combined logits                
                    combined_logits = None
                    with torch.no_grad():
                        for k, particle_k in enumerate(self.particles):
                            particleUtils.particleToNet(particle=particle_k, net=net)
                            net.eval()
                            outputs = net(images)

                            # Weighted logits
                            weight = np.exp(weights[k])
                            weighted_outputs = outputs * weight

                            if combined_logits is None:
                                combined_logits = weighted_outputs
                            else:
                                combined_logits += weighted_outputs
                            _, predicted = torch.max(outputs.data, 1)
                            correct = (predicted == labels).sum().item()
                            particle_accuracies[superclass][k] += correct
                    
                    total += labels.size(0)
                    _, predicted = torch.max(combined_logits.data, 1)
                    weighted_correct += (predicted == labels).sum().item()
                for k in range(self.num_particles):
                    particle_accuracies[superclass][k] = 100 * particle_accuracies[superclass][k] / total
                weighted_accuracy = 100 * weighted_correct / total
                print(f'{GREEN}Weighted accuracy of the network on the test images of superclass {self.superclass_names[superclass]}: {weighted_accuracy:.2f}%')
                non_weighted_avg_accuracy = sum(particle_accuracies[superclass]) / self.num_particles
                print(f'{BLUE}Non-weighted average accuracy for superclass {self.superclass_names[superclass]}: {non_weighted_avg_accuracy:.2f}%')
                random_particle_accuracy = particle_accuracies[superclass][random_particle_index]
                print(f'{YELLOW}Random Single particle accuracy for superclass {self.superclass_names[superclass]} (Particle {random_particle_index}): {random_particle_accuracy:.2f}%{RESET}')
            else:
                for k, particle_k in enumerate(self.particles):
                    particleUtils.particleToNet(particle=particle_k, net=net)
                    net.eval()
                    correct = 0
                    particle_total = 0
                    with torch.no_grad():
                        for data in testloader:
                            images, original_labels = data
                            images, original_labels = images.to(self.device), original_labels.to(self.device)

                            outputs = net(images)

                            if self.setting == "selective_domain_mnist":
                                logits_indices = torch.tensor(current_pair, device=self.device)
                                outputs_relevant = outputs[:, logits_indices]
                                remapped_labels = torch.tensor([0 if label == current_pair[0] else 1 for label in original_labels], device=self.device)

                            if self.setting == "selective_domain_cifar":
                                superclass_indices = self.mapping_coarse_fine_indices[self.superclass_names[superclass]]
                                outputs_relevant = outputs[:, superclass_indices]
                                remapped_labels = torch.tensor([self.mapping_coarse_fine_indices[self.superclass_names[superclass]].index(label.item()) for label in original_labels], device=self.device)

                            _, predicted_relevant = torch.max(outputs_relevant.data, 1)
                            particle_total += remapped_labels.size(0)
                            correct += (predicted_relevant == remapped_labels).sum().item()

                    accuracy = 100 * correct / particle_total
                    particle_accuracies[superclass][k] = accuracy

                    weighted_correct += correct * np.exp(weights[k])
                    total += particle_total
                weighted_accuracy = 100 * weighted_correct / particle_total
                print(f'{GREEN}Weighted accuracy of the network on the test images of superclass {self.superclass_names[superclass]}: {weighted_accuracy:.2f}%{RESET}')
                non_weighted_avg_accuracy = sum(particle_accuracies[superclass]) / len(particle_accuracies[superclass])
                print(f'{BLUE}Non-weighted average accuracy for superclass {self.superclass_names[superclass]}: {non_weighted_avg_accuracy:.2f}%{RESET}')
                random_particle_accuracy = particle_accuracies[superclass][random_particle_index]
                print(f'{YELLOW}Random Single particle accuracy for superclass {self.superclass_names[superclass]} (Particle {random_particle_index}): {random_particle_accuracy:.2f}%{RESET}')

            
        for particle_index in range(self.num_particles):
            for superclass_index in range(epoch+1):
                self.all_epochs_particle_accuracies[particle_index][superclass_index][epoch] = particle_accuracies[superclass_index][particle_index]
