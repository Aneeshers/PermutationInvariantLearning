import sys
from model import ClassCIFAR100Classifier, ClassMNISTClassifier, DomainCIFAR100Classifier, DomainMNISTClassifier
from dataUtils import dataUtils
from particle_filters import GB_particle_filter
import torch
def main(arg1, arg2, arg3, arg4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setting =  arg1 #"pure_domain_cifar" # or "pure_domain_cifar" or "selective_domain_mnist" or "selective_domain_cifar" 
    
    # define model and continual learning datasets. 
    # The model and data loaders are defined in the model.py and dataUtils.py files respectively.
    # This process can be used for any dataset -- you can structure the dataset in subdatasets and have a list of loaders.
    if setting == "pure_domain_mnist":
        model = DomainMNISTClassifier().to(device)
        train, test = dataUtils.getSplitMNIST()
    elif setting == "pure_domain_cifar":
        model = DomainCIFAR100Classifier().to(device)
        train, test = dataUtils.getSplitCIFAR()
    elif setting == "selective_domain_mnist":
        model = ClassMNISTClassifier().to(device)
        train, test = dataUtils.getSplitMNIST()
    elif setting == "selective_domain_cifar":
        model = ClassCIFAR100Classifier().to(device)
        train, test = dataUtils.getSplitCIFAR()
    num_particles = int(arg2)
    lr = float(arg3)
    permute = bool(arg4)
    
    # define particle filter
    particle_filter = GB_particle_filter(model, train, test, num_particles, lr, setting, permute)
    
    # train GB_particle_filter
    weights = particle_filter.train_model()
    print("running GB_particle_filter with num_particles = ", num_particles, " and lr = ", lr, " and setting = ", setting, " and permute = ", permute)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python main.py <setting> <particles> <lr> <permute>")
    else:
        arg1 = sys.argv[1]
        arg2 = sys.argv[2]
        arg3 = sys.argv[3]
        arg4 = sys.argv[4]
        main(arg1, arg2, arg3, arg4)
    