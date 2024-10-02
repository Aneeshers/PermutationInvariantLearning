# Permutation Invariant Learning with High-Dimensional Particle Filters

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Use Gradient-based Particle Filter with your own Torch Model
```model
# Simple model for Cifar100
class Net(nn.Module):

	def __init__(self):
	
		super(Net, self).__init__()
	
		self.conv1 = nn.Conv2d(1, 32, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(32, 64, 5)
		self.fc1 = nn.Linear(64 * 4 * 4, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)
	
	  
	
	def forward(self, x):
	
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))	
		x = x.view(-1, 64 * 4 * 4)
		x = F.relu(self.fc1(x))	
		x = F.relu(self.fc2(x))	
		x = self.fc3(x)
		
		return x
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Net().to(device)
```

**Define Continual Learning Dataset and Setting**
```setting
setting = "pure_domain_mnist" #"pure_domain_cifar" # or "pure_domain_cifar" or "selective_domain_mnist" or "selective_domain_cifar"
train, test = dataUtils.getSplitMNIST()
```

**Train Particle Filter**
```Train
particle_filter = GB_particle_filter(net, train, test, num_particles, lr, setting, permute)

# train GB_particle_filter

weights = particle_filter.train_model()
```

## Run experiment Models

To train the model(s) in the paper, run this command:

```train
python main.py "pure_domain_mnist" 100 0.01 "True" 
```

>📋  This will run the domain mnist weighted logits particle filter on 100 particles with a 0.01 lr on a Split MNIST Permutation.


## Pre-trained Models

Coming soon ➡️
