import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(conv_layer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.nl = nn.ReLU()

    def forward(self, x):
        return self.nl(self.bn(self.conv(x)))

class fc_layer(nn.Module):
    def __init__(self, in_features, out_features):
        super(fc_layer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.nl = nn.ReLU()

    def forward(self, x):
        return self.nl(self.linear(x))

class ConvLayers(nn.Module):
    def __init__(self, layers_config, include_pooling=False):
        super(ConvLayers, self).__init__()
        self.layers = nn.ModuleList()
        for config in layers_config:
            self.layers.append(conv_layer(*config))
        
        if include_pooling:
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.pooling = None

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        if self.pooling:
            x = self.pooling(x)
        return x

class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(fc_layer(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x):
        x = x.view(x.size(0), -1)  
        for layer in self.layers:
            x = layer(x)
        return x
    
mnist_conv_config = [(1, 16)]  
mnist_mlp_config = [12544, 400, 400]

conv_layers_mnist = ConvLayers(mnist_conv_config)
mlp_mnist = MLP(mnist_mlp_config)


cifar_conv_config = [(3, 16), (16, 32, 3, 2), (32, 64, 3, 2), (64, 128, 3, 2), (128, 256, 3, 2, 0)]
cifar_mlp_config = [256, 2000, 2000]

conv_layers_cifar = ConvLayers(cifar_conv_config, include_pooling=True)
mlp_cifar = MLP(cifar_mlp_config)

class ClassMNISTClassifier(nn.Module):
    def __init__(self):
        super(ClassMNISTClassifier, self).__init__()
        self.convE = ConvLayers(mnist_conv_config)
        self.flatten = nn.Flatten()
        self.fcE = MLP(mnist_mlp_config)
        self.classifier = fc_layer(400, 10)  

    def forward(self, x):
        x = self.convE(x)
        x = self.flatten(x)
        x = self.fcE(x)
        x = self.classifier(x)
        return x

class ClassCIFAR100Classifier(nn.Module):
    def __init__(self):
        super(ClassCIFAR100Classifier, self).__init__()
        self.convE = ConvLayers(cifar_conv_config, include_pooling=True)
        self.flatten = nn.Flatten()
        self.fcE = MLP(cifar_mlp_config)
        self.classifier = fc_layer(2000, 100)

    def forward(self, x):
        x = self.convE(x)
        x = self.flatten(x)
        x = self.fcE(x)
        x = self.classifier(x)
        return x

class DomainCIFAR100Classifier(nn.Module):
    def __init__(self):
        super(DomainCIFAR100Classifier, self).__init__()
        self.convE = ConvLayers(cifar_conv_config, include_pooling=True)
        self.flatten = nn.Flatten()
        self.fcE = MLP(cifar_mlp_config)
        self.classifier = fc_layer(2000, 5)  

    def forward(self, x):
        x = self.convE(x)
        x = self.flatten(x)
        x = self.fcE(x)
        x = self.classifier(x)
        return x

class DomainMNISTClassifier(nn.Module):
    def __init__(self):
        super(DomainMNISTClassifier, self).__init__()
        self.convE = ConvLayers(mnist_conv_config)
        self.flatten = nn.Flatten()
        self.fcE = MLP(mnist_mlp_config)
        self.classifier = fc_layer(400, 2)  

    def forward(self, x):
        x = self.convE(x)
        x = self.flatten(x)
        x = self.fcE(x)
        x = self.classifier(x)
        return x
    
class modelUtils:
    def print_model_summary(model, input_size=(8, 3, 32, 32)):
        model.eval()
        with torch.no_grad():  
            input_tensor = torch.randn(input_size)
            input_tensor = input_tensor.to(next(model.parameters()).device)
            print("Layer Name \t Output Shape")
            print("-" * 40)
            def forward_hook(module, input, output):
                print(f"{module.__class__.__name__} \t {output.shape}")
            hooks = []
            for layer in model.children():
                hook = layer.register_forward_hook(forward_hook)
                hooks.append(hook)
            model(input_tensor)
            for hook in hooks:
                hook.remove()