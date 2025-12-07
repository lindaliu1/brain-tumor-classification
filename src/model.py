import torch
import torch.nn as nn
import torchvision.models as models

class BrainTumorClassifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, freeze_layers=True):
        super().__init__()

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        if freeze_layers:
            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model.layer4.parameters():
                param.requires_grad = True

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
def get_model(num_classes=3, pretrained=True, freeze_layers=True):
    return BrainTumorClassifier(num_classes=num_classes, pretrained=pretrained, freeze_layers=freeze_layers)

if __name__ == "__main__":
    model = get_model()
    print(model)

    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(output.shape)