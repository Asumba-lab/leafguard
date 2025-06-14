
import torch
import torch.nn as nn
import os

# Dummy model class definition
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(3 * 224 * 224, 4)  # assuming RGB 224x224 input, 4 classes

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        return self.fc(x)

# Instantiate and save only the state_dict
model = DummyModel()
model_path = "scripts/plant_disease_model.pth"
torch.save(model.state_dict(), model_path)

print("Dummy model (4-class) saved as plant_disease_model.pth")