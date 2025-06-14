from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

dataset_path = 'plant_disease_data/PlantDoc-Dataset/PlantDoc-Dataset'
dataset = ImageFolder(root=dataset_path, transform=data_transforms)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

CLASS_NAMES = dataset.classes
print(CLASS_NAMES)