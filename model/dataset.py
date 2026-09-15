import torch, os
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
])

def load_data(data_dir="./data", batch_size=32, train_ratio=0.8):

    dataset = datasets.ImageFolder(root=data_dir)
    class_names = dataset.classes

    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_set, test_set = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_set.dataset.transform = train_transform

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"All: {len(dataset)}")
    print(f"{len(class_names)} : {class_names}")

    return train_loader, test_loader, class_names

if __name__ == "__main__":
    train_loader, test_loader, class_names = load_data()