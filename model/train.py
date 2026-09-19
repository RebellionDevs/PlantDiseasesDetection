import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import load_data
from model import get_model
from tqdm import tqdm

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)
        total += labels.size(0)
        
        pbar.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / total
    epoch_acc = correct.double() / total
    return epoch_loss, epoch_acc.item()

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct.double() / total
    return val_loss, val_acc.item()

def main():
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 0.001
    DATA_DIR = "./data"
    WEIGHTS_DIR = "./model/weights"
    
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    train_loader, test_loader, class_names = load_data(data_dir=DATA_DIR, batch_size=BATCH_SIZE)

    model = get_model(classes=len(class_names), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"| Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(WEIGHTS_DIR, "best_model.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': class_names
            }, save_path)
            print(f"--> Save best model with acc: {best_acc:.4f}")

if __name__ == "__main__":
    main()