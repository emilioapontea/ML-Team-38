from typing import Tuple, List
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision.datasets import ImageFolder

def testAccuracy(dataPath: str, model: Module, transform: Compose) -> float:
    testset = ImageFolder(root=dataPath, transform=transform)
    testloader = DataLoader(testset, batch_size=10, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the images: {100 * correct / total} on {dataPath}')
    return 100 * correct / total

def trainModel(
        model: torch.nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        transform: Compose,
        numEpochs: int = 5
        ) -> Tuple[List[float], List[float]]:
    train_acc_history = []
    val_acc_history = []

    i = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for epoch in range(numEpochs):
        model.train()
        for images, labels in dataloader:
            images.to(device)
            optimizer.zero_grad()
            print(f"Training: {epoch} {i}")
            i += 1
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        print(f"Evaluating on Train: {epoch}")
        train_acc_history.append(testAccuracy("./split_dataset/train", model, transform))
        print(f"Evaluating on Val: {epoch}")
        val_acc_history.append(testAccuracy("./split_dataset/val", model, transform))

        print(f"Epoch [{epoch+1}/{numEpochs}], Loss: {loss.item():.4f}")

    return train_acc_history, val_acc_history