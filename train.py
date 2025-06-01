import torch, time
from torch import optim, nn
from data_loader import get_mnist_loaders
from model import MNIST_CNN

def train(epochs=5, lr=1e-3, device='cpu'):
    model = MNIST_CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = get_mnist_loaders()
    for epoch in range(1, epochs+1):
        model.train()
        total_loss, correct = 0, 0
        start = time.time()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
        elapsed = time.time() - start
        acc = correct / len(train_loader.dataset)
        print(f"Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.4f}, Time={elapsed:.1f}s")

    # Save checkpoint
    torch.save(model.state_dict(), 'mnist_cnn.pth')
    print("Saved model to mnist_cnn.pth")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train(device=device)
