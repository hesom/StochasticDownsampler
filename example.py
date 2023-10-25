import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from Downsampler import StochasticDownsampler

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential( # mnist starts at 28 x 28
                nn.Conv2d(1, 32, 3, padding=1),
                nn.ReLU(),
                StochasticDownsampler(resolution=(14,14), spp=4, reduction="mean"), # 32 x 14 x 14
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                StochasticDownsampler(resolution=(7,7), spp=4, reduction="mean"), # 64 x 7 x 7
            )
        self.fc = nn.Sequential(
                nn.Linear(64*7*7, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )

    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def main():
    batch_size = 64
    epochs = 20
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081))
        ])
    dataset_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset_test = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, num_workers=4, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, num_workers=4, shuffle=False)

    model = SimpleNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100 * batch_idx / len(train_loader), loss.item()))

        # val
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100.*correct / len(test_loader.dataset)))

if __name__ == "__main__":
    main()

