import torch
import argparse
from image_dataset import get_datasets, TestDataset
from model import Net
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader


def validate(net, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the %d test images: %d %%' % (len(loader.dataset), 100 * correct / total))


def check(net, loader):
    total = 0
    com = 0
    with torch.no_grad():
        for data in loader:
            images, _ = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += predicted.size(0)
            com += predicted.sum().item()
    print(f"{com} / {total} are commercials")


def parse_args():
    parser = argparse.ArgumentParser(description="Training tool for TV station logo recognition")
    parser.add_argument('--path', default='data/preset')
    parser.add_argument('--split-ratio', default=0.8)
    parser.add_argument('--epochs', default=25)
    parser.add_argument('--batch-size', default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    hard_dataset = TestDataset('/home/ktoztam/CLionProjects/CommercialDetector/classifier/data/test')
    train_dataset, test_dataset = get_datasets(args.path, args.split_ratio)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=16)
    hard_loader = DataLoader(hard_dataset, batch_size=args.batch_size)
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(args.epochs):
        running_loss = 0.0
        for data in train_dataloader:
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('[%d] loss: %.3f' % (epoch + 1, running_loss))
        validate(net, test_dataloader)
        check(net, hard_loader)
        torch.onnx.export(net, test_dataset[0][0].unsqueeze(0), "polsat_detector.onnx", export_params=True)


if __name__ == '__main__':
    main()
