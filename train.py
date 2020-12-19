from tqdm import tqdm
from time import time

import torch
import torchvision
from torchsummary import summary
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import torch.optim as optim

from model import Model
from dataloader import HADataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device {}'.format(device))

def train(epoch, total_epoch, loader, model, optimizer, log_interval=1):
    train_losses = []
    train_counter = []
    model.to(device).train()
    print('Train Epoch {} of {}'.format(epoch, total_epoch))
    t = tqdm(loader)
    for batch_idx, (image, target) in enumerate(t):
        image = image.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            train_losses.append(loss.item())
            t.set_postfix({'Loss': sum(train_losses)/len(train_losses)})
            train_counter.append(batch_idx*64 + (epoch + 1)
                                 * len(train_loader.dataset))
    torch.save(model.state_dict(), 'result/model.pth')
    torch.save(optimizer.state_dict(), 'result/optimizer.pth')

def test(loader, model):
    model.to(device).eval()
    test_loss = 0
    test_losses = []
    correct = 0
    t = tqdm(loader)
    with torch.no_grad():
        for image, target in t:
            image = image.to(device)
            target = target.to(device)
            output = model(image)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    model = Model()
    # summary(model, (3, 320, 320), depth=6)
    train_loader = DataLoader(HADataset('train'), batch_size=24, shuffle=True)
    test_loader = DataLoader(HADataset('test'), batch_size=24, shuffle=False)
    # test(test_loader, model)
    optimizer = optim.Adam(model.parameters())
    total_training_time = 0
    n_epochs = 2
    for epoch in range(1, n_epochs + 1):
        start = time()
        train(epoch, n_epochs, train_loader, model, optimizer)
        end = time()
        print('\nTraning process takes {} seconds'.format(end - start))
        total_training_time += end - start
        test(test_loader, model)
    # print('Total traning process takes {} seconds'.format(total_training_time))