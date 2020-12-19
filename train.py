import torch
import torchvision

from torchsummary import summary

from model import Model

def train(epoch, total_epoch):
    network.to(device).train()
    print('Train Epoch {} of {}'.format(epoch, total_epoch))
    t = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(t):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            train_losses.append(loss.item())
            t.set_postfix({'Loss': sum(train_losses)/len(train_losses)})
            train_counter.append(batch_idx*64 + (epoch + 1)
                                 * len(train_loader.dataset))
    torch.save(network.state_dict(), 'result/model.pth')
    torch.save(optimizer.state_dict(), 'result/optimizer.pth')

def test():
    network.to(device).eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = network(data)
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
    summary(model, (3, 320, 320), depth=6)
    # test()
    # total_training_time = 0
    # for epoch in range(1, n_epochs + 1):
    #     start = time()
    #     train(epoch, n_epochs)
    #     end = time()
    #     print('\nTraning process takes {} seconds'.format(end - start))
    #     total_training_time += end - start
    #     test()
    # print('Total traning process takes {} seconds'.format(total_training_time))