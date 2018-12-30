import torch
import torch.nn as nn
import torch.onnx as onnx
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

def info(msg, char = "#", width = 75):
    print("")
    print(char * width)
    print(char + "   %0*s" % ((-1*width)+5, msg) + char)
    print(char * width)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

def get_dataloader(train=True, batch_size=64):
    digits = datasets.MNIST('data', train=train, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: x.reshape(28*28))
                        ]),
                        target_transform=transforms.Compose([
                            transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, y, 1))
                        ])
                     )

    return DataLoader(digits, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)


def train(model, device, dataloader, cost, optimizer, epoch):
    model.train()
    for batch, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = cost(pred, Y)
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print('loss: {:>10f}  [{:>5d}/{:>5d}]'.format(loss.item(), batch * len(X), len(dataloader.dataset)))
    

def test(model, device, dataloader, cost):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch, (X, Y) in enumerate(dataloader):
            X, Y = X.to(device), Y.to(device)
            pred = model(X)

            test_loss += cost(pred, Y).item()
            correct += (pred.argmax(1) == Y.argmax(1)).type(torch.float).sum().item()

    test_loss /= len(dataloader.dataset)
    correct /= len(dataloader.dataset)
    print('\nTest Error:')
    print('acc: {:>0.1f}%, avg loss: {:>8f}'.format(100*correct, test_loss))

def save_model(model, device, path):
    # create dummy variable to traverse graph
    x = torch.randint(255, (1, 28*28), dtype=torch.float).to(device) / 255
    onnx.export(model, x, path)
    print('Saved model to {}'.format(path))


def main(epochs=15, learning_rate=.001):
    # use GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # get data loaders
    training = get_dataloader(train=True)
    testing = get_dataloader(train=False)

    # model
    model = CNN().to(device)
    info('Model')
    print(model)

    # cost function
    cost = torch.nn.BCELoss()

    # optimizers
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, 5)
    
    for epoch in range(1, epochs + 1):
        info('Epoch {}'.format(epoch))
        scheduler.step()
        print('Current learning rate: {}'.format(scheduler.get_lr()))
        train(model, device, training, cost, optimizer, epoch)
        test(model, device, testing, cost)
        
    # save model
    info('Saving Model')
    save_model(model, device, 'model.onnx')
    print('Saving PyTorch Model as model.pth')
    torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
    main()