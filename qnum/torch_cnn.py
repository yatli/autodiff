import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn

class SimpleCnn(torch.nn.Module):
    def __init__(self):
        super(SimpleCnn, self).__init__()
        self.convs = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=2))
        self.fcs = torch.nn.Sequential(
            torch.nn.Linear(32 * 8 * 8, 128),
            torch.nn.ReLU(True),
            torch.nn.Linear(128, 10))
    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        return self.fcs(x)

batch_size = 40

transform = transforms.Compose(
    [transforms.ToTensor()]) #,
     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
net = SimpleCnn()
net.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

for epoch in range(40):  # loop over the dataset multiple times

    ncorrect = 0
    ntotal = 0

    # training loop
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        ntotal += batch_size
        _, predicted = torch.max(outputs, 1)
        for x in (predicted==labels).squeeze():
            if x: ncorrect += 1

        # print statistics
        if i % 100 == 0:
            batch_loss = loss.item()
            print(f'| TRAIN epoch= {epoch} step= {ntotal} batch_loss= {batch_loss} avg_acc= {ncorrect / ntotal}')

    test_loss = 0
    ntotal = 0
    ncorrect = 0
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            ntotal += batch_size
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            for x in (predicted==labels).squeeze():
                if x: ncorrect += 1
    print(f'| TEST epoch= {epoch} avg_loss= {test_loss / ntotal} acc= {ncorrect / ntotal}')

print('Finished Training')
