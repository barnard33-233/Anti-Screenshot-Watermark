from CNN import ResNet18
import torch
from torch import nn
import torchvision

import sys
sys.path.append('..')
from dataset import Block32_32

train_ds = Block32_32(Msg="1100000001101000111110100000101000111101000110010000010101100001",
    path="../images/CNN_dataset/train/")
test_ds = Block32_32(Msg="1100000001101000111110100000101000111101000110010000010101100001",
    path="../images/CNN_dataset/eval/")
BATCH_SIZE = 64
model = torchvision.models.resnet18()
train_dl = torch.utils.data.DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_dl = torch.utils.data.DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
)
loss_fn = nn.CrossEntropyLoss()
# 选择adam优化器，选择这个的话正确率相对比较高一些
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
if torch.cuda.is_available():
    model.to('cuda')


def fit(epoch, model, trainloader, testloader):
    correct = 0
    total = 0
    running_loss = 0
    model.train()
    for x, y in trainloader:
        if torch.cuda.is_available():
            x, y = x.to('cuda'), y.to('cuda')
        # print(x.shape)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            correct += (y_pred==y).sum().item()
            total += y.size(0)
            running_loss += loss.item()
    #    exp_lr_scheduler.step()
    epoch_loss = running_loss / len(trainloader.dataset)
    epoch_acc = correct / total

    test_correct = 0
    test_total = 0
    test_running_loss = 0

    model.eval()
    with torch.no_grad():
        for x, y in testloader:
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred==y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()

    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / test_total

    print('epoch: ', epoch,
        'loss： ', round(epoch_loss, 3),
        'accuracy:', round(epoch_acc, 3),
        'test_loss： ', round(epoch_test_loss, 3),
        'test_accuracy:', round(epoch_test_acc, 3)
    )

    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc


epochs = 50

train_loss = []
train_acc = []
test_loss = []
test_acc = []
high_test_acc = 0
for epoch in range(epochs):
    epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch,
        model,
        train_dl,
        test_dl)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)
    if epoch_test_acc > high_test_acc:
        high_test_acc = epoch_test_acc
        torch.save(model, './pretrain/resnet18.pth')
print(train_loss)
print(test_loss)
