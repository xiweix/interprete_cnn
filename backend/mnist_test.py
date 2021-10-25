import os
import time
import gzip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
import matplotlib.pyplot as plt
import random


class BatchFlatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Model(nn.Module):
    def __init__(self, n_layers, output_sizes, drop_out_rate):
        super(Model, self).__init__()
        self.n_layers = n_layers
        self.layers = []
        for i in range(n_layers):
            if i == 0:
                in_features = 1
            else:
                in_features = output_sizes[i - 1]
            if i == 0:
                self.layers.extend([
                    nn.Conv2d(in_features, output_sizes[i], 2, 1),
                    nn.ELU(),
                    nn.MaxPool2d(2, 2),
                ])
            else:
                self.layers.extend([
                    nn.Conv2d(in_features, output_sizes[i], 2, 1),
                    nn.ELU(),
                ])
        self.layers.append(BatchFlatten())
        if drop_out_rate > 0.:
            self.layers.extend([
                nn.Dropout(drop_out_rate),
                nn.Linear(int(((28 - 2 * n_layers) / 2)
                              ** 2 * output_sizes[-1]), 10),
            ])
        else:
            self.layers.extend([
                nn.Linear(int(((28 - 2 * n_layers) / 2)
                              ** 2 * output_sizes[-1]), 10),
            ])
        self.layers = nn.ModuleList(self.layers)
    def forward(self, x):
        for layers in self.layers:
            x = layers(x)
        return x


def train(model, device, train_loader, optimizer, train_loss_list, train_acc_list, epoch, flag):
    train_loss = 0
    correct = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss /= len(train_loader.dataset)
    train_accuracy = 100. * correct / len(train_loader.dataset)
    train_loss_list.append(train_loss)
    train_acc_list.append(train_accuracy)
    # print('{}: Epoch {}, Train: Average loss: {:.4f}'.format(
    #     flag, epoch, train_loss))


def test(model, device, test_loader, test_loss_list, test_acc_list, epoch, flag):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            loss_function = nn.CrossEntropyLoss()
            test_loss += loss_function(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    test_loss_list.append(test_loss)
    test_acc_list.append(test_accuracy)
    # print('{}: Epoch: {}, Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    #     flag, epoch, test_loss, correct, len(test_loader.dataset), test_accuracy))


def mnist_main(epochs, train_batch_size, lr_step_gamma, n_layers, output_sizes, drop_out_rate, init_lr, train_set, test_set, outputdir, flag):
    assert n_layers == len(
        output_sizes), f'n_layers ({n_layers}) is not equal to len(output_sizes) ({len(output_sizes)})'
    use_cuda = torch.cuda.is_available()
    print(flag, use_cuda, 'Start training')
    device = torch.device('cuda' if use_cuda else 'cpu')
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=1000,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )
    model = Model(n_layers=n_layers, output_sizes=output_sizes,
                  drop_out_rate=drop_out_rate).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=init_lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=lr_step_gamma)
    # print(flag, model)
    out_path = os.path.join(outputdir, flag)
    os.makedirs(out_path, exist_ok=True)
    # train and test
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train(model, device, train_loader, optimizer,
              train_loss_list, train_acc_list, epoch, flag)
        test(model, device, test_loader,
             test_loss_list, test_acc_list, epoch, flag)
        scheduler.step()
        # save trained model & training and testing results
        # model_path = os.path.join(out_path, f'epoch.{epoch}.statedict.pt.gz')
        model_path = os.path.join(out_path, f'epoch.{epoch}.pt.gz')
        np.save(os.path.join(
            out_path, f'epoch.{epoch}.train_loss.npy'), train_loss_list)
        np.save(os.path.join(
            out_path, f'epoch.{epoch}.test_acc.npy'), train_acc_list)
        np.save(os.path.join(
            out_path, f'epoch.{epoch}.test_loss.npy'), test_loss_list)
        np.save(os.path.join(
            out_path, f'epoch.{epoch}.test_acc.npy'), test_acc_list)

        with gzip.open(model_path, 'wb') as f:
            # torch.save(model.state_dict(), f)
            torch.save(model, f)

        if epoch >= 2:
            # save current version
            cur_model_path = os.path.join(out_path, 'model.statedict.pt.gz')
            np.save(os.path.join(out_path, 'train_loss.npy'), train_loss_list)
            np.save(os.path.join(out_path, 'train_acc.npy'), train_acc_list)
            np.save(os.path.join(out_path, 'test_loss.npy'), test_loss_list)
            np.save(os.path.join(out_path, 'test_acc.npy'), test_acc_list)
            with gzip.open(cur_model_path, 'wb') as f:
                torch.save(model.state_dict(), f)
        epoch_time = time.perf_counter() - epoch_start
        print(f'{flag}: Epoch: {epoch}, Time: {epoch_time}s')

        # # Below indicates how to load a saved model's statedict
        # model_path = os.path.join(out_path, 'model.statedict.pt.gz')
        # with gzip.open(model_path, 'rb') as f:
        #     model.load_state_dict(torch.load(f))
        #     model.to(device)


def get_activations(copy_model_path, n_layers, features, drop, sample_data, flag1, flag2):
    copy_model = Model(n_layers=n_layers,
                       output_sizes=features, drop_out_rate=drop)
    with gzip.open(copy_model_path, 'rb') as copy_f:
        copy_model = torch.load(copy_f).to('cpu')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    sample_tensor = transform(sample_data)
    sample_tensor = sample_tensor.unsqueeze(0)

    activations = []
    activations_name = []

    def hook(self, input, output):
        activations.append(output.detach().squeeze().numpy())
        activations_name.append(self.__class__.__name__)

    for la in copy_model.layers:
        la.register_forward_hook(hook)

    sample_output = copy_model(sample_tensor)
    prediction = int(sample_output.argmax(dim=1, keepdim=True))

    temp_dir = os.path.join(os.getcwd(), 'temp')

    # delete old activations for this model
    for fname in os.listdir(temp_dir):
        if fname.startswith(f'{flag1}.{flag2}'):
            os.remove(os.path.join(temp_dir, fname))
    
    saved_activations_name = []
    # save new files
    j = 0
    for i in range(len(activations)):
        if(activations[i].ndim == 3):
            random_select = random.randint(0, activations[i].shape[0] - 1)
            im = activations[i][random_select, :, :]
            plt.imsave(os.path.join(
                temp_dir, f'{flag1}.{flag2}_{activations_name[i]}_{j}.png'), im)
            saved_activations_name.append(f'{activations_name[i]}_{j}')
            j += 1
    return saved_activations_name, prediction
