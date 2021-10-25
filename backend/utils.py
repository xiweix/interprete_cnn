import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import torch
from torchvision import datasets, transforms
from PIL import Image


class mnistDataset(torch.utils.data.Dataset):
    def __init__(self, data, target):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data = data
        self.target = target

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img = self.data[idx]
        img = self.transform(np.array(img))
        img_label = int(self.target[idx])
        img_label_tensor = torch.as_tensor(img_label, dtype=torch.long)
        return img, img_label_tensor


def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data'), train=True, download=True,
                               transform=transform)
    test_set = datasets.MNIST(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data'), train=False,
                              transform=transform)
    return train_set, test_set


def get_total_dataset():
    if os.path.exists(os.path.join(os.path.dirname(os.getcwd()), 'data', 'mnist_total_data.npy')):
        mnist = np.load(os.path.join(os.path.dirname(
            os.getcwd()), 'data', 'mnist_total_data.npy'))
        mnist_target = np.load(os.path.join(os.path.dirname(
            os.getcwd()), 'data', 'mnist_total_target.npy'))
        train_set_np = np.load(os.path.join(os.path.dirname(
            os.getcwd()), 'data', 'mnist_train_data.npy'))
        train_set_target_np = np.load(os.path.join(os.path.dirname(
            os.getcwd()), 'data', 'mnist_train_target.npy'))
        test_set_np = np.load(os.path.join(os.path.dirname(
            os.getcwd()), 'data', 'mnist_test_data.npy'))
        test_set_target_np = np.load(os.path.join(os.path.dirname(
            os.getcwd()), 'data', 'mnist_test_target.npy'))
    else:
        train_set, test_set = get_data()
        train_set_np = train_set.data.numpy()
        train_set_target_np = train_set.targets.numpy()
        test_set_np = test_set.data.numpy()
        test_set_target_np = test_set.targets.numpy()
        mnist = np.concatenate((train_set_np, test_set_np), axis=0)
        mnist_target = np.concatenate(
            (train_set_target_np, test_set_target_np), axis=0)
        np.save(os.path.join(os.path.dirname(os.getcwd()),
                             'data', 'mnist_total_data.npy'), mnist)
        np.save(os.path.join(os.path.dirname(os.getcwd()),
                             'data', 'mnist_total_target.npy'), mnist_target)
        np.save(os.path.join(os.path.dirname(os.getcwd()),
                             'data', 'mnist_train_data.npy'), train_set_np)
        np.save(os.path.join(os.path.dirname(os.getcwd()), 'data',
                             'mnist_train_target.npy'), train_set_target_np)
        np.save(os.path.join(os.path.dirname(os.getcwd()),
                             'data', 'mnist_test_data.npy'), test_set_np)
        np.save(os.path.join(os.path.dirname(os.getcwd()), 'data',
                             'mnist_test_target.npy'), test_set_target_np)
        # example of how to load one image (with index 0-69999)
        # print(mnist[0].shape, mnist_target[0])
        # print(np.min(mnist[0]), np.max(mnist[0]))
        # new_im = Image.fromarray(mnist[0])
        # new_im.save('example.png')
    return mnist, mnist_target, train_set_np, train_set_target_np, test_set_np, test_set_target_np


def get_dataset_from_np():
    if not os.path.exists(os.path.join(os.path.dirname(os.getcwd()), 'data', 'mnist_train_data_b.npy')):
        def threshold(data, t=15):
            data[data >= t] = 255
            data[data < t] = 0
            return data
        mnist, _, train, _, test, _ = get_total_dataset()
        np.save(os.path.join(os.path.dirname(os.getcwd()), 'data',
                             'mnist_total_data_b.npy'), threshold(mnist))
        np.save(os.path.join(os.path.dirname(os.getcwd()), 'data',
                             'mnist_train_data_b.npy'), threshold(train))
        np.save(os.path.join(os.path.dirname(os.getcwd()),
                             'data', 'mnist_test_data_b.npy'), threshold(test))
    train = np.load(os.path.join(os.path.dirname(
        os.getcwd()), 'data', 'mnist_train_data_b.npy'))
    train_target = np.load(os.path.join(os.path.dirname(
        os.getcwd()), 'data', 'mnist_train_target.npy'))
    test = np.load(os.path.join(os.path.dirname(
        os.getcwd()), 'data', 'mnist_test_data_b.npy'))
    test_target = np.load(os.path.join(os.path.dirname(
        os.getcwd()), 'data', 'mnist_test_target.npy'))
    train_set = mnistDataset(train, train_target)
    test_set = mnistDataset(test, test_target)
    return train_set, test_set


def make_outputdir(postfix=''):
    if postfix is None:
        postfix = ''
    timestamp = time.strftime(f'%Y-%m-%d-%H-%M-%S{postfix}')
    outputdir = os.path.join(os.path.dirname(
        os.path.dirname(os.getcwd())), 'outputs', f'{timestamp}')
    os.makedirs(outputdir, exist_ok=True)
    return outputdir, timestamp


def dim_reduction():
    if os.path.exists(os.path.join(os.path.dirname(os.getcwd()), 'data', 'mnist_test_coord.npy')):
        embedding_total = np.load(os.path.join(os.path.dirname(
            os.getcwd()), 'data', 'mnist_total_coord.npy'))
        embedding_train = np.load(os.path.join(os.path.dirname(
            os.getcwd()), 'data', 'mnist_train_coord.npy'))
        embedding_test = np.load(os.path.join(os.path.dirname(
            os.getcwd()), 'data', 'mnist_test_coord.npy'))
    else:
        mnist, mnist_target, train, train_target, test, test_target = get_total_dataset()

        mnist.resize((70000, 784))
        embedding_total = umap.UMAP(
            n_neighbors=30, min_dist=1.0, n_components=2, random_state=27).fit_transform(mnist)
        np.save(os.path.join(os.path.dirname(os.getcwd()),
                             'data', 'mnist_total_coord.npy'), embedding_total)
        sns.set(style='white', rc={'figure.figsize': (10, 8)})
        plt.figure()
        plt.scatter(embedding_total[:, 0], embedding_total[:, 1], c=mnist_target.astype(
            int), s=0.1, cmap='Spectral')
        plt.savefig('scatter_total.png', bbox_inches='tight')

        train.resize((60000, 784))
        embedding_train = umap.UMAP(
            n_neighbors=30, min_dist=1.0, n_components=2, random_state=27).fit_transform(train)
        np.save(os.path.join(os.path.dirname(os.getcwd()),
                             'data', 'mnist_train_coord.npy'), embedding_train)
        sns.set(style='white', rc={'figure.figsize': (10, 8)})
        plt.figure()
        plt.scatter(embedding_train[:, 0], embedding_train[:, 1], c=train_target.astype(
            int), s=0.1, cmap='Spectral')
        plt.savefig(os.path.join(os.getcwd(), 'temp',
                                 'scatter_train.png'), bbox_inches='tight')

        test.resize((10000, 784))
        embedding_test = umap.UMAP(
            n_neighbors=30, min_dist=1.0, n_components=2, random_state=27).fit_transform(test)
        np.save(os.path.join(os.path.dirname(os.getcwd()),
                             'data', 'mnist_test_coord.npy'), embedding_test)
        sns.set(style='white', rc={'figure.figsize': (10, 8)})
        plt.figure()
        plt.scatter(embedding_test[:, 0], embedding_test[:, 1], c=test_target.astype(
            int), s=0.1, cmap='Spectral')
        plt.savefig(os.path.join(os.getcwd(), 'temp',
                                 'scatter_test.png'), bbox_inches='tight')
    return embedding_total, embedding_train, embedding_test


def get_scatterdata():
    _, mnist_target, _, train_target, _, test_target = get_total_dataset()
    embedding_total, embedding_train, embedding_test = dim_reduction()
    write_path = os.path.join(os.path.dirname(
        os.getcwd()), 'src', 'data', 'mnistInfo.js')
    with open(write_path, 'a') as ftxt:
        ftxt.write('export default {\n')
        ftxt.write('  mnist: [\n')
    for ll in range(10):
        content_list = []
        for i in range(mnist_target.shape[0]):
            x, y = embedding_total[i]
            label = mnist_target[i]
            if int(label) == int(ll):
                info = [x, y, label, i]
                content_list.append(info)
        if ll == 9:
            with open(write_path, 'a') as ftxt:
                ftxt.write('    {\n')
                ftxt.write(f'      label: \'{ll}\',\n')
                ftxt.write('      content: [\n')
                ftxt.write(f'        {content_list}\n')
                ftxt.write('      ]\n')
                ftxt.write('    }\n')
                ftxt.write('  ]\n')
                ftxt.write('}\n')
        else:
            with open(write_path, 'a') as ftxt:
                ftxt.write('    {\n')
                ftxt.write(f'      label: \'{ll}\',\n')
                ftxt.write('      content: [\n')
                ftxt.write(f'        {content_list}\n')
                ftxt.write('      ]\n')
                ftxt.write('    },\n')

    write_path = os.path.join(os.path.dirname(
        os.getcwd()), 'src', 'data', 'mnistInfoTrain.js')
    with open(write_path, 'a') as ftxt:
        ftxt.write('export default {\n')
        ftxt.write('  mnist: [\n')
    for ll in range(10):
        content_list = []
        for i in range(train_target.shape[0]):
            x, y = embedding_train[i]
            label = train_target[i]
            if int(label) == int(ll):
                info = [x, y, label, i]
                content_list.append(info)
        if ll == 9:
            with open(write_path, 'a') as ftxt:
                ftxt.write('    {\n')
                ftxt.write(f'      label: \'{ll}\',\n')
                ftxt.write('      content: [\n')
                ftxt.write(f'        {content_list}\n')
                ftxt.write('      ]\n')
                ftxt.write('    }\n')
                ftxt.write('  ]\n')
                ftxt.write('}\n')
        else:
            with open(write_path, 'a') as ftxt:
                ftxt.write('    {\n')
                ftxt.write(f'      label: \'{ll}\',\n')
                ftxt.write('      content: [\n')
                ftxt.write(f'        {content_list}\n')
                ftxt.write('      ]\n')
                ftxt.write('    },\n')

    write_path = os.path.join(os.path.dirname(
        os.getcwd()), 'src', 'data', 'mnistInfoTest.js')
    with open(write_path, 'a') as ftxt:
        ftxt.write('export default {\n')
        ftxt.write('  mnist: [\n')
    for ll in range(10):
        content_list = []
        for i in range(test_target.shape[0]):
            x, y = embedding_test[i]
            label = test_target[i]
            if int(label) == int(ll):
                info = [x, y, label, i]
                content_list.append(info)
        if ll == 9:
            with open(write_path, 'a') as ftxt:
                ftxt.write('    {\n')
                ftxt.write(f'      label: \'{ll}\',\n')
                ftxt.write('      content: [\n')
                ftxt.write(f'        {content_list}\n')
                ftxt.write('      ]\n')
                ftxt.write('    }\n')
                ftxt.write('  ]\n')
                ftxt.write('}\n')
        else:
            with open(write_path, 'a') as ftxt:
                ftxt.write('    {\n')
                ftxt.write(f'      label: \'{ll}\',\n')
                ftxt.write('      content: [\n')
                ftxt.write(f'        {content_list}\n')
                ftxt.write('      ]\n')
                ftxt.write('    },\n')
    print('done!')
