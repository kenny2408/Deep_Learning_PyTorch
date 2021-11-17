# Importar librerías

# Para graficar
import numpy as np
import matplotlib.pyplot as plt  # %matplotlib inline

# Principales de PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Otras mas de PyTorch
from torch.utils.data import DataLoader  # Para dividir nuestros datos
from torch.utils.data import sampler  # Para muestrar datos
import torchvision.datasets as dataset  # Para importar DataSets
import torchvision.transforms as T  # Para aplicar transformaciones a nuestros datos

# Jupyter Theme
from jupyterthemes import jtplot

jtplot.style()

NUM_TRAIN = 55000
BATCH_SIZE = 512

# Get our training, validation and test data
# data_patch 'D:\DataspellProjects\Deep_Learning_PyTorch\Data'
data_patch = 'D:\DataspellProjects\Deep_Learning_PyTorch\Data'
mnist_train = dataset.MNIST(data_patch, train=True, download=True, transform=T.ToTensor())
loader_train = DataLoader(mnist_train, batch_size=BATCH_SIZE, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
mnist_val = dataset.MNIST(data_patch, train=True, download=True, transform=T.ToTensor())
loader_val = DataLoader(mnist_val, batch_size=BATCH_SIZE, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 60000)))

mnist_test = dataset.MNIST(data_patch, train=False, download=True, transform=T.ToTensor())
loader_test = DataLoader(mnist_test, batch_size=BATCH_SIZE)

# DataLoader

# len(y_test)

y_test = loader_test.dataset.targets
x_test = loader_test.dataset.data
print(y_test.shape)
print(x_test.shape)

for it, (x, y) in enumerate(loader_test):
    print(f'{it} X.shape is {x.shape}, y.shape is {y.shape}')

input_feature = len(torch.flatten(x_test[0]))
input_feature


def plot_number(image):
    jtplot.style(grid=False)
    plt.imshow(image.squeeze(), cmap=plt.get_cmap('gray'))
    # plt.show()
    plt.axis('off')
    jtplot.style(grid=True)


def plot_mnist_grid():
    y_test = loader_test.dataset.targets
    x_test = loader_test.dataset.data
    samples = 8
    plt.rcParams['figure.figsize'] = (15.0, 15.0)  # set default sizeof plots
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for label, example in enumerate(classes):
        # random idx = np.random.randint(0, len(mnist_test))
        plt_idxs = np.flatnonzero(y_test == label)  # get all data equal to lab
        plt_idxs = np.random.choice(plt_idxs, samples, replace=False)  # muestre
        # print(plt_idxs)
        for i, idx in enumerate(plt_idxs):
            plt_idx = i * len(classes) + label + 1  # plt index starts a 1
            plt.subplot(samples, len(classes), plt_idx)
            # print(example, i, plt.idx)
            # plt.imshow(x_test[idx].type(torch.float32).reshape(28, 28)
            # plt.axis('off')
            plot_number(x_test[idx])
            if i == 0:
                plt.title(example)

    plt.show()


plot_mnist_grid()

# test an image

rnd_idx = np.random.randint(len(y_test))
print(f'La imagen muestreada representa un: {y_test[rnd_idx]}')
plot_number(x_test[rnd_idx])

# GPUs

dtype = torch.float32
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('using device: ', device)


# Compute Accuracy

def compute_acc(loader, model, eval_mode=False):
    num_correct = 0
    num_total = 0
    if eval_mode: model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, pred = scores.max(1)
            assert pred.shape == y.shape, 'Label shape and prediction shape do'
            num_correct += (pred == y).sum()
            num_total += pred.size(0)

        return float(num_correct) / num_total


# High Level Model

input_feature = len(torch.flatten(x_test[0]))
hidden = 1000
num_classes = 10

model1 = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features=input_feature, out_features=hidden),
    nn.ReLU(),
    nn.Linear(in_features=hidden, out_features=num_classes),
)


def plot_loss(losses):
    fig = plt.figure()
    f1 = fig.add_subplot()
    f1.set_ylabel("Cost")
    f1.set_xlabel("Epoch")
    f1.set_title("Cost vs Epoch")
    f1.plot(losses)
    plt.show()


def train(model, optimizer, epochs=100):
    model = model.to(device=device)
    losses = []

    num_batches = len(loader_train)
    for epochs in range(epochs):
        accum_loss = 0
        for i, (x, y) in enumerate(loader_train):
            model.train()
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)

            # calculate score
            scores = model(x)
            cost = F.cross_entropy(input=scores, target=y)

            # calculate gradients
            optimizer.zero_grad()  # reset gradients
            cost.backward()

            # update parameters
            optimizer.step()

            # save loss
            accum_loss += cost.item()

        losses.append(accum_loss / num_batches)

        print(f'Epoch: {epochs}, loss: {cost.item()}, val accuracy: {compute_acc(loader_val, model, True)}')
        print()
    plot_loss(losses)


learning_rate = 1e-2
epochs = 10
optimizer = torch.optim.SGD(model1.parameters(), lr=learning_rate)
train(model1, optimizer, epochs)

# More elegant high leve model
print()
print('More elegant high leve model')
print()


class FC_model(nn.Module):
    def __init__(self, input_features, hidden=1000, num_classes=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_features, out_features=hidden),
            nn.ReLU(),
            nn.Linear(in_features=hidden, out_features=num_classes),
        )

    def forward(self, x):
        return self.model(x)


input_feature = len(torch.flatten(x_test[0]))
hidden = 1000
num_classes = 10
learning_rate = 1e-2
epochs = 10
model2 = FC_model(input_feature)
optimizer = torch.optim.Adam(model2.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model2.parameters(), lr=learning_rate)
train(model2, optimizer, epochs)

compute_acc(loader_test, model2)
print()
print(f'Compute accuracy: {compute_acc(loader_test, model2, True)}')

# Guardando el modelo
model_patch = 'D:\DataspellProjects\Deep_Learning_PyTorch\Data\MNIST\FC_model1st.pth'

torch.save(model2.state_dict(), model_patch)

input_feature = len(torch.flatten(x_test[0]))
loaded_model = FC_model(input_feature=input_feature)
loaded_model.load_state_dict(torch.load(model_patch))
loaded_model = loaded_model.to(device=device)
loaded_model.eval()


def sample_number():
    rnd_idx = np.random.randint(10000)
    print(f'{y_test[rnd_idx]}')
    image2 = x_test[rnd_idx][None, :, :]
    return image2.to(device=device)


image = sample_number()
_, class_is = loaded_model(image).max(1)
print(f'The predicted class is: {class_is[0]}')
