import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(42)
x = np.random.rand(100, 1) * 10
y = 2 * x ** 3 + 3 * x ** 2 - 5 * x + 2 + 10 * np.random.randn(100, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).float()


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.hidden1 = nn.Linear(1, 10)
        self.hidden2 = nn.Linear(10, 10)
        self.hidden3 = nn.Linear(10, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, x_in):
        x_in = torch.relu(self.hidden1(x_in))
        x_in = torch.relu(self.hidden2(x_in))
        x_in = torch.relu(self.hidden3(x_in))
        x_in = self.output(x_in)
        return x_in


model = NeuralNetwork()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

num_epochs = 5000
loss_list = []
for epoch in range(num_epochs):

    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print('\rEpoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss/len(y_train)), end='')

x_test = torch.tensor(np.random.rand(100, 1) * 10).float()
y_predicted = model(x_test)

# plt.scatter(x_train, y_train, label='train_set', marker='o')
plt.scatter(x, y, label='ground_truth')
plt.scatter(x_test, y_predicted.detach().numpy(), label='test_set', marker='x')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fit vs. original')
plt.legend()
plt.show()
