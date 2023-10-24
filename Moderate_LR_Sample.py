import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)
x = np.random.rand(100, 1) * 10
y = 2 * x ** 3 + 3 * x ** 2 - 5 * x + 2 + 10 * np.random.randn(100, 1)

x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.hidden1 = nn.Linear(1, 10)
        self.hidden2 = nn.Linear(10, 10)
        self.hidden3 = nn.Linear(10, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, x_):
        x_ = torch.relu(self.hidden1(x_))
        x_ = torch.relu(self.hidden2(x_))
        x_ = torch.relu(self.hidden3(x_))
        x_ = self.output(x_)
        return x_


model = NeuralNetwork()
optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (epoch + 1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

x_test = torch.tensor(np.random.rand(100, 1) * 10).float()
predicted = model(x_test)

plt.scatter(x, y, label='Data')
plt.scatter(x_test.numpy(), predicted.detach().numpy(), label='Fitted Curve')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fit vs. original')
plt.legend()
plt.show()
