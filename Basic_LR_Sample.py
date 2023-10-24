import matplotlib.pyplot as plt
import torch

"""
the source code author converted the [1,2,3,4,5] to [[1],[2],[3],[4],[5]], I have no idea why
tensor.view() reshapes the tensor without copying memory, similar to the reshape() of numpy.
"""

X = torch.arange(-5, 5, 0.1)
func = -5 * X
Y = func + 0.4 * torch.randn(X.size())

"""
The forward() function takes an input and generates a prediction. (input towards output is defined as FORWARD)
The criterion() function calculates the loss (between the ) and stores it in loss variable.
Based on the model loss, the backward() method computes the gradients and w.data stores the updated parameters.
"""


# defining the function for forward pass for prediction
def forward(x):
    return w * x + b


# evaluating data points with Mean Square Error.
def criterion(y_pred, y):
    return torch.mean((y_pred - y) ** 2)


# arbitrary initialization of the two params of the network
w = torch.tensor(-10.0, requires_grad=True)
b = torch.tensor(-20.0, requires_grad=True)

step_size = 0.1
loss_list = []
epochs = 20

for i in range(epochs):
    # making predictions with forward pass (input and the output are both the entire dataset)
    Y_pred = forward(X)
    # calculating the loss between original and predicted data points
    loss = criterion(Y_pred, Y)
    # storing the calculated loss in a list
    loss_list.append(loss.item())
    # backward pass for computing the gradients of the loss w.r.t to learnable parameters
    loss.backward()
    # updating the parameters after each iteration
    w.data = w.data - step_size * w.grad.data
    b.data = b.data - step_size * b.grad.data
    # zeroing gradients after each iteration
    w.grad.data.zero_()
    b.grad.data.zero_()
    # printing the values for understanding
    print('epochs{}, \tloss{}, \t{}, \t{}'.format(i, loss.item(), w.item(), b.item()))

# Plotting the loss after each iteration
plt.plot(loss_list, 'r')
plt.tight_layout()
plt.xlabel("Epochs/Iterations")
plt.ylabel("Loss")
plt.show()
