import torch
from torch.autograd import Variable

input_size = 2
output_size = 1

hidden_layer_nodes = 10
batch_size = 4

# Create Tensors to hold the inputs and outputs and wrap them in variables.
x = Variable(torch.FloatTensor([[0, 0], [1, 0], [0, 1], [1, 1]]))
y = Variable(torch.FloatTensor([[0], [1], [0], [1]]))

x_pred = Variable(torch.FloatTensor([[0, 0], [1, 0], [0, 1], [1, 1]]))

# The neural network model to use
model = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_layer_nodes),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_layer_nodes, output_size),
    torch.nn.Sigmoid()
)

# Use Mean squared error as loss function
loss_fn = torch.nn.MSELoss()

# Using stochastic gradient descent algorithm for optimization
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

print("Print initial parameters")
print(list(model.parameters()))

print("Training neural network")

t = 0
while(True):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    print(t, loss.data[0])

    # Stop if loss is less that threshold
    if loss.data[0] < 0.001:
        break

    optimizer.zero_grad()

    # Backpropagate errors
    loss.backward()

    # Perform updates on parameters
    optimizer.step()

    t += 1

print("Predicting output values for [0, 0], [1, 0], [0, 1], [1, 1]")

predicted_values = model(x_pred)
print(predicted_values)
