import numpy as np
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
model = nn.Sequential(
    nn.Linear(2, 30),
    nn.ReLU(),
    nn.Linear(30, 1)
)

model.to(device)

print(model.forward(torch.FloatTensor(np.array([[1, 2]])).to(device)))

