import torch
import math
import random


class DynamicNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn())
        self.b = torch.nn.Parameter(torch.randn())
        self.c = torch.nn.Parameter(torch.randn())
        self.d = torch.nn.Parameter(torch.randn())

    def forward(self, x):
        y = self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3
        for exp in range(4, random.randint(4, 6)):
            y = y + self.e * x ** exp
        return y

    def string(self):
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3 + {self.e.item()} x^4 ? + {self.e.item()} x^5 ?'


x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

model = DynamicNet()

criterion = torch.nn.MSELoss(reduction="sum")

optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9)

for t in range(2000):
    y_pred = model(x)
    loss = criterion(y_pred, y)

    if t % 100 == 99:
        print(t, loss.item())

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


print(f'Result: {model.string()}')