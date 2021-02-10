import torch

x = torch.tensor(2.0, requires_grad=True)
y = 2*x**4 + x**3 + 3*x**2 + 5*x + 1
print(y)

print(type(y))

y.backward()

print(x.grad)

x = torch.tensor([[1., 2., 3.], [3., 2., 1.]], requires_grad=True)
print(x)

y = 3*x + 2
print(y)

z = 2*y**2
print(z)

out = z.mean()
print(out)

out.backward()
print(x.grad)
