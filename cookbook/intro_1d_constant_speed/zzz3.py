import torch
from torch.autograd import grad

"""
Now let's make our first PINN with single data point
"""

v_0 = torch.tensor(5.)
t = torch.tensor(2.)
x_true = torch.tensor(10.)

a = torch.rand(1, requires_grad=True)
b = torch.rand(1, requires_grad=True)

print(f'initial: {a=} {b=}')

t.requires_grad = True

h = a * v_0 + b
x_pred = h * t

v = grad(x_pred, t, create_graph=True)[0]
loss_physics = (v - v_0) ** 2
loss_physics.backward(retain_graph=True)
print(f'{a.grad=} {b.grad=}')

# t.requires_grad = False

loss_data = (x_pred - x_true) ** 2
loss_data.backward()
print(f'{a.grad=} {b.grad=}')

# We have gradients to we could now update the weights - we will do that soon. But now let's use more tha one data point.


"""
Ok, that was easy. Let's jump into having a batch of data
"""

v_0 = torch.tensor([5., 3., 1.])
t = torch.tensor([2., 6., 4.])
x_true = torch.tensor([10., 18., 4.])

a = torch.rand(1, requires_grad=True)
b = torch.rand(1, requires_grad=True)

print(f'initial: {a=} {b=}')

t.requires_grad = True

h = a * v_0 + b
x_pred = h * t

x_pred = torch.reshape(x_pred, (3, 1))
t = torch.reshape(t, (3, 1))

# v = grad(x_pred, t, grad_outputs=torch.ones_like(x_pred), create_graph=True, is_grads_batched=True)[0]
v = grad(x_pred, t, grad_outputs=torch.ones_like(x_pred), create_graph=True, is_grads_batched=False)

v = v[0]
# loss_physics = (v - v_0) ** 2
# loss_physics.backward(retain_graph=True)
# print(f'{a.grad=} {b.grad=}')


print('go to sleep')