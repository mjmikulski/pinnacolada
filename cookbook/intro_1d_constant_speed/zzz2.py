import torch
from torch.autograd import grad

# Leaf vs requires grad

a = torch.tensor([3.], requires_grad=True)
b = a * a

c = torch.tensor([5.])
d = c * c

assert a.requires_grad is True and a.is_leaf is True
assert b.requires_grad is True and b.is_leaf is False
assert c.requires_grad is False and c.is_leaf is True
assert d.requires_grad is False and d.is_leaf is True

# The fact that d.is_leaf is True steams from the convention
# All Tensors that have requires_grad which is False will be leaf Tensors by convention.
# ref: https://pytorch.org/docs/stable/generated/torch.Tensor.is_leaf.html
# Mathematically speaking it is not a leaf (as it is a result of other operation: c * c).
# But gradient computation will never go beyond it (i.e. there may not be any derivative with respect to c),
# so d can be treated as a leaf.
# To summarize: in pytorch, leaves are tensors that
# - are directly inputted (i.e. not calculated) and require grad,
#  like e.g. nn weights which are randomly sampled during model creation, or
# - don't require grad (no matter if inputted or calculated, for autograd, they are just constants),
# e.g. input data for nn (inputted data) or input image after mean removal (calcualated,
# but mean removal involves only non-grad-requiring tensors)

del a, b, c, d

# A separate thing is grad retention. All nodes in the graph (i.e. all used tensors)
# have grad computed, if they require grad, but only those that are leaf tensors have it retained.
# It makes sense - we use grad (usually) to update tensors, and only leaf tensors are those that can be updated during training.
# The non-leaf tensors (like b) in the example above, are not updated directly (they change as a result of change of a), so their gradient may be dropped.
# But sometimes we want to have this gradient of intermediate tensors (and that's often the case for PINNs).
# Then we have to explicitly mark non leaf tensors as retaining gradient.

a = torch.tensor([3.], requires_grad=True)
b = a * a
b.backward()
assert a.grad is not None
assert b.grad is None

"""
You even see a warning:
```
UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at aten\src\ATen/core/TensorBody.h:491.)
  assert b.grad is None
```
"""

# Now we set b to retain grad (`a` retains it always)


a = torch.tensor([3.], requires_grad=True)
b = a * a
b.retain_grad()  # <- the difference
b.backward()
assert a.grad is not None
assert b.grad is not None

"""
Now let's look at the grad itself. What is it? Is it a tensor? If so, is it a leaf tensor? Does it require or retains grad?

Apparently:
- grad itself is a tensor
- grad is a leaf tensor
- grad does not require grad

Does it retain grad? This question does not make sense, because it does not require grad. 

"""

assert a.grad.requires_grad is False and a.grad.retains_grad is False and a.grad.is_leaf is True
assert b.grad.requires_grad is False and b.grad.retains_grad is False and b.grad.is_leaf is True

"""
Whan happen when we calculate the same grad twice?

"""
a = torch.tensor([3.], requires_grad=True)
b = a * a
b.retain_grad()
b.backward()
try:
    b.backward()
except RuntimeError:
    """
    RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.
    """

# Ok, let's try:

a = torch.tensor([3.], requires_grad=True)
b = a * a
b.retain_grad()
b.backward(retain_graph=True)
print(a.grad)
b.backward(retain_graph=True)
print(a.grad)
b.backward(retain_graph=False)
print(a.grad)
# b.backward(retain_graph=False)  <- here we would get an error, because in the previous call we did not retain the graph

# BTW, you can also see, how the gradient accumulates in a: with every iteration it is added.

# Now let's see what create_graph flag does
a = torch.tensor([5.], requires_grad=True)
b = a * a
b.retain_grad()
b.backward(create_graph=True)

# Here an interesting thing happen: now a.grad will require grad! Which means that we can do further calculations with it!
assert a.grad.requires_grad is True

# On the other hand, the grad of b does not require grad
assert b.grad.requires_grad is False
# Why? I guess this is what `backwards` is designed to do - calculate grads for leaf tensors.

"""
Side note: if you set create_graph to True, it also sets retain_graph to True (if not explicitly set).
In pytorhc code it is:
    if retain_graph is None:
        retain_graph = create_graph
        
        
Side note 2:
You see here a warning: 
UserWarning: Using backward() with create_graph=True will create a reference cycle between the parameter and its gradient which can cause a memory leak. We recommend using autograd.grad when creating the graph to avoid this. If you have to use this function, make sure to reset the .grad fields of your parameters to None after use to break the cycle and avoid the leak. (Triggered internally at C:\cb\pytorch_1000000000000\work\torch\csrc\autograd\engine.cpp:1156.)
  Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
"""

# Now let's move from the somehow high-level `.backward()` to
# lower level grad method that explicitly calculates derivative of one tensor with respect to another tensor.


a = torch.tensor([3.], requires_grad=True)
b = a * a * a
db_da = grad(b, a, create_graph=True)[0]

# Now the derivative of b with respect can be treated as a function and differentiated further, see this:
assert db_da.requires_grad is True

# So in other words, the create_graph flag can be understood as: when calculating gradients,
# keep the history of how they were calculated, so we can treat them as non-leaf tensors that require grad and use further.

# This is actually key property that allows as to do PINN with pytorch.

d2b_da2 = grad(db_da, a, create_graph=True)[0]
assert d2b_da2.item() == 18
assert d2b_da2.requires_grad is True

print('ok')

del a, b, db_da, d2b_da2

"""
Now let's make our first PINN with single data points
"""
# data
# (5, 2) -> 10


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
