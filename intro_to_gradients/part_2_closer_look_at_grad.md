# Closer look at `grad` 

## Scalar function
Scalar function is a function that returns a 0-dimensional tensor, i.e. a single value.
In the example below, we use dot product that is a scalar function for two vectors.


```python
import torch
from torch.autograd import grad

a = torch.tensor([0., 1., 2.], requires_grad=True)
b = a @ a
assert b.numel() == 1
db_da = grad(b, a, create_graph=True)[0]  # tensor([0., 2., 4.], grad_fn=<AddBackward0>)
```


## Tensor function and `grad_outputs`

If in the first example we replace dot product with addition, we will get an error, because now the result is a vector.

```python
a = torch.tensor([0., 1., 2.], requires_grad=True)
b = a + a
db_da = grad(b, a, create_graph=True)[0] # RuntimeError: grad can be implicitly created only for scalar outputs
```

From the mathematical standpoint, we should get a jacobian matrix. 
But pytorch does not calculate it like this 
(there is a dedicated `torch.autograd.functional.jacobian`, but it takes an actual python function as input).
Instead, we have to provide a vector and pytorch will return results of a dot product of the jacobian and the vector.
The parameter is called `grad_outputs` which I find a bit misleading. 
I guess it assumes that we use `grad` as part of a chain rule during backward gradient propagation.
Anyway, let's see an example.
We may want to sum along the dimensions of the output. 
To do that, we provide a vector of ones:

```python
a = torch.tensor([0., 1., 2.], requires_grad=True)
b = a + a
db_da = grad(b, a, grad_outputs=torch.ones_like(b),  create_graph=True)[0]  # tensor([2., 2., 2.])
```

It works also for higher dimensional tensors, like this:

```python
a = torch.randn((2, 3), requires_grad=True)
b = torch.randn((3, 4), requires_grad=True)
c = a @ b  # 2x4 matrix
dc_db = grad(c, b, grad_outputs=torch.ones_like(c))[0]  # 3x4 matrix
```

## Handling batches
WIP: `is_grads_batched`
