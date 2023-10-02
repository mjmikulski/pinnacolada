# Part 1: Leaf that requires grad

## Basics terms

**Tensor** in the computer world means simply a multidimensional array, i.e. a
bunch of numbers indexed by one or more integers. To be precise, there exist
also zero-dimensional tensors, which are just single numbers. Some people say
that tensors are a generalization of matrices to more than two dimensions.

If you have studied general relativity before, you may know that mathematical
tensors have such things as covariant and contravariant indices. But forget
about it --- in PyTorch tensors are just multidimensional arrays. No finesse here.

**Leaf tensor** is a tensor that is a leaf (in the sense of a graph theory) of a
computation graph. We will look at those below.

The `requires_grad` property of a tensor tells PyTorch whether it should
remember how this tensor is used in further computations. Think of tensors with
`requires_grad=True` as variables, while tensors with `requires_grad=False` as
constants.

## Leaf tensors

Let's start by creating a few tensors and checking their properties
`requires_grad` and `is_leaf`.

```python
import torch

a = torch.tensor([3.], requires_grad=True)
b = a * a

c = torch.tensor([5.])
d = c * c

assert a.requires_grad is True and a.is_leaf is True
assert b.requires_grad is True and b.is_leaf is False
assert c.requires_grad is False and c.is_leaf is True
assert d.requires_grad is False and d.is_leaf is True  # sic!
del a, b, c, d
```

`a` is a leaf as expected, and `b` is not because it is a result of a
multiplication. `a` is set to require grad, so naturally `b` inherits this
property.

`c` is a leaf obviously, but why `d` is a leaf? The reason `d.is_leaf` is True
stems from a specific convention: all tensors with `requires_grad` set to False
are considered leaf tensors, as per 
[PyTorch's documentation](https://pytorch.org/docs/stable/generated/torch.Tensor.is_leaf.html).
While mathematically, `d` is not a leaf (since it results from another
operation, `c * c`), gradient computation will never extend beyond it. In other
words, there won't be any derivative with respect to `c`, allowing `d` to be
treated as a leaf.

In a nutshell, in PyTorch, leaf tensors are either:

* Directly inputted (i.e. not calculated from other tensors) and have
  `requires_grad=True`. Example: neural network weights that are randomly
  initialized.
* Do not require gradients at all, regardless of whether they are directly
  inputted or computed. In the eyes of autograd, these are just constants.
  Examples:
    * any neural network input data,
    * an input image after mean removal or other operations, which involves
      only non-gradient-requiring tensors.

A small remark for those who want to know more.
The `requires_grad` property is inherited as illustrated here:

```python
a = torch.tensor([5.], requires_grad=True)
b = torch.tensor([5.], requires_grad=True)
c = torch.tensor([5.], requires_grad=False)

d = torch.sin(a * b * c)

assert d.requires_grad == any((x.requires_grad for x in (a, b, c)))
```

*Code remark: all code snippets should be self-contained except for imports that
I include only when they appear first time.
I drop them in order to minimize boilerplate code.
I trust that the reader will be able to take care of those easily.*

## Grad retention

A separate issue is gradient retention. All nodes in the computation graph,
meaning all tensors used, have gradients computed if they require grad.
However, only leaf tensors retain these gradients. This makes sense because
gradients are typically used to update tensors, and only leaf tensors are
subject to updates during training. Non-leaf tensors, like `b` in the earlier
example, are not directly updated; they change as a result of changes in `a`,
so their gradients can be discarded. However, there are scenarios, especially
in Physics-Informed Neural Networks (PINNs), where you might want to retain
the gradients of these intermediate tensors. In such cases, you will need to
explicitly mark non-leaf tensors to retain their gradients. Let's see:

```python
a = torch.tensor([3.], requires_grad=True)
b = a * a
b.backward()

assert a.grad is not None
assert b.grad is None  # generates a warning

```

You probably have just seen a warning:

```
UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at aten\src\ATen/core/TensorBody.h:491.)
```

So let's fix it by forcing `b` to retain its gradient

```python
a = torch.tensor([3.], requires_grad=True)
b = a * a
b.retain_grad()  # <- the difference
b.backward()

assert a.grad is not None
assert b.grad is not None
```

## Mysteries of grad

Now let's look at the famous grad itself. What is it? Is it a tensor? If so, is
it a leaf tensor? Does it require or retain grad?

```python
a = torch.tensor([3.], requires_grad=True)
b = a * a
b.retain_grad()  # <- the difference
b.backward()

assert isinstance(a.grad, torch.Tensor)
assert a.grad.requires_grad is False and a.grad.retains_grad is False and a.grad.is_leaf is True
assert b.grad.requires_grad is False and b.grad.retains_grad is False and b.grad.is_leaf is True
```

Apparently:

- grad itself is a tensor,
- grad is a leaf tensor,
- grad does not require grad.

Does it retain grad? This question does not make sense because it does not
require grad in the first place. We will come back to the question of the grad
being a leaf tensor in a second, but now we will test a few things.

### Multiple backwards and `retain_graph`

What will happen when we calculate the same grad twice?

```python
a = torch.tensor([3.], requires_grad=True)
b = a * a
b.retain_grad()
b.backward()
try:
    b.backward()
except RuntimeError:
    """
    RuntimeError: Trying to backward through the graph a second time (or 
    directly access saved tensors after they have already been freed). Saved 
    intermediate values of the graph are freed when you call .backward() or 
    autograd.grad(). Specify retain_graph=True if you need to backward through 
    the graph a second time or if you need to access saved tensors after 
    calling backward.
    """
```

The error message explains it all. This should work:

```python
a = torch.tensor([3.], requires_grad=True)
b = a * a
b.retain_grad()

b.backward(retain_graph=True)
print(a.grad)  # prints tensor([6.])

b.backward(retain_graph=True)
print(a.grad)  # prints tensor([12.])

b.backward(retain_graph=False)
print(a.grad)  # prints tensor([18.])

# b.backward(retain_graph=False)  # <- here we would get an error, because in the 
# previous call we did not retain the graph.
```

Side (but important) note: you can also observe, how the gradient accumulates in
`a`: with every iteration it is added.

### Powerful `create_graph` argument

```python
a = torch.tensor([5.], requires_grad=True)
b = a * a
b.retain_grad()
b.backward(create_graph=True)

# Here an interesting thing happens: now a.grad will require grad! 
assert a.grad.requires_grad is True
assert a.grad.is_leaf is False

# On the other hand, the grad of b does not require grad, as previously. 
assert b.grad.requires_grad is False
assert b.grad.is_leaf is True
```

The above is very useful: `a.grad` which mathematically is
$\frac{\partial b}{\partial a}$ is not a constant (leaf) anymore, but a
regular member of the computation graph that can be further used. 
We will use that fact in Part 2.

Why the `b.grad` does not require grad?
Because derivative of `b` with respect to `b` is simply 1.

If the `backward` feel unintuitive for you now, don't worry. We will soon switch
to another method called nomen omen `grad` that allows to precisely choose
ingredients of the derivatives. Before, two side notes:

* Side note 1: if you set `create_graph` to True, it also sets `retain_graph`
  to True (if not explicitly set). In the pytorch code it looks exactly like this:

    ```python (skip=True)
    if retain_graph is None:
        retain_graph = create_graph
    ```

* Side note 2:
  You probably saw a warning like this:

    ``` 
    UserWarning: Using backward() with create_graph=True will create a reference 
    cycle between the parameter and its gradient which can cause a memory leak. 
    We recommend using autograd.grad when creating the graph to avoid this. If you 
    have to use this function, make sure to reset the .grad fields of your 
    parameters to None after use to break the cycle and avoid the leak. 
    (Triggered internally at C:\cb\pytorch_1000000000000\work\torch\csrc\autograd\engine.cpp:1156.)
      Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
    ```

  And we will follow the advice and use `autograd.grad` now.

## Taking derivatives with `autograd.grad` function

Now let's move from the somehow high-level `.backward()` method to lower level
`grad` method that explicitly calculates derivative of one tensor with respect
to another.

```python
from torch.autograd import grad

a = torch.tensor([3.], requires_grad=True)
b = a * a * a
db_da = grad(b, a, create_graph=True)[0]
assert db_da.requires_grad is True
```

Similarly, as with `backward`, the derivative of `b` with respect to `a` can be
treated as a function and differentiated further. So in other words, the
`create_graph` flag can be understood as:
> when calculating gradients, keep the history of how they were calculated,
> so we can treat them as non-leaf tensors that require grad, and use further.

In particular, we can calculate second-order derivative:

```python (continued=True)
d2b_da2 = grad(db_da, a, create_graph=True)[0]
assert d2b_da2.item() == 18
assert d2b_da2.requires_grad is True
```

As I said before: this is actually the key property that allows us to do PINN
with pytorch.

Side note: the `grad` function returns a tuple and always the first element of
it is what we need.
TODO: investigate what else it could return.

Let's meet in Part 2 :)
