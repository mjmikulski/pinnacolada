import torch as th
from torch.autograd import grad
from torch.nn import grad as grad2

# Weights
A = th.Tensor([[2, 0],
               [1, 0]])
A.requires_grad = True

B = th.Tensor([[0, 1],
               [0.6, 0]])
B.requires_grad = True


def zero_grad():
    A.grad = None
    B.grad = None


def model(inputs):
    u1, u2 = A @ inputs
    w = th.stack([u1, u2 * t])
    outputs = B @ w
    return outputs


# Inputs
inputs = th.Tensor([1, 3])
t = inputs[1]
t.requires_grad = True  # Otherwise grad will not work: RuntimeError: One of the differentiated Tensors does not require grad

# b_21 = B[1, 0]

x, v = model(inputs)

x_grad = grad(x, t, create_graph=True)[0]  # Otherwise x_grad.requires_grad is False after it. But we will use it so we want it to be True.

loss = (v - x_grad)**2

loss.backward()

print('ok')

comments = """
inputs = th.Tensor([1, 3])
t = inputs[1]
t.requires_grad = True  # Otherwise grad will not work: RuntimeError: One of the differentiated Tensors does not require grad
# t.retain_grad()

x, v = model(inputs)

x_grad = grad(x, t, create_graph=True)[0]  # Otherwise x_grad.requires_grad is False after it. But we will use it so we want it to be True.

b_21 = B[1, 0]
v_b21 = grad(v, b_21, create_graph=True)
RuntimeError: One of the differentiated Tensors appears to not have been used in the graph. Set allow_unused=True if this is the desired behavior.
Jak to teraz rozwiązać?

-> Rozwiązanie - trzeba zrobić grad po B, bo operacja indeksowania nie pozwala na grad
b_21 jest traktowane jako osobna zmienna która nie była użyta w grafie.








Tensor.retain_grad() → None
Enables this Tensor to have their grad populated during backward(). This is a no-op for leaf tensors.

Rozumiem, że w takim razie nie ma sensu dla t, bo t nie zależy od niczego, więc jest leafem.




Kolejny problem:
RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.
 po tym jak zrobiłem

v_B = grad(v, B)
v_A = grad(v, A)

Rozumiem, w takim razie, że muszę dać retain_graph = True za pierwszy razem



# nie ok
v_B = grad(v, B)
v_A = grad(v, A)

# ok
v_B = grad(v, B, retain_graph=True)
v_A = grad(v, A)


# nie ok, tzn za drugim nie działa
v_B = grad(v, B)
v_A = grad(v, A, retain_graph=True)


# też nie ok
v_B = grad(x, B)
v_A = grad(v, B)

# też nie ok
v_B = grad(x, A)
v_A = grad(v, B)


Czyli rozumiem, że graf jako taki można raz tykać, a później już błąd.


Też nie:
v.retain_grad()
x.retain_grad()
A.retain_grad()
B.retain_grad()
_ = grad(v, B)
_ = grad(x, A)

Czyli retain_grad to nie to samo co retain_graph


"""
