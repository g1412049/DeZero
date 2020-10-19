# import modules
import numpy as np
import unittest

# User module
# Variable class
class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        
    def cleargrad(self):
        self.grad = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()

            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

            if x.creator is not None:
                funcs.append(x.creator)

class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)

        self.inputs  = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    def backward(self, gy):
        return gy, gy

class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x  = self.inputs[0].data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x  = self.input.data
        gx = np.exp(x) * gy
        return gx

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.))
        y = square(x)
        expected = np.array(4.)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.))
        y = square(x)
        y.backward()
        expected = np.array(6.)
        self.assertEqual(x.grad, expected)

def add(x0, x1):
    return Add()(x0, x1)

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def numerical_diff(f, x, eps=1.e-6):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))

def main():
    x = Variable(np.array(3.))
    y = Variable(np.array(3.))
    z = add(add(x, x), x)
    z.backward()
    print(z.data)
    print(x.grad)


if __name__ == "__main__":
    main()