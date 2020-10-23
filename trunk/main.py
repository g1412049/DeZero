# import modules
import numpy as np
import unittest

# User module
# --Variable class--
class Variable:
    # Variable class インスタンス
    # クラス初期化を実施
    def __init__(self, data):
        # 入力データがNoneでかつnp.ndarrayでない場合はエラー生成
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        # 入力データをメンバ変数に代入
        self.data = data
        # 勾配情報と生成元関数を設定するメンバ変数を作成
        self.grad = None
        self.creator = None

    # 勾配情報を消去するメゾット
    def cleargrad(self):
        self.grad = None

    # 関数と出力変数を関連付けるメゾット
    def set_creator(self, func):
        self.creator = func

    # 自動微分を計算するメゾット
    def backward(self):
        # 勾配を記述するメンバ変数self.gradがNoneの場合
        # 既存の配列と同じサイズですべての要素の値が 1 の配列を生成
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            # 指定した位置の要素を削除して値を取得
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


# --Function class--
# 各関数の基本構造を記述した抽象クラス
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

# --Add class--
# 各変数を足し合わせる関数のクラス
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    def backward(self, gy):
        return gy, gy

# --Square class--
# 変数に対して2乗を計算する関数のクラス
class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x  = self.inputs[0].data
        gx = 2 * x * gy
        return gx


# --Exp class--
# 変数に対してexpを計算する関数のクラス
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

# 各変数を足し合わせる関数
def add(x0, x1):
    return Add()(x0, x1)

# 変数に対して2乗を計算する関数
def square(x):
    return Square()(x)

# 変数に対してexpを計算する関数
def exp(x):
    return Exp()(x)

# 入力引数がスカラであった場合は行列形式に変換する関数
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
    z = add(add(x, x), x)
    z.backward()
    print(z.data)
    print(x.grad)


if __name__ == "__main__":
    main()