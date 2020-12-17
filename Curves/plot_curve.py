import os
import math
import numpy as np
import matplotlib.pyplot as plt

class plot_loss_function_curve():
    def __init__(self):
        self.fig_index = 1
        import matplotlib.pyplot as plt
        self.plt = plt
        self.plt.grid()

    def _plot_curve(self, func, range_a=-10, range_b=10, range_delta=0.1, plot_name="sigmoid", stay=False, **kwargs):
        x = np.arange(range_a, range_b, range_delta)
        y = []
        for t in x:
            y_1 = func(t, **kwargs)
            y.append(y_1)
        if stay and self.fig_index > 1:
            self.fig_index -= 1
        self.plt.grid()
        self.plt.figure(self.fig_index)
        self.fig_index += 1
        self.plt.plot(x, y, label=plot_name)
        self.plt.xlabel("x")
        self.plt.ylabel("y")
        # plt.ylim(0, 1)
        self.plt.legend()

    def sigmoid(self, **kwargs):
        def func(a, **kwargs):
            return 1 / (1 + math.exp(-a))

        name = self._get_param(p_key='name', default_value='', **kwargs)
        name += 'sigmoid'
        self._plot_curve(func=func, plot_name=name, **kwargs)

    def quadratic1(self, **kwargs):
        def func(a, **kwargs):
            return 2 * a

        name = self._get_param(p_key='name', default_value='', **kwargs)
        name += 'quadratic1'
        self._plot_curve(func=func, plot_name=name, **kwargs)

    def quadratic2(self, **kwargs):
        def func(a, **kwargs):
            return a * a

        name = self._get_param(p_key='name', default_value='', **kwargs)
        name += 'quadratic2'
        self._plot_curve(func=func, plot_name=name, **kwargs)

    def truncated_quadratic1(self, **kwargs):
        def func(a, **kwargs):
            lambda_ = self._get_param(p_key='lambda', **kwargs)
            alpha_ = self._get_param(p_key='alpha', **kwargs)
            if np.abs(a) < (np.sqrt(alpha_) / np.sqrt(lambda_)):
                return 2 * lambda_ * a
            else:
                return 0

        name = self._get_param(p_key='name', default_value='', **kwargs)
        name += 'truncated_quadratic1'
        self._plot_curve(func=func, plot_name=name, **kwargs)

    def truncated_quadratic2(self, **kwargs):
        def func(a, **kwargs):
            lambda_ = self._get_param(p_key='lambda', **kwargs)
            alpha_ = self._get_param(p_key='alpha', **kwargs)
            if np.abs(a) < (np.sqrt(alpha_) / np.sqrt(lambda_)):
                return a * lambda_ * a
            else:
                return alpha_

        name = self._get_param(p_key='name', default_value='', **kwargs)
        name += 'truncated_quadratic2'
        self._plot_curve(func=func, plot_name=name, **kwargs)

    def lorentzian1(self, **kwargs):
        def func(a, **kwargs):
            delta = self._get_param(p_key='delta', **kwargs)
            temp = (a / delta) * (a / delta) / 2
            temp = np.log(1 + temp)
            return temp

        name = self._get_param(p_key='name', default_value='', **kwargs)
        name += 'lorentzian1'
        self._plot_curve(func=func, plot_name=name, **kwargs)

    def lorentzian2(self, **kwargs):
        def func(a, **kwargs):
            delta = self._get_param(p_key='delta', **kwargs)
            return 2 * a / (2 * delta * delta + a * a)

        name = self._get_param(p_key='name', default_value='', **kwargs)
        name = 'lorentzian2-' + name
        self._plot_curve(func=func, plot_name=name, **kwargs)

    def show(self):
        self.plt.show()

    def _get_param(self, p_key='delta', default_value=None, **kwargs):
        default_value = 2 if default_value is None else default_value
        delta = default_value if p_key not in kwargs.keys() else kwargs[p_key]
        return delta

    @classmethod
    def demo(cls, ):
        a = plot_loss_function_curve()
        a.sigmoid()
        a.quadratic1()  # stay=True可以保留到上一个图中
        a.truncated_quadratic1(stay=True)
        a.quadratic2()  # stay=True可以保留到上一个图中
        a.truncated_quadratic2(stay=True)
        a.lorentzian1()
        a.lorentzian2()
        for i in [0.1, 1, 10, 100]:
            a.lorentzian2(delta=i, name='delta-%s' % i)
        a.show()

if __name__ == '__main__':
    plot_loss_function_curve.demo()