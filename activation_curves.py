import os
import math
import numpy as np
from plot_function import plot_loss_function_curve


class ActivationCurves(plot_loss_function_curve):
    def sigmoid(self, **kwargs):
        def func(a, **kwargs):
            return 1 / (1 + math.exp(-a))

        name = self._get_param(p_key='name', default_value='', **kwargs)
        name += 'sigmoid'
        self._plot_curve(func=func, plot_name=name, **kwargs)

    def sigmoid_x(self, **kwargs):
        def func(a, **kwargs):
            return a / (1 + math.exp(-a))

        name = self._get_param(p_key='name', default_value='', **kwargs)
        name += 'sigmoid-x'
        self._plot_curve(func=func, plot_name=name, **kwargs)

    def sigmoid_bias(self, **kwargs):
        def func(a, **kwargs):
            delta = self._get_param(p_key='delta', **kwargs)
            res = 1 / (1 + math.exp(-a)) + 1 / (delta + math.exp(a))
            return res

        name = self._get_param(p_key='name', default_value='', **kwargs)
        name += 'sigmoid-bias'
        self._plot_curve(func=func, plot_name=name, **kwargs)

    @classmethod
    def demo(cls, ):
        a = ActivationCurves()
        a.sigmoid()  # stay=True可以保留到上一个图中
        a.sigmoid_x()
        a.sigmoid_bias(delta=10)
        a.show()


if __name__ == '__main__':
    ActivationCurves.demo()
