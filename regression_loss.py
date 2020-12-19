import os
import math
import numpy as np
from plot_function import plot_loss_function_curve


class RegressionCurves(plot_loss_function_curve):
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

    @classmethod
    def demo(cls, ):
        a = RegressionCurves()
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
    RegressionCurves.demo()