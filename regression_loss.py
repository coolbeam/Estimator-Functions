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
    def lorentzian1a(self, **kwargs):
        def func(a, **kwargs):
            temp = a**2
            temp = np.log(1 + temp)
            return temp

        name = self._get_param(p_key='name', default_value='', **kwargs)
        name += 'lorentzian1a'
        self._plot_curve(func=func, plot_name=name, **kwargs)

    def lorentzian2(self, **kwargs):
        def func(a, **kwargs):
            delta = self._get_param(p_key='delta', **kwargs)
            return 2 * a / (2 * delta * delta + a * a)

        name = self._get_param(p_key='name', default_value='', **kwargs)
        name = 'lorentzian2-' + name
        self._plot_curve(func=func, plot_name=name, **kwargs)

    def abs_robust(self, **kwargs):
        def func(a, **kwargs):
            temp = np.abs(a)
            temp = (temp + 1e-6) ** 0.4
            return temp

        name = self._get_param(p_key='name', default_value='', **kwargs)
        name += 'abs_robust'
        self._plot_curve(func=func, plot_name=name, **kwargs)
    def charbonnier(self, **kwargs):
        def func(a, **kwargs):
            temp = a**2
            temp = (temp + 1e-6) ** 0.4
            return temp

        name = self._get_param(p_key='name', default_value='', **kwargs)
        name += 'charbonnier'
        self._plot_curve(func=func, plot_name=name, **kwargs)
    @classmethod
    def demo(cls, ):
        a = RegressionCurves()
        # range_a=-2.5, range_b=2.5, range_delta=0.01
        # a.quadratic1()  # stay=True可以保留到上一个图中
        # a.truncated_quadratic1(stay=True)
        # a.quadratic2()  # stay=True可以保留到上一个图中
        # a.truncated_quadratic2(stay=True)
        # a.lorentzian1()
        # a.lorentzian2()
        # a.lorentzian1a()
        a.abs_robust(range_a=-10, range_b=10, range_delta=0.01)
        a.charbonnier(range_a=-10, range_b=10, range_delta=0.01,stay=True)
        # for i in [0.1, 1, 10, 100]:
        #     a.lorentzian1(delta=i, name='delta-%s' % i)
        a.show()


if __name__ == '__main__':
    RegressionCurves.demo()
