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
        self.plt.grid()

    def show(self):
        self.plt.show()

    def _get_param(self, p_key='delta', default_value=None, **kwargs):
        default_value = 2 if default_value is None else default_value
        delta = default_value if p_key not in kwargs.keys() else kwargs[p_key]
        return delta
