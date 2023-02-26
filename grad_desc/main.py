import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import rosen


class GradientDescending():

    @staticmethod
    def grad(f, x, h=1e-5):
        return (f(x[:, np.newaxis] + h * np.eye(2)) - f(x[:, np.newaxis] - h * np.eye(2))) / (2 * h)

    @staticmethod
    def directed_derivative(f, x, direction, h=1e-5):
        step = direction * h
        return (f(x + step) - f(x - step)) / 2

    def one_dimension_method(self, func, x, direction, alpha, eps, max_iterations):
        return x + direction * alpha

    def find_min(self, func, initial, alpha=0.5, eps=0.001, max_iterations=1000):
        coords = initial
        for i in range(max_iterations):
            print(coords)
            direction = -self.grad(func, coords)
            next_coords = self.one_dimension_method(func, coords, direction, alpha, eps, max_iterations)
            delta = next_coords - coords
            if np.sqrt(delta.dot(delta)) < eps:
                return next_coords
            coords = next_coords
        return coords


class DichtGradientDescending(GradientDescending):

    def one_dimension_method(self, func, x, direction, alpha, eps, max_iterations):
        current = x
        next = x + direction*alpha
        for i in range(max_iterations):
            if np.linalg.norm(func(current) - func(next)) < eps:
                return current
            middle = (current + next) / 2
            derive = self.directed_derivative(f, middle, direction)
            if derive < 0:
                current = middle
            else:
                next = middle
        return current


def f(x):
    return np.sin(0.5 * x[0]**2 - 0.25 * x[1]**2 + 3)*np.cos(2*x[0]+1-np.exp(x[1]))


if __name__ == "__main__":
    alphas = [1, 0.5, 0.1, 2]
    for alpha in alphas:
        print(DichtGradientDescending().find_min(rosen, initial=np.array([1, 1]), alpha=alpha), "dicht")
        print(GradientDescending().find_min(rosen, initial=np.array([1, 1]), alpha=alpha), "simple")
