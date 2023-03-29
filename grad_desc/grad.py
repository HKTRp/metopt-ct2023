import numpy as np
from matplotlib import pyplot as plt


class QuadraticFunction:

    def __init__(self, diag_matrix: np.array):
        self.coeffs = np.array([diag_matrix[i][i]
                               for i in range(len(diag_matrix))], dtype=np.double)

    def __call__(self, args: np.array) -> float:
        return sum([np.power(args[i], 2) * self.coeffs[i] for i in range(len(args))])

    def __str__(self):
        result = ''
        for i in range(len(self.coeffs) - 1):
            result += f'{self.coeffs[i]:.5f}*x_{i}^2 + '
        result += f'{self.coeffs[-1]:.5f}*x_{len(self.coeffs) - 1}^2'
        return result


class GradientDescending:

    @staticmethod
    def grad(f: QuadraticFunction, x: np.array, h=1e-5) -> np.array:
        return (f(x[:, np.newaxis] + h * np.eye(x.size)) -
                f(x[:, np.newaxis] - h * np.eye(x.size))) / (2 * h)

    @staticmethod
    def directed_derivative(f: QuadraticFunction, x: np.array, direction: np.array, h=1e-5) -> float:
        step = direction * h
        return (f(x + step) - f(x - step)) / 2

    def one_dimension_method(self,
                             func: QuadraticFunction,
                             x: np.array,
                             direction: np.array,
                             learning_rate: float,
                             eps: float,
                             max_iterations: int) -> np.array:
        return x + direction * learning_rate

    def find_min(self,
                 func: QuadraticFunction,
                 initial: np.array,
                 learning_rate: float = 0.5,
                 eps: float = 0.001,
                 max_iterations: int = 10000) -> tuple[np.array, bool]:
        points = initial
        coords = initial
        for i in range(max_iterations):
            direction = -self.grad(func, coords)
            next_coords = self.one_dimension_method(func, coords,
                                                    direction, learning_rate, eps, max_iterations)
            delta = next_coords - coords
            if np.sqrt(delta.dot(delta)) < eps:
                return points, True
            coords = next_coords
            points = np.vstack((points, coords))
        return points, False


class DichtGradientDescending(GradientDescending):

    def one_dimension_method(self, 
                             func: QuadraticFunction,
                             x: np.array,
                             direction: np.array,
                             learning_rate: float,
                             eps: float,
                             max_iterations: int) -> np.array:
        current = x
        next = x + direction * learning_rate
        for i in range(max_iterations):
            if np.linalg.norm(func(current) - func(next)) < eps:
                return current
            middle = (current + next) / 2
            derive = self.directed_derivative(func, middle, direction)
            if derive < 0:
                current = middle
            else:
                next = middle
        return current


def generate(n: int, k: int) -> QuadraticFunction:
    diag_matrix = np.zeros((n, n))
    diag_matrix[0][0] = 1
    diag_matrix[1][1] = k
    for i in range(2, n):
        diag_matrix[i][i] = np.random.uniform(1, k)

    return QuadraticFunction(diag_matrix)

# fig = plt.figure()
ax = plt.figure().add_subplot(projection='3d')
bx = plt.figure().add_subplot(projection='3d')
cx = plt.figure().add_subplot(projection='3d')
ax.set_xlabel('N')
ax.set_ylabel('K')
ax.set_zlabel('Iterations')
bx.set_xlabel('N')
bx.set_ylabel('K')
bx.set_zlabel('Iterations')
bx.view_init(elev=0, azim=-90, roll=0)
cx.set_xlabel('N')
cx.set_ylabel('K')
cx.set_zlabel('Iterations')
cx.view_init(elev=0, azim=0, roll=0)
# with open(".\\output.txt") as out:
#     for line in out:
#         n, k, iter_count = map(int, line.split())
#         ax.scatter(n, k, iter_count)
#         bx.scatter(n, k, iter_count)
#         cx.scatter(n, k, iter_count)
# plt.show()