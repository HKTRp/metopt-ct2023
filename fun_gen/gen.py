import numpy as np
from matplotlib import pyplot as plt
from grad_desc import grad


class MyFunction:

    def __init__(self, diag_matrix):
        self.coeffs = np.array([diag_matrix[i][i] for i in range(len(diag_matrix))])

    def __call__(self, args):
        return sum([args[i] * args[i] * self.coeffs[i] for i in range(len(args))])

    def __str__(self):
        result = ''
        for i in range(len(self.coeffs) - 1):
            result += f'{self.coeffs[i]}*{chr(ord("x") + i)}^2 + '
        result += f'{self.coeffs[-1]}*{chr(ord("x") + len(self.coeffs) - 1)}^2'
        return result


class QuadraticFunctionGenerator:

    @staticmethod
    # n - number of arguments, k - condition number
    def generate(n, k):
        diag_matrix = np.zeros((n, n))
        diag_matrix[0][0] = 1
        diag_matrix[1][1] = k
        for i in range(2, n):
            diag_matrix[i][i] = np.random.uniform(1, k)

        return MyFunction(diag_matrix)


# xdata = []
# ydata = []
# zdata = []

with open("output.txt", "w") as output:
    for n in range(2, 1001):
        for k in range(1, 1001):
            F = QuadraticFunctionGenerator().generate(n, k)
            points, isValid = grad.DichtGradientDescending().find_min(
                F, initial=np.array([np.random.uniform(-10, 10) for _ in range(n)],))
            if isValid:
                output.write(f'{n} {k} {points.size // n}\n')

# ax = plt.axes(projection='3d')
# ax.set_xlabel('N')
# ax.set_ylabel('K')
# ax.set_zlabel('Iterations')
# ax.scatter3D(xdata, ydata, zdata)
