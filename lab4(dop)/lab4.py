import datetime
import torch

from scipy import optimize
import numpy as np


def grad(_f, x, h=1e-5):
    return (_f(x[:, np.newaxis] + h * np.eye(x.size)) - _f(x[:, np.newaxis] - h * np.eye(x.size))) / (2 * h)


def f_test(x):
    return 2 * (x[0] - 10) ** 2 + 0.2 * (x[1] + 13) ** 2 + 3


def f_squares(x):
    return np.array([np.sqrt(2) * (x[0] - 10), np.sqrt(0.2) * (x[1] + 13)])


def f_grad(x):
    return [4 * x[0] - 40, 0.4 * x[1] + 5.2]


def f_hess(x):
    return np.array([[2, 0], [0, 0.2]])


def g_test(x):
    return (x[0] - 4) ** 2 + 100 * (x[1] + 5) ** 2


def h_test(x):
    return 4 * (x[0] - 1) ** 2 + 8 * (x[1] + 3) ** 2 - 4


ranged_arr = np.array([i + 1 for i in range(100)])


def many_vars(x):
    return ranged_arr.dot((x - ranged_arr) ** 2)


def many_vars_squares(x):
    return np.sqrt(ranged_arr) * (x - ranged_arr)


mvh = np.array([[0 if i != j else 2 * i for j in range(500)] for i in range(500)])


def many_vars_hess(x):
    return mvh


initial = np.ones(100)

methods = ["BFGS", "L-BFGS-B", "trust-krylov"]
functions = [(f_test, torch.ones(2), f_hess),
             (many_vars, torch.ones(500), many_vars_hess),
             (optimize.rosen, torch.zeros(2), optimize.rosen_hess)]


def fun_rosenbrock(x):
    return np.array([10 * (x[1] - x[0] ** 2), (1 - x[0])])


functions_squares = [(f_squares, torch.ones(2), f_hess),
                     (many_vars_squares, torch.ones(100), many_vars_hess),
                     (fun_rosenbrock, torch.zeros(2), optimize.rosen_hess)]


def hess(fuc, x):
    grd = grad(fuc, x)
    res = np.outer(grd, grd)
    return res


"""for met in methods:
    for func in functions:
        start = datetime.datetime.now()
        result = optimize.minimize(func[0], func[1], method=met, jac=lambda x: grad(func[0], x),
                                   hess=func[2])
        print(met, result["nit"], (datetime.datetime.now() - start).total_seconds())
"""

for func in functions_squares:
    start = datetime.datetime.now()
    result = optimize.least_squares(func[0], func[1], method="dogbox", max_nfev=1000)
    print("dogbox", result, (datetime.datetime.now() - start).total_seconds())

"""for area in range(1, 10):
    shift = area*0.1
    func = functions[2]
    start = datetime.datetime.now()
    result = optimize.minimize(func[0], func[1], method="TNC", jac=lambda x: grad(func[0], x),
                               hess=func[2], bounds=[(5, None), (5, None)])
    print("TNC with area", result["nit"], result['x'], (datetime.datetime.now() - start).total_seconds())
"""

"""f_zeros = torch.tensor([10, -13])
coefs = torch.tensor([2, 0.2])
x_value = torch.tensor([1.0, 1.0], requires_grad=True)
torch_f = coefs.dot(((x_value - f_zeros) ** 2))
start = datetime.datetime.now()
torch_f.backward()
print(x_value.grad, (datetime.datetime.now() - start).total_seconds())
start = datetime.datetime.now()
x_value = x_value.detach().numpy()
print(grad(f_test, x_value), (datetime.datetime.now() - start).total_seconds())
start = datetime.datetime.now()
print(f_grad(x_value), (datetime.datetime.now() - start).total_seconds())

ranged = torch.tensor([i + 1.0 for i in range(1000)])
x_value = torch.tensor([1.0 for i in range(1000)], requires_grad=True)
func = ranged.dot((x_value - ranged) ** 2)
start = datetime.datetime.now()
func.backward()
print((datetime.datetime.now() - start).total_seconds())
start = datetime.datetime.now()
x_value = x_value.detach().numpy()
grad(many_vars, x_value)
print((datetime.datetime.now() - start).total_seconds())
start = datetime.datetime.now()
an_grad = 2*ranged_arr*x_value - 2*(ranged_arr**2)
print((datetime.datetime.now() - start).total_seconds()) """
