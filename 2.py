import numpy as np
from scipy.fft import fft
from matplotlib import pyplot as plt

def swap_in_half(arr):
    half = int(len(arr) / 2)
    return np.hstack([arr[half:len(arr)], arr[0:half]])

def finite_fft(f, a, N, M):
    h_x = 2 * a / N
    zeroes_count = int((M - N)/2)
    zeroes = np.array([0] * zeroes_count)
    f = swap_in_half(np.hstack([zeroes, f, zeroes]))
    F = swap_in_half(fft(f) * h_x)
    temp1 = zeroes_count - 1
    temp2 = temp1 + N
    F1 = F[temp1:temp2]
    b = N**2/(4*a*M)
    return (F1, b)
def fft2(u, N, a, funct, M):
    def rect_method(u, n, a, fun):
        x = np.linspace(-a, a, n)
        def func(x):
            return fun(x)*np.exp(-2*np.pi*complex(0,1)*u*x)
        h = 2*a/n
        res = 0
        for i in range(0, n-1):
            res += func(x[i])
        return res*h
    f = funct(np.linspace(-a, a, N))
    h_x = 2 * a / N
    zeroes_count = int((M - N) / 2)
    zeroes = np.array([0] * zeroes_count)
    f = swap_in_half(np.hstack([zeroes, f, zeroes]))
    F = swap_in_half(rect_method(u, M, a, funct) * h_x)
    temp1 = zeroes_count - 1
    temp2 = temp1 + N
    F1 = F[temp1:temp2]
    b = N ** 2 / (4 * a * M)
    return (F1, b)


s = float(input())
gauss_beam = lambda x: np.exp(-s*x**2)
N = 512
M = 4096
a = 5
x = np.linspace(-a, a, N)

y = gauss_beam(x)
plt.figure(1)
plt.plot(x, abs(y))
res = finite_fft(y, a, N, M)
y1 = res[0]
x1 = np.linspace(-res[1], res[1], N)
plt.figure(2)
plt.plot(x1, abs(y1))
u = np.array([0] * M)
res1 = fft2(u, N, a, gauss_beam, M)
x_1 = np.linspace(-res[1], res[1], N)
y_1 = res[0]
plt.plot(x_1, abs(y_1))
plt.show()