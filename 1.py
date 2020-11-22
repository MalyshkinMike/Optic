import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

i = complex(0, 1)
core = lambda alpha, ksi, x: exp(-alpha * abs(i * x + ksi) ** 2)
f = lambda x, b: exp(i * b * x)

def ish(func, b, x):
    y = func(x, b)
    plt.figure(1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend("exp(i*b*x) ampl")
    plt.grid()
    plt.title("Амплитуда")
    plt.plot(x, abs(y))
    y1 = np.angle(func(x, b))
    plt.figure(2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend("exp(i*b*x) ampl")
    plt.grid()
    plt.title("Фаза")
    plt.plot(x, y1)

def res(F, args):
    plt.figure(3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend("F(ksi)")
    plt.title("Амплитуда")
    plt.grid()
    plt.plot(args, abs(F))
    plt.figure(4)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend("F(ksi)")
    plt.title("Фаза")
    plt.grid()
    plt.plot(args, np.angle(F))

if __name__ == '__main__':
    alpha = 1
    b = 1
    p, q = -1, 1
    m, n = 1000, 1000
    hx = (q - p)/n
    x = np.linspace(p, q, n)
    ksi = np.linspace(p, q, m)
    ish(f, b, x)

    '''
    F = []
    for j in range(m):
        F_temp = 0
        for k in range(n - 1):
            F_temp += core(alpha, ksi[j], x[k]) * f(x[k], b) * hx
        F.append(F_temp)
    '''
    x_i, ksi_i = np.meshgrid(x, ksi)
    A = core(alpha, x_i, ksi_i)
    fs = f(x, b)
    F = A.dot(fs)*hx
    res(np.array(F), ksi)
    plt.show()