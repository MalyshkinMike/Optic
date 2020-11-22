""" Optical informatic lab 1"""

import numpy as np
import cmath
import matplotlib
import matplotlib.pylab as pylab


# variant 11
# K(ksi,x) = exp( -x**2 * ksi ** 2) * H5(alpha * x * ksi)
# f(x) = exp(i * beta * x)

def get_f_val(a: float = -1, b: float = -1, beta: (float, complex) = 1., n: int = 1000, x=None):
    """
    return: array of values and step
    """
    if x is None:
        x = np.linspace(a, b, n)
    f_val = np.exp(complex(0, 1) * beta * x)
    return f_val


def get_core_val(p: float = -1, q: float = 1, a: float = -1, b: float = 1, alpha: (float, complex) = 1., n: int = 1000,
                 m: int = 1000, x=None, ksi=None):
    if ksi is None:
        ksi = np.linspace(p, q, m)
    if x is None:
        x = np.linspace(a, b, n)
    xx, ksi_ksi = np.meshgrid(x, ksi)
    # Hermite polynomial 5
    c = (0, 0, 0, 0, 1)
    # np.polynomial.hermite.hermval(alpha * xx * ksi_ksi, c)
    core_val = np.exp(-xx ** 2 * ksi_ksi ** 2) * (32 * x ** 5 - 160 * x ** 3 + 120 * x)
    return core_val


def main():
    console_output = True
    n = 2000
    m = 3000
    alpha = 1
    beta = np.pi * 2
    a, b = -1, 1
    p, q = -2, -0.5

    x = np.linspace(a, b, n)
    ksi = np.linspace(p, q, m)
    # const step
    h_x = x[1] - x[0]

    f_val = get_f_val(a, b, beta, n, x)
    core_val = get_core_val(p, q, a, b, alpha, n, m, x, ksi)
    # left triangle method integrating
    temp_val = f_val[len(f_val) - 1]
    f_val[len(f_val) - 1] = 0
    F_val = np.dot(np.matmul(core_val, f_val), h_x)
    f_val[len(f_val) - 1] = temp_val

    f_val_afin = (np.absolute(f_val), np.angle(f_val, deg=False))
    F_val_afin = (np.absolute(F_val), np.angle(F_val, deg=False))

    if console_output:
        print(f"f_val: \n{f_val}\n")
        print(f"core_val: \n{core_val}\n")
        print(f"F_val: \n{F_val}")
        print(f"f afin:\n {[(f_val_afin[0][i], f_val_afin[1][i]) for i in range(len(x))]}\n")
        # print(f"F afin:\n {[(F_val_afin[0][i], F_val_afin[1][i]) for i in range(len(x))]}\n")

    # visualise

    # input signal
    pylab.figure(1)

    ax1 = pylab.subplot(211)
    ax1.set_title(f"Amplitude of input signal, n={n},m={m}")
    pylab.xlabel("x")
    pylab.ylabel("Amplitude")
    # amplitude
    pylab.plot(x, f_val_afin[0])

    ax2 = pylab.subplot(212)
    ax2.set_title("Angle of input signal," + f"beta= {beta}")
    pylab.xlabel("x")
    pylab.ylabel("Angle")
    # angle
    pylab.plot(x, f_val_afin[1])

    # conversion function
    pylab.figure(2)
    ax1 = pylab.subplot(211)
    pylab.xlabel("ξ")
    pylab.ylabel("Amplitude")
    ax1.set_title("Amplitude of conversion result," + f" alpha= {alpha},n={n},m={m}")
    pylab.plot(ksi, F_val_afin[0])

    ax2 = pylab.subplot(212)
    ax2.set_title("Angle of conversion result," + f" beta= {beta}")
    pylab.xlabel("ξ")
    pylab.ylabel("Angle")
    # angle
    pylab.plot(ksi, F_val_afin[1])

    pylab.show()


if __name__ == '__main__':
    main()