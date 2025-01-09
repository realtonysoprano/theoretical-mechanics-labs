import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def Anidisk(i):
    A.set_data(X_A[i], Y_A[i])
    B.set_data(X_B[i], Y_B[i])
    Line_AB.set_data([X_A[i], X_B[i]], [Y_A[i], Y_B[i]])
    Drawed_Disk.set_data(X_C1 + X_Disk, Y_C1 + Y_Disk)


    thetta = np.linspace(0, Nv *5.2 - csi[i], 1001)
    X_SpilarSpring = -(R1 + thetta * (R2 - R1) / thetta[-1]) * np.sin(thetta)
    Y_SpilarSpring = (R1 + thetta * (R2 - R1) / thetta[-1]) * np.cos(thetta)
    Drawed_SpiralSpring.set_data(X_SpilarSpring+X_A[i], Y_SpilarSpring+Y_A[i])

    return [A, B, Line_AB, Drawed_Disk]

Steps = 1001
t = np.linspace(0, 10, Steps)

csi = (np.sin(5 * t))
phi = t

WheelR = 4
l = 4


X_A = 4 - 2*np.sin(-phi)
Y_A = 4 + 2*np.cos(-phi)
X_B = X_A - l*np.sin(csi)
Y_B = Y_A - l*np.cos(csi)


psi = np.linspace(0, 7, 1001)
X_Disk = WheelR*np.sin(psi)
Y_Disk = WheelR*np.cos(psi)

X_C1 = 4
Y_C1 = 4


Nv = 4
R1 = 0.2
R2 = 0.7
thetta = np.linspace(0, Nv*7-phi[0], 1001)
X_SpilarSpring =-(R1+thetta*(R2-R1)/thetta[-1])*np.sin(thetta)
Y_SpilarSpring = (R1+thetta*(R2-R1)/thetta[-1])*np.cos(thetta)


fig = plt.figure(figsize = [13, 7]) #создаем фигуру
ax = fig.add_subplot(1, 1, 1)
ax.axis('equal')
ax.set(xlim=[-2, 10], ylim=[-2, 10])


Line_AB = ax.plot([X_A[0], X_B[0]], [Y_A[0],Y_B[0]])[0]  #рисуем стержень AB

A = ax.plot(X_A[0], Y_A[0], marker = 'o')[0]
B = ax.plot(X_B[0], Y_B[0], marker = 'o', markersize = 30)[0]

Drawed_SpiralSpring = ax.plot(X_SpilarSpring + X_A[0], Y_SpilarSpring+ Y_A[0])[0]
Drawed_Disk = ax.plot(X_C1 + X_Disk, Y_C1 + Y_Disk,color=[0, 0.5, 0])[0]
nechto = FuncAnimation(fig, Anidisk, frames = Steps, interval = 20)

plt.show()