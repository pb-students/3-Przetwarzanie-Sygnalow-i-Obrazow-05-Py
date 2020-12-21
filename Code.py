#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from math import pi as PI
from math import sin
from scipy import signal
import IPython.display as ipd
import wave

DPI = 100

# In[2]:


def Plot(y0, y1, system, limits, title):
    fig, plots = plt.subplots(2, 2, figsize= (8,4), dpi= DPI)
    for a_y in range(2):
        for a_x in range(2):
            plot = plots[a_y][a_x]
            plot.grid(True)
            plot.set_xlabel("Nr Próbki"); plot.set_ylabel("Amplituda")
            plot.set_xlim(0, 31); plot.set_ylim(limits[a_y][0], limits[a_y][1])

    plots[0][0].plot(y0); plots[0][0].set_title("x1[n]")
    plots[0][1].plot(y1); plots[0][1].set_title("x2[n]")
    plots[1][0].plot(system(y0) + system(y1)); plots[1][0].set_title("S{x1[n]} + S{x2[n]}")
    plots[1][1].plot(system(y0 + y1)); plots[1][1].set_title("S{x1[n] + x2[n]}")

    plt.suptitle(f"System {title}", fontsize='xx-large')
    fig.set_tight_layout(tight=True)
    plt.show()

N = 32

x = np.arange(N)
y = []
y.append(np.sin(2*PI*x/N))
y.append(np.ones(N))

Plot(y[0], y[1], lambda x: 2*x, [[-1, 4],[-1, 4]], "A")
Plot(y[0], y[1], lambda x: x+1, [[-1, 4],[-1, 4]], "B")
Plot(y[0], y[1], lambda x: [x[i+1] - x[i] for i in range(0, len(x)-1)], [[-1, 4],[-0.3, 0.3]], "C")



# In[3]:


N = 64
k_range = [0, 16, 32]

x = np.arange(N)
y = []
y.append([np.sin(2*PI*x/N), "x1"])
y.append([np.sin(4*PI*x/N), "x2"])
y.append([np.arange(1), "h"])


# In[4]:


# A, B i C


for k in k_range:
    y[2][0] = signal.unit_impulse(N, k)

    fig, plots = plt.subplots(3, 3, figsize= (8,6), dpi= DPI)

    for i in range(3):
        if i == 2:
            plots[i][0].stem(y[i][0])
        else:
            plots[i][0].plot(y[i][0])
        plots[i][0].set_xlim(0, N-1)
        plots[i][0].set_title(y[i][1])

        plots[i][1].plot(signal.convolve(y[i-2][0], y[i-1][0]))
        plots[i][1].set_title(f"Splot liniowy {y[i-2][1]}*{y[i-1][1]}")
        plots[i][1].set_xlim(0, 2*N-1)

        plots[i][2].plot(signal.convolve(y[i-2][0], np.concatenate((y[i-1][0], y[i-1][0])) ))
        plots[i][2].set_title(f"Splot kołowy {y[i-2][1]}*{y[i-1][1]}")
        plots[i][2].set_xlim(0, 2*N-1)

        for ii in range(3):
            plots[i][ii].set_xlabel("Nr Próbki")
            plots[i][ii].set_ylabel("Amplituda")

    plt.suptitle(f"Porównanie splotu liniowego z kołowym dla k= {k}", fontsize='xx-large')
    fig.set_tight_layout(tight=True)
    plt.show()


# In[5]:


# D


y[2][0] = signal.unit_impulse(N, 32)

fig, plots = plt.subplots(3, 3, figsize= (8,6), dpi= DPI)
for i in range(3):
    if i == 2:
        plots[i][0].stem(y[i][0])
    else:
        plots[i][0].plot(y[i][0])
    plots[i][0].set_xlim(0, N-1)
    plots[i][0].set_title(y[i][1])

    plots[i][1].plot(signal.convolve(y[i-2][0], y[i-1][0]))
    plots[i][1].set_title(f"Splot liniowy {y[i-2][1]}*{y[i-1][1]}")
    plots[i][1].set_xlim(0, 2*N-1)

    plots[i][2].plot(signal.convolve(y[i-1][0], y[i-2][0]))
    plots[i][2].set_title(f"Splot kołowy {y[i-1][1]}*{y[i-2][1]}")
    plots[i][2].set_xlim(0, 2*N-1)

plt.suptitle("Test przemienności splotu (k=32)", fontsize='xx-large')
fig.set_tight_layout(tight=True)
plt.show()


# In[6]:


# E

y[2][0] = signal.unit_impulse(N, 32)

fig, plots = plt.subplots(3, 2, figsize= (8,6), dpi= DPI)
for i in range(3):
    if i == 2:
        plots[i][0].stem(y[i][0])
    else:
        plots[i][0].plot(y[i][0])
        plots[i][1].set_xlim(0, 2*N-1)
    plots[i][0].set_xlim(0, N-1)
    plots[i][0].set_title(y[i][1])

plots[0][1].plot(signal.convolve((y[0][0] + y[1][0]), y[2][0]))
plots[0][1].set_title("Splot sumy (x1+x2)*h")
plots[0][1].set_xlim(0, 2*N-1)

plots[1][1].plot((signal.convolve(y[0][0], y[2][0]) + signal.convolve(y[1][0], y[2][0])))
plots[1][1].set_title("Suma splotów x1*h + x2*h")
plots[1][1].set_xlim(0, 2*N-1)

fig.delaxes(plots[2][1])
plt.suptitle("Test liniowości splotu (k=32)", fontsize='xx-large')
fig.set_tight_layout(tight=True)
plt.show()

# In[7]:


N = 64

x = np.arange(N)
y = np.sin(2*PI*x/N)
Y = np.fft.fft(y)
h = np.exp(-x/10)
H = np.fft.fft(h)
G = Y*H

fig, plots = plt.subplots(3, 2, figsize= (8,6), dpi= DPI)

plots[0][0].set_title("Re(X=FFT(y))")
plots[0][0].stem(Y.real)
plots[0][0].set_ylim(-1, 1)
plots[0][1].set_title("Re(H=FFT(h))")
plots[0][1].stem(H.real)
plots[1][0].set_title("Re(G=Y*H)")
plots[1][0].stem(G.real)
plots[2][0].set_title("IFFT(G)")
plots[2][0].stem(np.fft.ifft(G))
plots[2][0].set_xlim(0, 2*N-1)
plots[2][1].set_title("Splot liniowy y*h")
plots[2][1].stem(signal.convolve(y, h))
plots[2][1].set_xlim(0, 2*N-1)

fig.delaxes(plots[1][1])
plt.suptitle("Porównanie mnożenia transformat fouriera sygnałów z\n ich splotem liniowym", fontsize='xx-large')
fig.set_tight_layout(tight=True)
plt.show()

# In[8]:


# Fragment wideo: https://www.youtube.com/watch?v=6ssGj65U2F8
data = wave.open("Saxophone.wav")
framerate = data.getframerate()
recording = np.frombuffer(data.readframes(-1), dtype = "int16")# / 2000
recording = recording / max(recording)

ir = [] # Impulse response.
# Sala koncertowa:
# http://legacy.spa.aalto.fi/projects/poririrs/
data = wave.open("s1_r1_b.wav")
ir.append(np.frombuffer(data.readframes(-1), dtype = "int16"))
ir[0] = ir[0] / max(ir[0])
# Długi tunel:
# https://www.voxengo.com/impulses/
data = wave.open("Large Long Echo Hall.wav")
ir.append(np.frombuffer(data.readframes(-1), dtype = "int16"))
ir[1] = ir[1] / max(ir[1])


plt.plot(recording, label= "Saksofon")
plt.plot(ir[0], label= "IR sali koncertowej")
plt.plot(ir[1], label= "IR silosa")
plt.legend(loc="upper right")

ipd.display(ipd.Audio(recording, rate=framerate*2))
print("Sala koncertowa:")
ipd.display(ipd.Audio(ir[0], rate=framerate*2))
ipd.display(ipd.Audio(signal.convolve(recording, ir[0]), rate=framerate*2))
print("Tunel:")
ipd.display(ipd.Audio(ir[1], rate=framerate*2))
ipd.display(ipd.Audio(signal.convolve(recording, ir[1]), rate=framerate*2))