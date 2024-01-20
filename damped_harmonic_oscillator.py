# Code to calculate and plot the transient response of a damped, harmonic oscillator

# import the  necessary libraries and rename them
import numpy as np
import array
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image


# Define the Parameter Names, and give them numerical values

param_names = ["amplitude", "tau", "resonant-freq", "phase"]
guesses = (1, 0.002, 500, 0)

# Define the Function for the Harmonic Oscillator Transient

def fit_function(x, amplitude, tau, resonantf, phase):
    return amplitude * np.exp(-x/tau) * np.cos(2.0*np.pi * resonantf * x + phase)


# Define a set of x values that will be used for the calculation
# Note that in your fitting code, x is defined differently 
#  - do not change that part of the fitting code when you get there.

x = np.arange(0.0, 0.02, 0.0001)

y_model = fit_function(x, *guesses)

plt.plot(x,y_model)
plt.xlabel("time")
plt.ylabel("Voltage")
plt.title("Damped Oscillator")

# save and plot image 
plt.savefig("DampedOscillator1.jpeg")
plt.show()
