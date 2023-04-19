#!/usr/bin/env python
# coding: utf-8
# %%
# Benchmarking torchdiffeq ODE solvers for ensemble problems, via vmap. The Lorenz ODE is integrated by Tsit5.

# Created By: Utkarsh
# Last Updated: 19 April 2023

# %%

import torch
import sys
import os
import timeit
import sys

numberOfParameters = int(sys.argv[1])

# %%


import torchdiffeq
import math
import torch.nn as nn
import timeit
from torchdiffeq import odeint


# %%

## Checking if torch installation has cuda enabled
print("CUDA enabled: ", torch.has_cuda)


# %%
# Defining the Lorenz ODE problem
class LorenzODE(torch.nn.Module):

    def __init__(self, rho = torch.tensor(21.0)):
        super(LorenzODE, self).__init__()
        self.sigma = nn.Parameter(torch.as_tensor([10.0]))
        self.rho = nn.Parameter(rho)
        self.beta = nn.Parameter(torch.as_tensor([8/3]))

    def forward(self, t, u):
        x, y, z = u[0],u[1],u[2]
        du1 = self.sigma[0] * (y - x)
        du2 = x * (self.rho - z) - y
        du3 = x * y - self.beta[0] * z
        return torch.stack([du1, du2, du3])


# %%
# Uncomment for smoke test

# u0 = torch.tensor([1.0,0.0,0.0]).cuda()
# t = torch.linspace(0, 1.0, 1001).cuda()
# y = odeint(LorenzODE(), u0, t, method='rk4',options=dict(step_size=0.001))


# %%
# Define the solve without gradient calculations
# Note: I was't able to JIT compile the code with this application, torchdiffeq + vmap
def solve(p):
    with torch.no_grad():
        traj = odeint(LorenzODE(rho = p), u0, t, method='rk4', options=dict(step_size=0.001))
        return traj

# Define the initial conditions and timepoints to save
u0 = torch.tensor([1.0,0.0,0.0]).cuda()
t = torch.linspace(0, 1.0, 2).cuda()


# %%
# Generate parameter list
parameters = torch.linspace(0.0,21.0,numberOfParameters).cuda()


# %%

import timeit
res = timeit.repeat(lambda: torch.vmap(solve)(parameters), repeat = 10, number = 1)


# %%
# Print the best result

best_time  = min(res)*1000
print("{:} ODE solves with fixed time-stepping completed in {:.1f} ms".format(numberOfParameters, best_time))


# %%
# Save the result

file = open("./data/PYTORCH/Torch_times_unadaptive.txt","a+")
file.write('{0} {1}\n'.format(numberOfParameters, best_time))
file.close()


# %%
