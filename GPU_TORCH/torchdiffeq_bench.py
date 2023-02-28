#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import sys
import os
import timeit
import sys

numberOfParameters = int(sys.argv[1])
# numberOfParameters = 10

# In[7]:


import torchdiffeq
import math
import torch.nn as nn
import timeit
from torchdiffeq import odeint


# In[2]:


torch.has_cuda


# In[3]:


#smoke test
torch.rand(10).cuda()

# In[64]:


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


# In[65]:


torch.tensor(0.0210)


# In[69]:


u0 = torch.tensor([1.0,0.0,0.0]).cuda()
t = torch.linspace(0, 1.0, 1001).cuda()
y = odeint(LorenzODE(), u0, t, method='rk4',options=dict(step_size=0.001))


# In[70]:


# In[71]:


print(y.device)


# In[72]:


tmp = LorenzODE(rho = torch.tensor(28.0))

def time_func():
    with torch.no_grad():
        odeint(LorenzODE(), u0, t, method='rk4', options=dict(step_size=0.001))

time_func()
timeit.Timer(time_func).timeit(number=2)/2 # 8.595667100000014 seconds


# In[81]:


def solve(p):
    with torch.no_grad():
        t = torch.linspace(0, 1.0, 2).cuda()
        traj = odeint(LorenzODE(rho = p), u0, t, method='rk4', options=dict(step_size=0.001))
        return traj

def other_func(X):
    mapped_traj = vmap(solve)(X)


# In[82]:


parameters = torch.linspace(0.0,21.0,numberOfParameters).cuda()
print(len(parameters))
# vmap()


# In[83]:


solve(parameters[-1])


# In[ ]:


import timeit
res = timeit.repeat(lambda: torch.vmap(solve)(parameters), repeat = 10, number = 1)


# In[ ]:


best_time  = min(res)*1000
print(best_time)


# In[ ]:


file = open("./data/Torch_times_unadaptive.txt","a+")
file.write('{0} {1}\n'.format(numberOfParameters, best_time))
file.close()


# In[ ]:


# def solve(p):
#     with torch.no_grad():
#         t = torch.linspace(0, 1.0, 2).cuda()
#         traj = odeint(LorenzODE(rho = p), u0, t, method='dopri5', options=dict(first_step = 0.001), rtol = 1e-8, atol = 1e-8)
#         return traj

# def other_func(X):
#     mapped_traj = vmap(solve)(X)


# # In[82]:


# parameters = torch.linspace(0.0,21.0,numberOfParameters).cuda()
# print(len(parameters))
# # vmap()


# # In[83]:


# solve(parameters[-1])


# # In[ ]:


# import timeit
# res = timeit.repeat(lambda: torch.vmap(solve)(parameters), repeat = 10, number = 1)


# # In[ ]:


# best_time  = min(res)*1000
# print(best_time)


# # In[ ]:


# file = open("./data/Torch_times_adaptive.txt","a+")
# file.write('{0} {1}\n'.format(numberOfParameters, best_time))
# file.close()


# # In[ ]:



