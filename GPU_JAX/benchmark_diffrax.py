#!/usr/bin/env python
# coding: utf-8

# In[55]:


import time

import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.numpy as jnp
import numpy as np
import os
import timeit
import sys

numberOfParameters = int(sys.argv[1])

# In[2]:


# from matplotlib import pyplot as  plt


# In[3]:


from jax.lib import xla_bridge
print("Working on :", xla_bridge.get_backend().platform)


# In[4]:


class Lorenz(eqx.Module):
    k1: float

    def __call__(self, t, y, args):
        f0 = 10.0*(y[1] - y[0])
        f1 = self.k1 * y[0] - y[1] - y[0] * y[2]
        f2 = y[0] * y[1] - (8/3)*y[2]
        return jnp.stack([f0, f1, f2])


# In[5]:


@jax.jit
def main(k1):
    lorenz = Lorenz(k1)
    terms = diffrax.ODETerm(lorenz)
    t0 = 0.0
    t1 = 1.0
    y0 = jnp.array([1.0, 0.0, 0.0])
    dt0 = 0.001
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts = jnp.array([t0,t1]))
    stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-3)
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0,
        t1,
        dt0,
        y0,
#         saveat=saveat,
        #stepsize_controller=stepsize_controller,
    )
    return sol


# In[11]:

main(28.0)

start = time.time()
sol = main(28.0)
end = time.time()

print("Results:")
for ti, yi in zip(sol.ts, sol.ys):
    print(f"t={ti.item()}, y={yi.tolist()}")
print(f"Took {sol.stats['num_steps']} steps in {end - start} seconds.")


# In[12]:


parameterList = jnp.linspace(0.0,21.0,numberOfParameters)


# In[13]:


parameterList


# In[17]:


res = timeit.repeat(lambda: jax.vmap(main)(parameterList),repeat = 100,number = 1)

best_time  = min(res)*1000
print(best_time)


# In[ ]:


file = open("./data/Jax_times_unadaptive.txt","a+")
file.write('{0} {1}\n'.format(numberOfParameters, best_time))
file.close()


# In[18]:


@jax.jit
def main(k1):
    lorenz = Lorenz(k1)
    terms = diffrax.ODETerm(lorenz)
    t0 = 0.0
    t1 = 1.0
    y0 = jnp.array([1.0, 0.0, 0.0])
    dt0 = 0.001
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts = jnp.array([t0,t1]))
    stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0,
        t1,
        dt0,
        y0,
#         saveat=saveat,
        stepsize_controller=stepsize_controller,
    )
    return sol


# In[33]:


import timeit


# In[50]:


res = timeit.repeat(lambda: jax.vmap(main)(parameterList),repeat = 100,number = 1)


# In[52]:

best_time  = min(res)*1000
print(best_time)


# In[59]:


file = open("./data/Jax_times_adaptive.txt","a+")
file.write('{0} {1}\n'.format(numberOfParameters, best_time))
file.close()

