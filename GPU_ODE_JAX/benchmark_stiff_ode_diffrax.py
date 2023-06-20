#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import time

import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.numpy as jnp
import numpy as np
import os
import sys


# %%

numberOfParameters = int(sys.argv[1])

# %%


from jax.lib import xla_bridge
print("Working on :", xla_bridge.get_backend().platform)


# %%


jax.config.update("jax_enable_x64", True)


# %%


y0 = jnp.array([1.0, 0.0, 0.0])


# %%


class Robertson(eqx.Module):
    k1: float
    k2: float
    k3: float

    def __call__(self, t, y, args):
        f0 = -self.k1 * y[0] + self.k3 * y[1] * y[2]
        f1 = self.k1 * y[0] - self.k2 * y[1] ** 2 - self.k3 * y[1] * y[2]
        f2 = self.k2 * y[1] ** 2
        return jnp.stack([f0, f1, f2])


# %%


@jax.jit
def main(k3):
    robertson = Robertson(0.04, 3e7, k3)
    terms = diffrax.ODETerm(robertson)
    t0 = 0.0
    t1 = 1e5
    y0 = jnp.array([1.0, 0.0, 0.0])
    dt0 = 0.001
    solver = diffrax.Kvaerno5()
    saveat = diffrax.SaveAt(ts = jnp.array([t0,t1]))
    stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0,
        t1,
        dt0,
        y0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=None
    )
    return sol


# %%


# main(1e4)

# start = time.time()
# sol = main(1e4)
# end = time.time()

# print("Results:")
# for ti, yi in zip(sol.ts, sol.ys):
#     print(f"t={ti.item()}, y={yi.tolist()}")
# print(f"Took {sol.stats['num_steps']} steps in {end - start} seconds.")


# %%

# print(numberOfParameters)

# numberOfParameters = 2097152

parameterList = jnp.linspace(10.0,1e4,numberOfParameters)


# %%


# a = jax.numpy.array([[ 1.01290589e-03,  2.75272126e-05, -2.69166597e-04,
#         -5.58780779e-06],
#        [ 2.75272126e-05,  1.34740128e-03, -4.34192721e-06,
#         -3.00849575e-04],
#        [-2.69166597e-04, -4.34192721e-06,  1.28766222e-04,
#          7.41944929e-07],
#        [-5.58780779e-06, -3.00849575e-04,  7.41944929e-07,
#          7.99537441e-05]])

# jax.numpy.linalg.cholesky(a)   # NOk


# %%


import timeit


# %%


# # %timeit jax.vmap(main)(parameterList)


# %%


res = timeit.repeat(lambda: jax.vmap(main)(parameterList),repeat = 10,number = 1)


# %%


best_time = min(res)*1000


# %%

print("{:} ODE solves with adaptive time-stepping completed in {:.1f} ms".format(numberOfParameters, best_time))



# %%


file = open("./data/JAX/stiff/Jax_times_adaptive.txt","a+")
file.write('{0} {1}\n'.format(numberOfParameters, best_time))
file.close()


# %%




