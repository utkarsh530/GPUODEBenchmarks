import torch
torch.cuda.is_available()
torch.cuda.get_device_name(0)
import torchode as to
from functorch import vmap

def f(t, y, k):
    x, y, z = y[..., 0] , y[..., 1], y[..., 2]
    return torch.stack((10*(y-x), k * x - y - x * z, x * y - (8/3)*z), dim = -1)

y0 = torch.tensor([[1.0, 0.0, 0.0]])
n_steps = 1000
t_eval = torch.stack((torch.linspace(0, 1, n_steps),))
term = to.ODETerm(f, with_args = True)
step_method = to.Tsit5(term = term)
controller = to.FixedStepController()
dt0 = torch.tensor([0.001])

solver = to.AutoDiffAdjoint(step_method, controller)

jit_solver = torch.jit.script(solver)

numberOfParameters = 768000
parameterList = torch.linspace(0.0,21.0,numberOfParameters)
sol = solver.solve(to.InitialValueProblem(y0 = y0, t_eval = t_eval), args = parameterList[-1], dt0 = dt0)
psol = lambda p : solver.solve(to.InitialValueProblem(y0 = y0, t_eval = t_eval), args = p, dt0 = dt0)
vmap(psol)(parameterList) ## Faies

# def f(t, y, k):
#     return torch.tensor([[10*(y[1]-y[0])],[k * y[0] - y[1] - y[0] * y[2]],[y[0] * y[1] - (8/3)*y[2]]])

# y0 = torch.tensor([[1.0],[0.0], [0.0]])
# n_steps = 1000
# t_eval = (torch.linspace(0, 1, n_steps))
# term = ODETerm(f, with_args = True)
# controller = FixedStepController()
# dt0 = torch.tensor([0.001])
# numberOfParameters = 768000
# parameterList = torch.linspace(0.0,21.0,numberOfParameters)
# sol = solve_ivp(term, y0, t_eval, args = parameterList[-1], controller = controller,dt0 = dt0)

# psol = lambda p : solve_ivp(term, y0, t_eval, args = p, controller = controller,dt0 = dt0)

# vmap(psol)(parameterList)
# print(sol.stats)