import numpy as np
from qokit.portfolio_optimization import get_problem
from qokit.portfolio_optimization import portfolio_brute_force, get_sk_ini
from qokit.qaoa_objective_portfolio import get_qaoa_portfolio_objective
from scipy.optimize import minimize
import nlopt
import sys

def minimize_nlopt(f, x0, rhobeg=None, p=None, cd=False):
    def nlopt_wrapper(x, grad):
        if grad.size > 0:
            sys.exit("Shouldn't be calling a gradient!")
        return f(x).real
    num_parameters = 2*p
    if cd:
        num_parameters = 3*p

    opt = nlopt.opt(nlopt.LN_BOBYQA, num_parameters)
    opt.set_min_objective(nlopt_wrapper)

    opt.set_xtol_rel(1e-8)
    opt.set_ftol_rel(1e-8)
    opt.set_initial_step(rhobeg)
    xstar = opt.optimize(x0)
    minf = opt.last_optimum_value()

    return xstar, minf



po_problem = get_problem(N=6,K=3,q=0.5,seed=1,pre=1)
means_in_spins = np.array([po_problem['means'][i] - po_problem['q'] * np.sum(po_problem['cov'][i, :]) for i in range(len(po_problem['means']))])
scale = 1 / (np.sqrt(np.mean((( po_problem['q']*po_problem['cov'])**2).flatten()))+np.sqrt(np.mean((means_in_spins**2).flatten())))

po_problem = get_problem(N=6,K=3,q=0.5,seed=2,pre=scale)

p = 2
cd = 1
#qaoa_obj = get_qaoa_portfolio_objective(po_problem=po_problem,p=p,ini='dicke',mixer='trotter_ring',T=1,simulator='python')
qaoa_obj_qiskit = get_qaoa_portfolio_objective(po_problem=po_problem,p=p,ini='dicke',mixer='trotter_ring',T=1,simulator='qiskit',cd=cd)

best_portfolio = portfolio_brute_force(po_problem,return_bitstring=False)

#x0 = get_sk_ini(p=p) 
for _ in range(20):
    x0 = 2*np.pi*np.random.rand(2*p)
    if cd:
        x0 = 2*np.pi*np.random.rand(3*p)

    _, opt_energy = minimize_nlopt(qaoa_obj_qiskit, x0, p=p, rhobeg=0.01, cd=cd)
    opt_ar = (opt_energy-best_portfolio[1])/(best_portfolio[0]-best_portfolio[1])

    print(f"energy = {opt_energy}, Approximation ratio = {opt_ar}")



