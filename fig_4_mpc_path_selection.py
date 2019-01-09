"""
j.h.abel 8/15/2017

we know that usually bang-bang is optimal, or at least we can solve the
optimal control numerically. however, how does that optimal control change with
predictive horison? here, we get the different paths taken by each predictive horizon
to show that we need a different metric to safely choose direction.

ultimately, we choose the dividing line from the true optimal as the one for the
piecewise-constant optimal. that way our error is bounded anyway.
"""

# imports
from __future__ import division

import numpy as np
from scipy import integrate, optimize, stats, interpolate
import matplotlib.pyplot as plt
from matplotlib import gridspec
import casadi as cs
import brewer2mpl as colorbrewer
from concurrent import futures

#local imports
from LocalModels.hirota2012 import model, param, y0in
from LocalImports import LimitCycle as lco
from LocalImports import PlotOptions as plo
from LocalImports import Utilities as ut


pmodel = lco.Oscillator(model(), param, y0in)
pmodel.calc_y0()
pmodel.corestationary()
pmodel.limit_cycle()
pmodel.find_prc()

# start at negative prc region
times = np.linspace(0,pmodel.T,10001)
roots = pmodel.pPRC_interp.splines[15].root_offset()
neg_start_root = roots[0] # for when region becomes positive
pos_start_root = roots[1] # for when region becomes negative
umax = 0.06

# define a periodic spline for the prc
start_time = pos_start_root
prc_pos = ut.threshold(-pmodel.pPRC_interp(times+start_time)[:, 15],
                        threshmin=0)
prc_neg = ut.threshold(-pmodel.pPRC_interp(times+start_time)[:, 15],
                        threshmax=0)
prc = -pmodel.pPRC_interp(times+start_time)[:, 15]
prc_spl = pmodel.pPRC_interp.splines[15]

# the optimal cases - capture only + or -
prc_pos_spl = ut.PeriodicSpline(times, prc_pos, period=pmodel.T)
prc_neg_spl = ut.PeriodicSpline(times, prc_neg, period=pmodel.T)

# define integration for one step
def step_integrate(phi0, u_val, step):
    """ function that integrates one step forward. returns final phase,
    total shift value """
    def dphidt(phi, t):
        return ((2*np.pi)/pmodel.T
                - u_val*prc_spl(start_time+(phi)*pmodel.T/(2*np.pi)))

    int_times = np.linspace(0,step,101) # in hours
    phis = integrate.odeint(dphidt, [phi0], int_times, hmax=0.01)
    return int_times, phis, phis[-1][0]-phi0-2*np.pi/pmodel.T*step


def mpc_pred_horiz(init_phi, delta_phi_f, stepsize, pred_horiz):
    """
    performs the optimization over the predictive horizon to give the optimal
    set of steps u
    """
    # get init phase
    phi0 = init_phi
    # get the ref phase
    ref_phase0 = init_phi + delta_phi_f
    # get times
    steps = np.arange(pred_horiz)
    tstarts = steps*stepsize
    tends   = (steps+1)*stepsize

    # get ref phases at end of each step
    ref_phis_ends = tends*2*np.pi/pmodel.T + ref_phase0

    def optimize_inputs(us, phi0=phi0):
        endphis = []
        for u in us:
            results = step_integrate(phi0, u, stepsize)
            endphis.append(results[1][-1])
            phi0 = results[1][-1][0]
        endphis = np.asarray(endphis).flatten()
        errs= np.asarray([(endphis-ref_phis_ends)%(2*np.pi),
                (ref_phis_ends-endphis)%(2*np.pi)]).min(0)
        return 10*(errs**2).sum()+(us**2).sum()

    mins = optimize.minimize(optimize_inputs, [0.03]*pred_horiz,
                             bounds = [[0,0.06]]*pred_horiz)

    return mins.x

def solutions_from_horizon(us, stepsize, init_phi):
    """ takes a set of us, a stepsize, and a phi0; returns the trajectories of the solutions_from_horizon """

    pred_horiz = len(us)
    phases = []
    ref_phases = []
    states = []
    times = []
    running_time = 0
    phi0 = init_phi
    for u in us:
        results = step_integrate(phi0, u, stepsize)
        # collect results
        times = np.hstack([times, results[0][:-1]+running_time])
        phases = np.hstack([phases, results[1][:-1].flatten()])
        # set up for next step
        phi0 = results[1][-1][0]
        running_time = results[0][-1]+running_time

    ref_phases = init_phi+times*2*np.pi/pmodel.T

    return {'ref_phases': ref_phases,
            'times': times,
            'phases': phases,
            'us': us,
            'pred_horiz': pred_horiz}


# get what all the predictive horizons suggest
step_2h = 2*pmodel.T/24
us_horiz = []
for pred_horiz in np.arange(1,13):
    us_horiz.append(mpc_pred_horiz(0, np.pi, step_2h, pred_horiz))

# then recover the trajectories
results_dict_of_dicts = {}
for us in us_horiz:
    results = solutions_from_horizon(us, step_2h, 0)
    results_dict_of_dicts[results['pred_horiz']] = results



plo.PlotOptions(ticks='in')
plt.figure(figsize=(3.3,2.))



ax = plt.subplot()

for key in [5, 3, 10]:
    results = results_dict_of_dicts[key]
    normtimes = results['times']*24/23.7
    ax.plot(normtimes, results['phases'], label = 'N$_p$ = '+str(key))

fulltimes = np.copy(normtimes)
refphases = results['ref_phases']
ax.plot(normtimes, results['ref_phases'], color='k', label='u = 0')
ax.plot(normtimes, results['ref_phases']-np.pi, color='k', ls=':')
ax.plot(normtimes, results['ref_phases']+np.pi, color='k', ls=':',
    label='$\phi_r$')


ax.set_yticks([-np.pi, 0, np.pi, 2*np.pi, 3*np.pi])
ax.set_yticklabels(['-$\pi$', '0', '$\pi$', '2$\pi$', '3$\pi$'])
ax.set_ylim([-np.pi, 3*np.pi])
ax.set_xlim([0,20])

ax.set_xlabel('Time (h)')
ax.set_ylabel('$\phi$')

plt.legend()




plo.PlotOptions(ticks='in')
plt.figure(figsize=(3.3,2.5))
gs = gridspec.GridSpec(2,1, height_ratios = (2.5,1))
ax = plt.subplot(gs[0,0])

# check which
results = results_dict_of_dicts[3]
normtimes = results['times']*24/23.7

ax.plot(normtimes, np.sin(results['phases']), 'f', label = 'MPC Solution, N$_p$=5')
ax.plot(fulltimes, np.sin(refphases+np.pi), 'k:', label='Reference Tracjectory')
ax.set_xlim([0,20])
ax.set_ylabel('sin($\phi$)')
ax.set_xticklabels('')
ax.legend()

bx = plt.subplot(gs[1,0])
# get spline representation of phis
bx.plot(normtimes, -pmodel.pPRC_interp.splines[15](pmodel._phi_to_t(results['phases'])+start_time), 'k', label='PRC')

bx2 = bx.twinx()
us = np.hstack([[0],results['us']])
utimes = 2*np.arange(len(us))
bx2.step(utimes, us)

bx2.set_ylim([-0.002,0.1])
bx2.set_ylabel('u(t)')
bx.set_ylabel('PRC')
bx.set_ylim([-3,6])
bx.set_xlim([0,20])
bx.set_xlabel('Time (h)')
plt.legend()
plt.tight_layout(**plo.layout_pad)




plo.PlotOptions(ticks='in')
plt.figure(figsize=(3.3,2.5))
gs = gridspec.GridSpec(2,1, height_ratios = (2.5,1))
ax = plt.subplot(gs[0,0])

# check which
results = results_dict_of_dicts[5]
normtimes = results['times']*24/23.7

ax.plot(normtimes, np.sin(results['phases']), 'f', label = 'MPC Solution, N$_p$=5')
ax.plot(fulltimes, np.sin(refphases+np.pi), 'k:', label='Reference Tracjectory')
ax.set_xlim([0,20])
ax.set_ylabel('sin($\phi$)')
ax.set_xticklabels('')
ax.legend()

bx = plt.subplot(gs[1,0])
# get spline representation of phis
bx.plot(normtimes, -pmodel.pPRC_interp.splines[15](pmodel._phi_to_t(results['phases'])+start_time), 'k', label='PRC')

bx2 = bx.twinx()
us = np.hstack([[0],results['us']])
utimes = 2*np.arange(len(us))
bx2.step(utimes, us)
bx2.set_ylim([-0.002,0.1])
bx2.set_ylabel('u(t)')
bx.set_ylabel('PRC')
bx.set_xlim([0,20])
bx.set_ylim([-3,6])
bx.set_xlabel('Time (h)')
plt.legend()
plt.tight_layout(**plo.layout_pad)


plo.PlotOptions(ticks='in')
plt.figure(figsize=(3.3,2.5))
gs = gridspec.GridSpec(2,1, height_ratios = (2.5,1))
ax = plt.subplot(gs[0,0])

# check which
results = results_dict_of_dicts[10]
normtimes = results['times']*24/23.7

ax.plot(normtimes, np.sin(results['phases']), 'f', label = 'MPC Solution, N$_p$=5')
ax.plot(fulltimes, np.sin(refphases+np.pi), 'k:', label='Reference Tracjectory')
ax.set_xlim([0,20])
ax.set_ylabel('sin($\phi$)')
ax.set_xticklabels('')
ax.legend()

bx = plt.subplot(gs[1,0])
# get spline representation of phis
bx.plot(normtimes, -pmodel.pPRC_interp.splines[15](pmodel._phi_to_t(results['phases'])+start_time), 'k', label='PRC')

bx2 = bx.twinx()
us = np.hstack([[0],results['us']])
utimes = 2*np.arange(len(us))
bx2.step(utimes, us)
bx2.set_ylim([-0.002,0.1])
bx2.set_ylabel('u(t)')
bx.set_ylabel('PRC')
bx.set_xlim([0,20])
bx.set_ylim([-3,6])
bx.set_xlabel('Time (h)')
plt.legend()
plt.tight_layout(**plo.layout_pad)




