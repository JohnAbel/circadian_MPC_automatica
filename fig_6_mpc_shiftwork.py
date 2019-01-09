"""
j.h.abel 8/15/2017

solves the mpc problem for some token set of reference phases
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
start_time = 6*23.7/24
prc = -pmodel.pPRC_interp(times+start_time)[:, 15]
prc_spl = pmodel.pPRC_interp.splines[15]



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

    uses a dividing line of pi/2 as the adv/del
    """
    dividing_phi_f = np.pi/2

    # get init phase
    phi0 = init_phi
    # get the ref phase
    ref_phase0 = init_phi + delta_phi_f
    # get times
    steps = np.arange(pred_horiz)
    tstarts = steps*stepsize
    tends   = (steps+1)*stepsize

    # get ref phases at end of each step
    ref_phis_pred = tends*2*np.pi/pmodel.T + ref_phase0

    # choose the direction
    if delta_phi_f > dividing_phi_f:
        # achieve the shift by a delay
        def optimize_inputs(us, phi0=phi0):
            endphis = []
            for u in us:
                results = step_integrate(phi0, u, stepsize)
                endphis.append(results[1][-1])
                phi0 = results[1][-1][0]
            endphis = np.asarray(endphis).flatten()
            errs= np.abs(endphis-(ref_phis_pred-2*np.pi))
            return 10*(errs**2).sum()+(us**2).sum()
    else:
        # achieve the shift by an advance
        def optimize_inputs(us, phi0=phi0):
            endphis = []
            for u in us:
                results = step_integrate(phi0, u, stepsize)
                endphis.append(results[1][-1])
                phi0 = results[1][-1][0]
            endphis = np.asarray(endphis).flatten()
            errs= np.abs(endphis-ref_phis_pred)
            return 10*(errs**2).sum()+(us**2).sum()


    mins = optimize.minimize(optimize_inputs, [0.03]*pred_horiz,
                             bounds = [[0,0.06]]*pred_horiz)

    return mins.x


def mpc_problem(init_phi, ts, ref_phis, pred_horizon):
    """
    uses MPC to track the reference phis, using a stepsize and a predictive horizon.
    ts should be separated by stepsize.
    """
    # set up the system it gets applied to
    y0mpc = pmodel.lc(init_phi*pmodel.T/(2*np.pi)+start_time)

    # get step size, current phase, etc
    stepsize = ts[1]-ts[0]
    u_input = []
    sys_state = y0mpc

    sys_phis = []
    for idx, inst_time in enumerate(ts):
        #get the ref phase at the time, compare to system phase at the time
        ref_phi = ref_phis[idx]
        # remember, phi0 is defined as when the 0-cross happens
        sys_phi = (pmodel.phase_of_point(sys_state)-
                        pmodel._t_to_phi(start_time))%(2*np.pi)
        sys_phis.append(sys_phi)

        # phase error
        phi_diff = np.angle(np.exp(1j*sys_phi))-np.angle(np.exp(1j*ref_phi))
        delta_phi_f = -phi_diff%(2*np.pi) # the desired shift is to make up that angle

        if np.abs(phi_diff) > 0.1: #this value may be changed as desired
            # calculate the optimal inputs
            us_opt = mpc_pred_horiz(sys_phi, delta_phi_f, stepsize, pred_horizon)
            u_apply = us_opt[0]

        else:
            u_apply = 0


        print delta_phi_f, u_apply
        # move forward a step
        u_input.append(u_apply)
        mpc_param = np.copy(param)
        mpc_param[15] = mpc_param[15]-u_apply
        mpc_sys = lco.Oscillator(model(), mpc_param, sys_state)
        sys_progress = mpc_sys.int_odes(stepsize)
        sys_state = sys_progress[-1]

    return ts, sys_phis, u_input

def solutions_from_us(us, stepsize, init_phi):
    """ takes a set of us, a stepsize, and a phi0; returns the trajectories of the solutions """

    pred_horiz = len(us)
    phases = []
    ref_phases = []
    states = []
    times = []
    running_time = 0
    phi0 = init_phi

    y0mpc = pmodel.lc(phi0*pmodel.T/(2*np.pi)+start_time)
    sys_state = y0mpc

    for u_apply in us:
        mpc_param = np.copy(param)
        mpc_param[15] = mpc_param[15]-u_apply
        mpc_sys = lco.Oscillator(model(), mpc_param, sys_state)
        sys_progress = mpc_sys.int_odes(stepsize, numsteps=10)

        # append new times and phases
        times = times+list(mpc_sys.ts[:-1]+running_time)
        phases = phases + [pmodel.phase_of_point(state) for state in sys_progress[:-1]]

        #update for next step
        sys_state = sys_progress[-1]
        running_time = running_time+mpc_sys.ts[-1]

    u0_phases = init_phi+np.asarray(times)*2*np.pi/pmodel.T

    return {'u0_phases': np.asarray(u0_phases),
            'times': np.asarray(times),
            'phases': np.asarray(phases),
            'us': np.asarray(us)
            }

# phase changes
t1 = 19# 38h in
t2 = 47# 94h in
t3 = 75# 150h in
ref_phase_jump1 = -8/24*2*np.pi
ref_phase_jump2 = -8/24*2*np.pi
ref_phase_jump3 = -8/24*2*np.pi

# solve the mpc problem
initial_phase = 0
step_2h = 2*pmodel.T/24
ts = np.arange(0,200,step_2h) # reset to 200
time_jump1 = ts[t1]
time_jump2 = ts[t2]
time_jump3 = ts[t3]

# 8 steps in we see a delay of pi/4 (3h)
reference_phases = (2*np.pi/pmodel.T)*ts +\
             np.array([0]*t1+[ref_phase_jump1]*(len(ts)-t1)) +\
             np.array([0]*t2+[ref_phase_jump2]*(len(ts)-t2)) +\
             np.array([0]*t3+[ref_phase_jump3]*(len(ts)-t3))
ts, sys_phis, us_mpc = mpc_problem(initial_phase, ts, reference_phases, 3)

print "recovering solns"
# collect the result from us
mpc_solution = solutions_from_us(us_mpc, step_2h, initial_phase)
print "solns recovered"

# get more tightly sampled phases
ts_tight = np.arange(0,200,0.1)
ref_phases_tight = (2*np.pi/pmodel.T)*ts_tight
for idx, time in enumerate(ts_tight):
    if time > time_jump1:
        ref_phases_tight[idx] = ref_phases_tight[idx]+ref_phase_jump1
    if time > time_jump2:
        ref_phases_tight[idx] = ref_phases_tight[idx]+ref_phase_jump2
    if time > time_jump3:
        ref_phases_tight[idx] = ref_phases_tight[idx]+ref_phase_jump3

# plot to show it works
plt.plot(ts,reference_phases, label = 'step phis')
plt.plot(ts_tight, ref_phases_tight, label = 'tight phis')
plt.plot(mpc_solution['times'],
    np.unwrap(mpc_solution['phases'])-pmodel._t_to_phi(start_time),
    label='mpc phis')
plt.legend()






plo.PlotOptions(ticks='in')
plt.figure(figsize=(7,3.1))
gs = gridspec.GridSpec(2,1, height_ratios = (2.5,1))
ax = plt.subplot(gs[0,0])

# check which
results = mpc_solution
normtimes = results['times']*24/23.7
normtstight = ts_tight*24/23.7

ax.axvspan(6, 14, alpha=0.25, ymax=0.2, color='gray')
ax.axvspan(30, 38, alpha=0.25,ymax=0.2,  color='gray')
ax.axvspan(62, 70, alpha=0.25,ymax=0.2,  color='gray')
ax.axvspan(86, 94, alpha=0.25, ymax=0.2, color='gray')
ax.axvspan(118, 126, alpha=0.25, ymax=0.2, color='gray')
ax.axvspan(142, 150, alpha=0.25, ymax=0.2,  color='gray', label='Work Shift')

ax.plot(normtimes, np.sin(results['phases']-pmodel._t_to_phi(start_time)+pmodel._t_to_phi(pos_start_root)),
    'i', label = 'MPC Solution, N$_p$=5')
ax.plot(normtstight, np.sin(ref_phases_tight+pmodel._t_to_phi(pos_start_root)), 'k:', label='Reference Tracjectory')
ax.set_xlim([0,180])
ax.set_ylabel('sin($\phi$)')
ax.set_xticklabels('')
ax.set_ylim([-1.05,1.35])
ax.legend()
ax.set_xticks([0,24,48,72,96,120,144,168])




bx = plt.subplot(gs[1,0])
# get spline representation of phis
bx.plot(normtimes, -pmodel.pPRC_interp.splines[15](pmodel._phi_to_t(results['phases'])), 'k', label='PRC')

bx2 = bx.twinx()
us = np.hstack([[0],results['us']])
utimes = 2*np.arange(len(us))
bx2.step(utimes, us, 'hl')

bx2.set_ylim([0.0,0.1])
bx2.set_ylabel('u(t)')
bx.set_ylabel('PRC')
bx.set_ylim([-3,6])
bx.set_xlim([0,180])
bx.set_xticks([0,24,48,72,96,120,144,168])
bx.set_xlabel('Time (h)')
plt.legend()
plt.tight_layout(**plo.layout_pad)
