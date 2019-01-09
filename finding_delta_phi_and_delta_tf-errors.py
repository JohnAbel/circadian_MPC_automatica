"""
j.h.abel 8/15/2017

we know that bang-bang is optimal when the whole step is + or -. what about when
it's not, and the step is half and half?

- finally - can we get phase shifting to within \delta_phi of a final phase in 
  optimal time?

- then, let's find the max
"""

# imports
from __future__ import division


import numpy as np
from scipy import integrate, optimize, stats, interpolate
import matplotlib.pyplot as plt
from matplotlib import gridspec
import casadi as cs
from pyswarm import pso
import brewer2mpl as colorbrewer
from concurrent import futures

#local imports
from LocalModels.hirota2012 import model, param, y0in
from LocalImports import LimitCycle as lco
from LocalImports import PlotOptions as plo
from LocalImports import Utilities as ut



#
#
step_test = 2*23.7/24# time in h
delta_phi_star = 1.2# set rather arbitrarily... but we can choose this to
# reflect the minimum error rather than the optimal, since it changes...
# otherwise 1.3755358873430426# boundary for adv/delay from calc below
#
#


# problem setup
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
prc_pos = stats.threshold(-pmodel.pPRC_interp(times+start_time)[:, 15],
                        threshmin=0)
prc_neg = stats.threshold(-pmodel.pPRC_interp(times+start_time)[:, 15],
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

# find the max advance or delay in each cycle
def dphitot_dt(phis, t):
    [phi_osc_pos, phi_osc_neg, phi_shift_pos, phi_shift_neg] = phis
    dphi_osc_pos_dt = (2*np.pi)/pmodel.T +\
                             umax*prc_pos_spl(phi_osc_pos*pmodel.T/(2*np.pi))
    dphi_osc_neg_dt = (2*np.pi)/pmodel.T +\
                             umax*prc_neg_spl(phi_osc_neg*pmodel.T/(2*np.pi))
    dphi_shft_pos_dt = umax*prc_pos_spl(phi_osc_pos*pmodel.T/(2*np.pi))
    dphi_shft_neg_dt = umax*prc_neg_spl(phi_osc_neg*pmodel.T/(2*np.pi))
    return dphi_osc_pos_dt, dphi_osc_neg_dt, dphi_shft_pos_dt, dphi_shft_neg_dt

int_times = np.linspace(0,3*pmodel.T, 10001)
delta_phis_total = integrate.odeint(dphitot_dt, [0,0,0,0], int_times,
                                    hmax=0.001)
phis_adv = delta_phis_total[:,0]
phis_del = delta_phis_total[:,1]
advs = delta_phis_total[:,2]
dels = delta_phis_total[:,3]
delta_phi_star_calc = delta_phis_total[np.min(
            np.where(delta_phis_total[:,2]-delta_phis_total[:,3]>2*np.pi)
            ), 2]

# max 1 cycle advances are where the oscillator reaches 0 again
max_1cyc_adv = advs[np.min(
            np.where(phis_adv>2*np.pi)[0]
            )]
max_1cyc_del = dels[np.min(
            np.where(phis_del>2*np.pi)[0]
            )]



def loss_zero_cross(umax, stepsize, cross=6, start_bound=[0,12]):
    """
    calculates the max loss for a specific zero-crossing of the PRC
    """
    # first, find where the pulse lines up the worst
    def min_shift(init):
        init = init[0]
        times, phases, shift = step_integrate(init*2*np.pi/pmodel.T,
                                              umax, stepsize)
        return np.abs(shift)
    # get alignment
    mins = optimize.minimize(min_shift, cross, bounds = [start_bound])
    even_start = mins.x[0]

    times, phases, shift = step_integrate(even_start*2*np.pi/pmodel.T,
                                              umax, stepsize)
    stim_PRC = prc_spl(start_time+phases*pmodel.T/(2*np.pi))
    ePRC = interpolate.UnivariateSpline(times, stim_PRC, k=3, s=0)
    loss = np.max(umax*np.abs(integrate.cumtrapz(ePRC(times), times)))
    zero_cross = times[np.argmax(umax*np.abs(integrate.cumtrapz(ePRC(times),
                     times)))]
    return even_start, zero_cross, loss


# find the loss for each crossing, where it starts, where it crosses
zero_neg_start, zero_neg_cross, loss_neg = loss_zero_cross(0.06, step_test)
zero_pos_start, zero_pos_cross, loss_pos = loss_zero_cross(0.06, step_test,
                             cross=22, start_bound=[18,27])

adv_per_cycle = max_1cyc_adv - loss_neg - loss_pos
del_per_cycle = max_1cyc_del + loss_neg + loss_pos

# for the advance, there is also the slowdown loss - the phase advance lost
# by the oscillator not advancing
slowdown_pos = np.abs(step_integrate(zero_pos_start*2*np.pi/pmodel.T, 
    0.06, zero_pos_cross)[-1])
slowdown_neg = np.abs(step_integrate(zero_neg_start*2*np.pi/pmodel.T, 
    0.06, zero_neg_cross)[-1])
# slowdown_loss is how much phase shift we miss at most due to this
def max_shift_pos(init):
    init = init[0]
    times, phases, shift = step_integrate(init*2*np.pi/pmodel.T,
                                          umax, slowdown_pos*pmodel.T/(2*np.pi))
    return -shift
def max_shift_neg(init):
    init = init[0]
    times, phases, shift = step_integrate(init*2*np.pi/pmodel.T,
                                          umax, slowdown_neg*pmodel.T/(2*np.pi))
    return shift

pos_loss_maximization = optimize.minimize(max_shift_pos, [0], bounds=[[0,8]])
neg_loss_maximization = optimize.minimize(max_shift_neg, [0], bounds=[[12,24]])
slowdown_pos_loss = pos_loss_maximization.fun
slowdown_neg_loss = neg_loss_maximization.fun


# figure out the direction and number of cycle bounds
delta_phi_fs = np.arange(0,2*np.pi,0.001)
adv_del = [] #0 if advance, 1 if delay
numcycles = []
del_phi = []
for delta_phi in delta_phi_fs:
    direction = int(delta_phi > delta_phi_star)
    if direction==0:
        ncyc = 1 + delta_phi//adv_per_cycle
        numcycles.append(ncyc)
        adv_del.append(0)
        del_phi.append((ncyc)*
            (np.abs(slowdown_pos_loss+slowdown_pos_loss)+loss_neg+loss_pos))
    elif direction==1:
        ncyc = 1+ (2*np.pi-delta_phi)//-del_per_cycle
        adv_del.append(1)
        numcycles.append(ncyc)
        del_phi.append((ncyc)*
            (loss_neg+loss_pos))





plo.PlotOptions(ticks='in')
plo.PlotOptions(uselatex=True, ticks='in')
plt.figure(figsize=(3.5,2.85))
gs = gridspec.GridSpec(3,1)

ax = plt.subplot(gs[0,0])
bx = plt.subplot(gs[1,0])
cx = plt.subplot(gs[2,0])


ax.plot(delta_phi_fs, numcycles)
ax.axvline(delta_phi_star, color = 'k', ls=":")
ax.set_ylabel("Number Cycles")
bx.set_ylim([0,5.2])

bx.plot(delta_phi_fs, np.asarray(del_phi))
bx.set_ylim([0,1.])
bx.axvline(delta_phi_star, color = 'k', ls=":")
bx.set_ylabel(r"Optimal-time\\ error (rad)")

cx.set_ylabel(r"Reset-time\\ error (h)")
cx.set_xlabel(r"$\Delta\phi_f$")

plo.format_2pi_axis(ax)
plo.format_2pi_axis(bx)
plo.format_2pi_axis(cx)
ax.set_xticklabels([])
bx.set_xticklabels([])


plt.tight_layout(**plo.layout_pad)