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
import brewer2mpl as colorbrewer
from concurrent import futures

#local imports
from LocalModels.hirota2012 import model, param, y0in
from LocalImports import LimitCycle as lco
from LocalImports import PlotOptions as plo
from LocalImports import Utilities as ut


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


def calculate_mintime_toshift(step_test, pulse_loc):
    """
    Calculates the dots - the phase error for each desired shift

    """

    # we want the pulse at the same time every day.
    first_step = pulse_loc
    steps = 12 # whole 24 periods
    on_step = step_test
    off_step = pmodel.T-step_test

    int_times_off, phis_off, accum_shift = step_integrate(0, 0, first_step)
    phis = phis_off.flatten()[:-1]
    int_times = int_times_off[:-1]
    dt = int_times_off[1]
    accum_shift = [accum_shift]

    # fix the on-off
    for step in range(steps):
        # the on step
        int_times_on, phis_on, accum_shift_on = step_integrate(phis_off[-1][0], 0.06, on_step)
        int_times = np.hstack([int_times, int_times_on[:-1]+dt+int_times[-1]])
        phis = np.hstack([phis, phis_on.flatten()[:-1]])
        accum_shift+=[accum_shift_on]

        # the off step
        int_times_off, phis_off, accum_shift_off = step_integrate(phis_on[-1][0],
                                                        0, off_step)
        int_times = np.hstack([int_times, int_times_off[:-1]+dt+int_times[-1]])
        phis = np.hstack([phis, phis_off.flatten()[:-1]])

    return_dict = {'times':int_times,
                   'phis':phis,
                   'shift':accum_shift,
                   'us':[]}

    return return_dict

# step size
step_test = 12*pmodel.T/24

# this will tell the final phase alignment, aka where we should put the pulse
zero_neg_start, zero_neg_cross, loss_neg = loss_zero_cross(0.06, step_test)

pulse_loc = np.arange(0.1,23.8,0.2)
results_list = [calculate_mintime_toshift(step_test, pulse) for pulse in pulse_loc]






plo.PlotOptions(ticks='in')
plo.PlotOptions(uselatex=True, ticks='in')
plt.figure()

for res in results_list:
    plt.plot(np.abs(res['shift']))
plt.xlabel('Day')
plt.ylabel('abs(Shift)')
plt.tight_layout(**plo.layout_pad)



