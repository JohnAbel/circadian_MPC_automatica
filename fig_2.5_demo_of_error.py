"""
j.h.abel 27/7/2016

we know that bang-bang is optimal when the whole step is + or -. what about when
it's not, and the step is half and half?

- first test - is the phase shift the amount that we integrate over time (yes,
  so prc is based on time) or integrated over oscillator phase? will the area
  under the effective (integrated) PRC tell us anything?

  ans: yes, the area under the ePRC dictates the shift


- second test - for a time step over the 0-cross, what is the maximal adv/delay?
  if it is bang-bang, how do we do this? what about the 0-cross of the iPRC?

  ans:

- third test - can we find a bound on the amount of loss?

- finally - can we get phase shifting to within \delta of a final phase in
  optimal time?
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
# the optimal cases
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

def loss_zero_cross(umax, stepsize, cross=5, start_bound=[0,12]):
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
step_4h = 4*23.7/24
zero_neg_start, zero_neg_cross, loss_neg = loss_zero_cross(0.06, step_4h)


us_max = [0.00,0.06,0.00]
phi_i = pmodel._t_to_phi(zero_neg_start-step_4h)
times = [0]
maxphis  = [phi_i]
tot_max_shift = 0
for inp in us_max:
    inttimes, new_phi, step_shift = step_integrate(phi_i, inp, step_4h)
    phi_i = new_phi[-1][0]
    tot_max_shift += step_shift
    print step_shift
    times+=list(inttimes+times[-1])
    maxphis +=list(new_phi)

us_0 = [0.00,0.00,0.00]
phi_i = pmodel._t_to_phi(zero_neg_start-step_4h)
times0 = [0]
phis0  = [phi_i]
tot_max_shift = 0
for inp in us_0:
    inttimes, new_phi, step_shift = step_integrate(phi_i, inp, step_4h)
    phi_i = new_phi[-1][0]
    tot_max_shift += step_shift
    print step_shift
    times0+=list(inttimes+times0[-1])
    phis0 +=list(new_phi)

# correct times
times = np.asarray(times)*24/pmodel.T
times0 = np.asarray(times0)*24/pmodel.T
maxphis = np.asarray(maxphis).flatten()
phis0 = np.asarray(phis0).flatten()



plt.figure(figsize=(3.5,2.2))
gs = gridspec.GridSpec(2,2)
ax=plt.subplot(gs[0,0])
bx=plt.subplot(gs[0,1])
cx=plt.subplot(gs[1,0])
dx=plt.subplot(gs[1,1])


ax2 = ax.twinx()
us0 = np.hstack([[0],us_0])
ax2.step([0,4,8,12], us0, 'hl')
ax2.set_ylim([-0.005,0.08])
ax2.set_yticklabels([])
ax.plot(times0, -pmodel.pPRC_interp.splines[15](start_time+pmodel._phi_to_t(phis0)), 'k', label='PRC')
ax.axhline(0, c='k', ls=':')

bx2 = bx.twinx()
us = np.hstack([[0],us_max])
bx2.step([0,4,8,12], us, 'hl')
bx2.set_ylim([-0.005,0.08])
bx2.set_ylabel('u(t)')
bx.plot(times, -pmodel.pPRC_interp.splines[15](start_time+pmodel._phi_to_t(maxphis)), 'k', label='PRC')
bx.axhline(0, c='k', ls=':')


cx.plot(times0, phis0, 'k', label = '$u_2=0$')
dx.plot(times, maxphis, 'k', label = '$u_2=u_{max}$')
cx.legend()
dx.legend()

#formatting
axs = [ax,bx,cx,dx]
for axi in axs:
    axi.set_xlim([0,12])
    axi.set_xticks([0,4,8,12])

ax.set_xticklabels([])
bx.set_xticklabels([])
bx.set_yticklabels([])
dx.set_yticklabels([])
ax.set_ylabel('PRC')
cx.set_xlabel('Time (h)')
cx.set_ylabel(r'$\varphi$')

plt.tight_layout(**plo.layout_pad)














