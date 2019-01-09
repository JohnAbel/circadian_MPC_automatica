"""
j.h.abel 7/8/2017

we know that bang-bang is optimal when the whole step is + or -. what about when
it's not, and the step is half and half?

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

step_test = 2# time in h

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
    return even_start, loss


# find the loss for each crossing
zero_neg_start, loss_neg = loss_zero_cross(0.06, step_test)
zero_pos_start, loss_pos = loss_zero_cross(0.06, step_test,
                             cross=22, start_bound=[18,27])

# for the advance, there is also the slowdown loss - the phase advance lost
# by the oscillator not advancing
slowdown = step_integrate(pos_start_root-step_test, 0.06, step_test)[-1]
# slowdown_loss is how much phase shift we miss at most due to this
def max_shift(init):
    init = init[0]
    times, phases, shift = step_integrate(init*2*np.pi/pmodel.T,
                                          umax, slowdown/2)
    return -np.abs(shift)

loss_maximization = optimize.minimize(max_shift, [0], bounds=[[0,8]])
slowdown_loss = -loss_maximization.fun

#
#
#           finally, let's try it out
#
#

# so we have a bound on phase shifts achievable in a certain amount of time,
# relative to the optimal. we should find a way to show the predicted outcome
# and then how close in phase you actually get in the optimal time. 

# this should be done for a given phi0 - say, 0

# start by finding how many cycles each shift takes.

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
time_max = int_times[np.min(
            np.where(delta_phis_total[:,2]-delta_phis_total[:,3]>2*np.pi)
            )]

# for each delta_phi_fs, find the time to reach it optimally, and the error
delta_phi_fs = np.arange(0,2*np.pi,0.01)
error_bound = []
time_to_hit = []

# what phase is the neg 0-cross
neg_0_phase = (neg_start_root+pmodel.T-pos_start_root)*2*np.pi/24

#> issue: step crosses the boundary but we dont recognize that here for some 
#> example: first positive step crossing the negative boundary at delta phi = 
#> and time = 6h. this is because speedup is not accounted for!

for delta_phi_f in delta_phi_fs:
    first_hit_adv = np.where(delta_phis_total[:,2]>delta_phi_f)[0]
    first_hit_del = np.where(delta_phis_total[:,3]+2*np.pi<delta_phi_f)[0]
    first_hit = np.min(np.hstack([first_hit_adv, first_hit_del]))
    time_to_hit.append(int_times[first_hit])
    # if it hits from delay
    if first_hit in first_hit_del:
        # number of complete cycles = times the phase crosses 2pi
        numcycles = (delta_phis_total[first_hit,1]+0)//(2*np.pi)
        # crosses - to + at the end of each cycle
        num_pos_crosses = numcycles 
        # crosses + to - within each cycle so numcycles+1
        num_neg_crosses = numcycles +1
        # no slowdown error from missing a + shift
        error_bound.append(num_pos_crosses*loss_pos +
                           num_neg_crosses*loss_neg)
    # if it hits from advance
    if first_hit in first_hit_adv:
        # number of complete cycles = times the phase crosses 2pi
        numcycles =(delta_phis_total[first_hit,0]+0)//(2*np.pi)
        # pos and neg crosses are same
        num_pos_crosses = numcycles
        num_neg_crosses = numcycles +1
        # slowdown error occurs when the oscillator misses its
        slowdown_error = slowdown_loss*(numcycles) 
        #not +1 because it occurs next cycle
        error_bound.append(num_pos_crosses*loss_pos +
                           num_neg_crosses*loss_neg + slowdown_error)


# thus, we now have the times to hit, and the error bound

#next, calculate how many steps it takes to get there
numsteps = np.array(time_to_hit)//step_test + 1

# now, find the most advance or most delay for each time step count numerically
# then see if that is sufficient to achieve each shift
allowed_steps = np.arange(1,int(np.max(numsteps)+1))

# same optimization, but done sequentially using a different minimizaiton
def maximal_shifts_scipy(inputs):
    """
    This function returns the maximal shifts (+/-) for an oscillator starting
    at some initial phase and moving some number of steps forward
    """
    numsteps, phi_init, step_duration, ig_max, ig_min = inputs
    step_norm = step_duration*pmodel.T/24 # step in hours assuming 24h cycle
    phi_osc_pos = phi_init
    phi_osc_neg = phi_init
    
    # define a function that integrates for step size
    def step_integrate(phi0, u_val, step):
        """ function that integrates one step forward. returns final phase,
        total shift value """
        def dphidt(phi, t):
            return ((2*np.pi)/pmodel.T 
                    - u_val*prc_spl(start_time+(phi)*pmodel.T/(2*np.pi)))

        int_times = np.linspace(0,step,101) # in hours
        phis = integrate.odeint(dphidt, [phi0], int_times, hmax=0.01)
        return phis[-1][0], phis[-1][0]-phi0-2*np.pi/pmodel.T*step
            
    def total_shift(control_inputs, phi_init, maxmin):
        """ this is the function we maximize or minimize """
        tot_shift = 0
        phi_i = phi_init
        for inp in control_inputs:
            new_phi, step_shift = step_integrate(phi_i, inp, step_norm)
            phi_i = new_phi
            tot_shift += step_shift
        if maxmin is 'max':
            return -tot_shift
        elif maxmin is 'min':
            return tot_shift

    def max_shift(us):
        return total_shift(us, phi_init, 'max')
    def min_shift(us):
        return total_shift(us, phi_init, 'min')


    # scipy optimization: multistart at either end
    max_opt1 = optimize.minimize(max_shift, # fcn to maximize
                    np.hstack([ig_max,[0.00]]), # initial guess for max shift
                    bounds=[[0,0.06]]*(numsteps)) # bounds
    max_opt2 = optimize.minimize(max_shift, # fcn to maximize
                    np.hstack([ig_max,[0.06]]), # initial guess for max shift
                    bounds=[[0,0.06]]*(numsteps)) # bounds
    max_opts = [max_opt1, max_opt2]
    max_opt = max_opts[np.argmin([max_opt1.fun, max_opt2.fun])]
    multi = False
    if max_opt1.fun != max_opt2.fun:
        multi=True
        maxopt = max_opt.x
        maxshift = -max_opt.fun
    else:
        maxopt = max_opt.x
        maxshift = -max_opt.fun


    min_opt1 = optimize.minimize(min_shift, # fcn to maximize
                    np.hstack([ig_min,[0.00]]), # initial guess for max shift
                    bounds=[[0,0.06]]*(numsteps)) # bounds
    min_opt2 = optimize.minimize(min_shift, # fcn to maximize
                    np.hstack([ig_min,[0.06]]), # initial guess for max shift
                    bounds=[[0,0.06]]*(numsteps)) # bounds
    min_opts = [min_opt1, min_opt2]
    min_opt = min_opts[np.argmin([min_opt1.fun, min_opt2.fun])]
    multi = False
    if min_opt1.fun != min_opt2.fun:
        multi=True
        minopt = min_opt.x
        minshift = min_opt.fun
    else:
        minopt = min_opt.x
        minshift = min_opt.fun

    return maxopt, maxshift, minopt, minshift


#2h errors actually calculated
numsteps2 = np.array(time_to_hit)//2 + 1
allowed_steps2 = np.arange(1,int(np.max(numsteps2)+1))

inputs2 = []
for step in allowed_steps2:
    inputs2.append([step, 0, 2])

ig_max = []; ig_min = [] # initial guesses for max and min
times2 = allowed_steps2*2
maxs2 = [0]; mins2 = [0]; max_us2 = [[]]; min_us2 = [[]] # data to collect
for inp in inputs2:
    input_with_guesses=inp+[ig_max, ig_min]
    maxopt, maxshift, minopt, minshift \
                        = maximal_shifts_scipy(input_with_guesses)

    maxs2.append(maxshift)
    mins2.append(minshift)
    max_us2.append([maxopt])
    ig_max = maxopt
    min_us2.append([minopt])
    ig_min = minopt


# now, let's see how everything compares. we want to compare phase at the end of
# $numsteps to the desired delta phi f
errors_2 = []
for i,delta_phi_f in enumerate(delta_phi_fs):
    steps_allowed = numsteps2[i]
    max2 = maxs2[int(steps_allowed)]
    min2 = mins2[int(steps_allowed)]
    if min2+2*np.pi < delta_phi_f:
        error=0.
    elif max2 > delta_phi_f:
        error = 0.
    else:
        error = np.min([np.abs(max2-delta_phi_f), 
                        np.abs(min2+2*np.pi-delta_phi_f)])
    errors_2.append(error)




plo.PlotOptions(ticks='in')
plt.plot(delta_phi_fs, np.array(error_bound)*12/np.pi, 'k', label = '2h bound')
plt.plot(delta_phi_fs, np.asarray(errors_2)*24/(2*np.pi), 'f.')


#plt.plot(delta_phi_fs, (np.asarray(numcycles)+1)*cycle_loss_6*24/(2*np.pi), 'h',
#    label = '4h step bound')
#
#plt.plot(delta_phi_fs, np.asarray(errors_6)*24/(2*np.pi), 
#                color='h', marker='.', ls='')

plt.legend()
plt.xlabel('$\Delta\phi_f$ (rad)')
plt.ylim([0,2])
plt.ylabel('Optimal-Time Residual Error (h)')
plt.tight_layout(**plo.layout_pad)

