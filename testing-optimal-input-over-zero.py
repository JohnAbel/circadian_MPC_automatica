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

#
#
#           first test
#
#

# get phases and shift total
times, phases, shift = step_integrate(0, 0.06, 9)
# get effective PRC
stim_PRC = prc_spl(start_time+phases*pmodel.T/(2*np.pi))
ePRC = interpolate.UnivariateSpline(0+times, stim_PRC, k=3, s=0)
au_ePRC = integrate.cumtrapz(ePRC(times), times)

print 'shift = '+str(shift)
print 'area under ePRC = '+str(-0.06*au_ePRC[-1])

# so, the area under the ePRC dictates the shift


#
#
#           second test
#
#

# or, find where the area under the ePRC is equal for 0.06. for 0.00 it is equal
# get phases and shift total
times, phases, shift = step_integrate(6*2*np.pi/pmodel.T, 0.06, 4)
# get effective PRC
stim_PRC = prc_spl(start_time+phases*pmodel.T/(2*np.pi))
ePRC = interpolate.UnivariateSpline(0+times, stim_PRC, k=3, s=0)
au_ePRC = integrate.cumtrapz(ePRC(times), times)

# so starting at 6h and going for 3.6h leaves it neutral
times, phases, shift = step_integrate(6*2*np.pi/pmodel.T, 0.06, 3.60)
times, phases0, shift0 = step_integrate(6*2*np.pi/pmodel.T, 0.0, 3.60)

# find the best shift
shifts = []
ePRCs = []
inputs = np.arange(0,0.06,0.001)
for inp in inputs:
    times, phases, shift = step_integrate(6*2*np.pi/pmodel.T, inp, 3.60)
    shifts.append(shift)
    stim_PRC = prc_spl(start_time+phases*pmodel.T/(2*np.pi))
    ePRC = interpolate.UnivariateSpline(6+times, stim_PRC, k=3, s=0)
    ePRCs.append(ePRC)
plt.plot(inputs, shifts)
plt.figure()
for ePRC in [ePRCs[0], ePRCs[-1]]:
    plt.plot(6+times,ePRC(6+times))

# so the WORST case for advance or delay is doing nothing... doing SOMEthing 
# will result in an advance or delay (whichever comes first?)
# they end up at the same phase regardless of all or nothing, so we can 
# calculate the loss


#
#
#          third test
#
#

# how do we found out how much loss there is?
# each time it passes the zero-cross, the worst case is lining up to lose 
# equal amounts of advance and delay, and the 0-input case provides just as much
# shift as the full-input case

# regardless of input, each perfectly misaligned zero-cross can only incur 
# either advance or delay
# so, the worst-case scenario for either is the same - zero/max input. for 
# advance or delay one will improve with max/2, for example. but 0 is safely
# the lowest.

# so the maximum loss at each zero-cross is equal to the area under the ePRC_max
# on only the positive or negative size (these are equal). to calculate this, we
# can write the following function:
step_test = 4# time in h

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


# test it!
zero_neg_start, loss_neg = loss_zero_cross(0.06, step_test)
times, phases, shift = step_integrate(zero_neg_start*2*np.pi/pmodel.T, 0.06, 
                                            step_test)
times, phases0, shift0 = step_integrate(zero_neg_start*2*np.pi/pmodel.T, 0.0, 
                                            step_test)
# show that case is the same
plt.plot(times, phases0)
plt.plot(times, phases)

# test where the result is 0 for the later cross
zero_pos_start, loss_pos = loss_zero_cross(0.06, step_test,
                             cross=22, start_bound=[18,27])
times, phases, shift = step_integrate(zero_pos_start*2*np.pi/pmodel.T, 0.06, 
                                            step_test)
times, phases0, shift0 = step_integrate(zero_pos_start*2*np.pi/pmodel.T, 0.0, 
                                            step_test)
# show that case is the same
plt.plot(times, phases0)
plt.plot(times, phases)

# bam, that worked. so, in the end, we can get a max loss per cycle on precision
# of the alignment


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

# for each delta_phi_fs, find the 
delta_phi_fs = np.arange(0,2*np.pi,0.01)
# find the time at which the oscillator reaches it, and how many times it has 
# looped
numcycles = []
time_to_hit = []
for delta_phi_f in delta_phi_fs:
    first_hit_adv = np.where(delta_phis_total[:,2]>delta_phi_f)[0]
    first_hit_del = np.where(delta_phis_total[:,3]+2*np.pi<delta_phi_f)[0]
    first_hit = np.min(np.hstack([first_hit_adv, first_hit_del]))
    time_to_hit.append(int_times[first_hit])
    # if it hits from delay
    if first_hit in first_hit_del:
        numcycles.append(delta_phis_total[first_hit,1]//(2*np.pi))
    # if it hits from advance
    if first_hit in first_hit_adv:
        numcycles.append(delta_phis_total[first_hit,0]//(2*np.pi))

# thus, we now have the times to hit, and the number of cycles it takes to do so

#next, calculate how many steps it takes to get there
numsteps = np.array(time_to_hit)//step_test + 1

# now, find the most advance or most delay for each time step count numerically
# then see if that is sufficient to achieve each shift
allowed_steps = np.arange(1,int(np.max(numsteps)+1))

# function to return max/min shift using particle swarm
def maximal_shifts_pso(inputs):
    """
    This function returns the maximal shifts (+/-) for an oscillator starting
    at some initial phase and moving some number of steps forward
    """
    numsteps, phi_init, step_duration = inputs
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
            
    def total_shift(inputs, phi_init, maxmin):
        """ this is the function we maximize or minimize """
        tot_shift = 0
        phi_i = phi_init
        for inp in inputs:
            new_phi, step_shift = step_integrate(phi_i, inp, step_norm)
            phi_i = new_phi
            tot_shift += step_shift
        if maxmin is 'max':
            return -1*tot_shift
        elif maxmin is 'min':
            return tot_shift

    # pyswarm optimization
    minopt, minshift = pso(total_shift, [0.0]*numsteps, 
                         [0.06]*numsteps, 
                        args=([phi_init, 'min']),
                        maxiter=100, swarmsize=200, minstep=1e-6,
                        minfunc=1e-5)

    maxopt, maxshift = pso(total_shift, [0.0]*numsteps, 
                         [0.06]*numsteps, 
                        args=([phi_init, 'max']),
                        maxiter=100, swarmsize=200, minstep=1e-6,
                        minfunc=1e-5)

    return maxopt, -maxshift, minopt, minshift

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


inputs = []
for step in allowed_steps:
    inputs.append([step, 0, step_test])

"""
# for the pyswarm method
with futures.ProcessPoolExecutor(max_workers=27) as executor:
    result = executor.map(maximal_shifts, inputs)
"""

# optimize all the things sequentially, with the initial guess as the
# last optimized run
ig_max = []; ig_min = [] # initial guesses for max and min
maxs = []; mins = []; max_us = []; min_us = [] # data to collect
for inp in inputs:
    input_with_guesses=inp+[ig_max, ig_min]
    maxopt, maxshift, minopt, minshift \
                        = maximal_shifts_scipy(input_with_guesses)

    maxs.append(maxshift)
    mins.append(minshift)
    max_us.append([maxopt])
    ig_max = maxopt
    min_us.append([minopt])
    ig_min = minopt



#
#
#              a final test: a plot showing bound and results for 2h and 6h
#
#


#2h
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



cycle_loss_2 = 0.118085551869491
cycle_loss_6 = 0.32605369526603262


plo.PlotOptions(ticks='in')
plt.plot(delta_phi_fs, (np.asarray(numcycles)+1)*cycle_loss_2*24/(2*np.pi), 'k',
    label = '2h step bound')

plt.plot(delta_phi_fs, np.asarray(errors_2)*24/(2*np.pi), 
                'k.')


#plt.plot(delta_phi_fs, (np.asarray(numcycles)+1)*cycle_loss_6*24/(2*np.pi), 'h',
#    label = '4h step bound')
#
#plt.plot(delta_phi_fs, np.asarray(errors_6)*24/(2*np.pi), 
#                color='h', marker='.', ls='')

plt.legend()
plt.xlabel('$\Delta\phi_f$ (rad)')
plt.ylabel('Optimal-Time Residual Error (h)')
plt.tight_layout(**plo.layout_pad)


#
#
# something wehnt wrong in that. 
#

error_loc = np.argmax(errors_2)
error_phaseshift = delta_phi_fs[error_loc]
error_steps = int(numsteps2[error_loc]-1)
error_us_max = max_us2[error_steps][0]
error_us_min = min_us2[error_steps][0]

# test the steps
def step_integrate(phi0, u_val, step):
    """ function that integrates one step forward. returns final phase,
    total shift value """
    def dphidt(phi, t):
        return ((2*np.pi)/pmodel.T 
                - u_val*prc_spl(start_time+(phi)*pmodel.T/(2*np.pi)))

    int_times = np.linspace(0,step,101) # in hours
    phis = integrate.odeint(dphidt, [phi0], int_times, hmax=0.01)
    return int_times, phis, phis[-1][0]-phi0-2*np.pi/pmodel.T*step

tot_max_shift = 0
phi_i = phi_init
times = [0]
maxphis  = [phi_init]
for inp in error_us_max:
    inttimes, new_phi, step_shift = step_integrate(phi_i, inp, step_norm)
    phi_i = new_phi[-1][0]
    tot_max_shift += step_shift
    print step_shift
    times+=list(inttimes+times[-1])
    maxphis +=list(new_phi)


tot_min_shift = 0
phi_i = phi_init
times = [0]
minphis  = [phi_init]
for inp in error_us_min:
    inttimes, new_phi, step_shift = step_integrate(phi_i, inp, step_norm)
    phi_i = new_phi[-1][0]
    tot_min_shift += step_shift
    print step_shift
    times+=list(inttimes+times[-1])
    minphis +=list(new_phi)

tot_min_shift+=2*np.pi

