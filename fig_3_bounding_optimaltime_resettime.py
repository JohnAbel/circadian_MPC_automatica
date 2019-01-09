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


def bounds(step_test, delta_phi_star):
    """
    Finds the number of cycles, optimal time error, and reset time error
    for a given step size (step_test) and dividing point (delta_phi_star)

    returns {'delta_phi_fs' : delta_phi_fs, # the phases
            'directions' : directions, # the direction of each shift
            'numcycles' : numcycles, # the associated number of cycles
            'regions_achieved' : regions_achieved, # the min shift achieved
            'cyc_to_reachs' : cyc_to_reachs, # the max cycles to reach it
            'del_phis' : del_phis} # the error
    """

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
    del_phis = []
    directions = []
    regions_missed = []
    cyc_to_reachs = []
    for delta_phi in delta_phi_fs:
        direction = int(delta_phi > delta_phi_star)
        if direction==0:
            ncyc = 1 + delta_phi//max_1cyc_adv
            cyc_to_reach = 1 + delta_phi//adv_per_cycle
            adv_del.append(0)
            # upper limit of how much can be missed
            region_missed = (ncyc)*(
                np.abs(slowdown_pos_loss+slowdown_pos_loss)+loss_neg+loss_pos)
            del_phi = region_missed
        elif direction==1:
            ncyc = 1+ (2*np.pi-delta_phi)//-max_1cyc_del
            cyc_to_reach = 1+ (2*np.pi-delta_phi)//-del_per_cycle
            adv_del.append(1)
            # upper limit of how much can be missed
            region_missed = (ncyc)*(loss_neg+loss_pos)
            # will the minimum achieved be better or worse than the max lost
            del_phi = region_missed

        del_phis.append(del_phi)
        directions.append(direction)
        numcycles.append(ncyc)
        cyc_to_reachs.append(cyc_to_reach)
        regions_missed.append(region_missed)

    result_dict = {'delta_phi_fs' : delta_phi_fs,
                   'directions' : directions,
                   'numcycles' : numcycles,
                   'regions_missed' : regions_missed,
                   'cyc_to_reachs' : cyc_to_reachs,
                   'del_phis' : del_phis}

    return result_dict





#
#
step_4h = 4*23.7/24
step_2h = 2*23.7/24# time in h
step_1h = 1*23.7/24
delta_phi_star = 1.3755358873430426# set rather arbitrarily... but we can choose
# reflect the minimum error rather than the optimal, since it changes...
# otherwise 1.3755358873430426# boundary for adv/delay from calc below
#
#


res_4h = bounds(step_4h, delta_phi_star)
res_2h = bounds(step_2h, delta_phi_star)
res_1h = bounds(step_1h, delta_phi_star)


def calculate_optimal_control(inputs):
    """
    This function returns the maximal shifts (+/-) for an oscillator starting
    at some initial phase and moving some number of steps forward

    numsteps, step_norm, ig_max, ig_min = inputs
    """
    phi_init = 0
    numsteps, step_norm, ig_max, ig_min = inputs
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

def optimal_controls(step_test):
    # errors actually calculated
    numsteps = np.array(64)//step_test + 1
    allowed_steps = np.arange(1,int(np.max(numsteps)+1))

    inputs = []
    for numstep in allowed_steps:
        inputs.append([numstep, step_test])

    ig_max = []; ig_min = [] # initial guesses for max and min
    times = allowed_steps*step_test
    maxs = [0]; mins = [0]; max_us = [[]]; min_us = [[]] # data to collect
    for inp in inputs:
        input_with_guesses=inp+[ig_max, ig_min]
        maxopt, maxshift, minopt, minshift \
                            = calculate_optimal_control(input_with_guesses)

        maxs.append(maxshift)
        mins.append(minshift)
        max_us.append([maxopt])
        ig_max = maxopt
        min_us.append([minopt])
        ig_min = minopt

    optimal_dict = {'times' : times,
                    'maxs' : maxs,
                    'mins' : mins,
                    'max_us' : max_us,
                    'min_us' : min_us}
    return optimal_dict


opt_4h = optimal_controls(step_4h)
opt_2h = optimal_controls(step_2h)
opt_1h = optimal_controls(step_1h)



def calculate_mintime_phase_error(step_test, delta_phi_fs, optimal_dict):
    """
    Calculates the dots - the phase error for each desired shift

    """
    maxs = optimal_dict['maxs']
    mins = optimal_dict['mins']
    # first get the optimal ocntrol
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

    errors = []
    times_to_hit = []
    steps_to_hit = []
    for i,delta_phi_f in enumerate(delta_phi_fs):
        # find where the optimal control reaches the end, get the step count
        # there
        first_hit_adv = np.where(delta_phis_total[:,2]>delta_phi_f)[0]
        first_hit_del = np.where(delta_phis_total[:,3]+2*np.pi<delta_phi_f)[0]
        first_hit = np.min(np.hstack([first_hit_adv, first_hit_del]))
        time_to_hit = (int_times[first_hit])

        steps_allowed = int(time_to_hit//step_test+1)
        maxh = maxs[int(steps_allowed)]
        minh = mins[int(steps_allowed)]
        if minh+2*np.pi < delta_phi_f:
            error=0.
        elif maxh > delta_phi_f:
            error = 0.
        else:
            error = np.min([np.abs(maxh-delta_phi_f),
                            np.abs(minh+2*np.pi-delta_phi_f)])
        errors.append(error)
        times_to_hit.append(time_to_hit)
        steps_to_hit.append(steps_allowed)

    return_dict = {'errors' : np.asarray(errors),
                   'times_to_hit' : np.asarray(times_to_hit),
                   'steps_to_hit' : np.asarray(steps_to_hit)}
    return return_dict


errors_4h = calculate_mintime_phase_error(step_4h,
                res_4h['delta_phi_fs'], opt_4h)
errors_2h = calculate_mintime_phase_error(step_2h,
                res_4h['delta_phi_fs'], opt_2h)
errors_1h = calculate_mintime_phase_error(step_1h,
                res_1h['delta_phi_fs'], opt_1h)


def plot_bounds(axs, bounds_dicts):
    ax, bx, cx = axs
    delta_phi_fs = np.asarray(bounds_dicts[0]['delta_phi_fs'])
    numcycles = np.asarray(bounds_dicts[0]['numcycles'])
    directions = np.asarray(bounds_dicts[0]['directions'])

    dir_switch = delta_phi_fs[np.argmax(np.abs(np.diff(directions)))]

    ax.plot(delta_phi_fs, numcycles, color='k')
    ax.axvline(dir_switch, color = 'k', ls=":")
    ax.set_ylabel("Number Cycles,\nContinuous Control")
    ax.set_ylim([0,3.4])

    colors = ['l', 'h', 'f']
    labels = ['4h','2h','1h']
    for idx, bounds_dict in enumerate(bounds_dicts):
        cyc_to_reachs = np.asarray(bounds_dict['cyc_to_reachs'])
        regions_missed = np.asarray(bounds_dict['regions_missed'])
        del_phis = np.asarray(bounds_dict['del_phis'])

        bx.plot(delta_phi_fs, del_phis, color = colors[idx],
            label=labels[idx])
        bx.set_ylabel("Minimal-time\nphase error (rad)")

        cx.set_ylabel("Max Additional\nCycles to Reset")
        cx.plot(delta_phi_fs, cyc_to_reachs-numcycles, color = colors[idx],
            label=labels[idx])
        cx.set_xlabel(r"$\Delta\phi_f$")

    bx.axvline(dir_switch, color = 'k', ls=":")
    cx.axvline(dir_switch, color = 'k', ls=":")
    bx.set_ylim([0,1.5])
    cx.set_ylim([0,8])

    plo.format_2pi_axis(ax)
    plo.format_2pi_axis(bx)
    plo.format_2pi_axis(cx)
    ax.set_xticklabels([])
    bx.set_xticklabels([])


def plot_errors(ax, delta_phi_fs, error_dicts):
    colors = ['l', 'h', 'f']
    for idx, error_dict in enumerate(error_dicts):
        ax.plot(delta_phi_fs, error_dict['errors'],
            marker = 'x', color = colors[idx], ls='')


plo.PlotOptions(ticks='in')
plo.PlotOptions(uselatex=True, ticks='in')
plt.figure(figsize=(3.5,3.85))
gs = gridspec.GridSpec(3,1)

ax = plt.subplot(gs[0,0])
bx = plt.subplot(gs[1,0])
cx = plt.subplot(gs[2,0])



plot_errors(bx, res_1h['delta_phi_fs'], [errors_4h, errors_2h, errors_1h])

plot_bounds([ax,bx,cx], [res_4h, res_2h, res_1h])

bx.legend()

plt.tight_layout(**plo.layout_pad)








plo.PlotOptions(ticks='in')
plo.PlotOptions(uselatex=True, ticks='in')
plt.figure(figsize=(3.5,2.85))
gs = gridspec.GridSpec(3,1)

ax = plt.subplot(gs[0,0])
bx = plt.subplot(gs[1,0])
cx = plt.subplot(gs[2,0])

plot_bounds([ax,bx,cx], res_4h)

plt.tight_layout(**plo.layout_pad)