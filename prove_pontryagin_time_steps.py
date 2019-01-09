"""
j.h.abel 19/7/2016

ok so weird thing: how do we tell if the input over a time step should be the
max or the min? is it still bang-bang in those regions? i do not think so
"""

#
#
# -*- coding: utf-8 -*-
from __future__ import division


import numpy as np
from scipy import integrate, optimize, stats
import matplotlib.pyplot as plt
from matplotlib import gridspec
import casadi as cs
from pyswarm import pso
import brewer2mpl as colorbrewer

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

# define a periodic spline for the 
start_time = pos_start_root
prc_pos = stats.threshold(-pmodel.pPRC_interp(times+start_time)[:, 15],
                        threshmin=0)
prc_neg = stats.threshold(-pmodel.pPRC_interp(times+start_time)[:, 15],
                        threshmax=0)
prc = -pmodel.pPRC_interp(times+start_time)[:, 15]
prc_spl = pmodel.pPRC_interp.splines[15]


def delta_phis_step(phi_init, tf, step=2, u_max=0.06):
    """
    Step is the increase in time, in h. This function will perform as noted in
    the paper. That is, it will determine if the control should be max or min,
    then apply the control
    """
    step_x = step*pmodel.T/23.7
    phi_osc_pos = phi_init
    phi_osc_neg = phi_init
    
    # define a function that integrates for step size
    def step_integrate(phi0, step):
        # function to integrate
        def dphidt(phi, t):
            return ((2*np.pi)/pmodel.T 
                    - u_max*prc_spl(start_time+(phi)*pmodel.T/(2*np.pi)))
        int_times = np.linspace(0,step,101)
        phis = integrate.odeint(dphidt, [phi0], int_times, hmax=0.01)
        return phis[-1][0], phis[-1][0]-phi0-2*np.pi/pmodel.T*step
            
    
    # for the negative oscillator
    t_neg = np.arange(0,tf,step)
    u_neg = [0]
    phi_neg = [phi_osc_neg]
    phi_shift_neg = [0]
    for t in t_neg:
        phi_ctrl_on, delta_phi = step_integrate(phi_neg[-1], step)
        if delta_phi<0:
            u_neg.append(u_max)
            phi_neg.append(phi_ctrl_on)
            phi_shift_neg.append(delta_phi+phi_shift_neg[-1])
        else:
            u_neg.append(0)
            phi_neg.append(phi_neg[-1]+2*np.pi/pmodel.T*step)
            phi_shift_neg.append(phi_shift_neg[-1])
    u_neg = np.asarray(u_neg)[:-1]
    phi_neg = np.asarray(phi_neg)[:-1]
    phi_shift_neg = np.asarray(phi_shift_neg)[:-1]
    
    
    
    # for the positive oscillator
    t_pos = np.arange(0,tf,step)
    u_pos = [0]
    phi_pos = [phi_osc_pos]
    phi_shift_pos = [0]
    for t in t_pos:
        phi_ctrl_on, delta_phi = step_integrate(phi_pos[-1], step)
        if delta_phi>0:
            u_pos.append(u_max)
            phi_pos.append(phi_ctrl_on)
            phi_shift_pos.append(delta_phi+phi_shift_pos[-1])
        else:
            u_pos.append(0)
            phi_pos.append(phi_pos[-1]+2*np.pi/pmodel.T*step)
            phi_shift_pos.append(phi_shift_pos[-1])
    u_pos = np.asarray(u_pos)[:-1]
    phi_pos = np.asarray(phi_pos)[:-1]
    phi_shift_pos = np.asarray(phi_shift_pos)[:-1]
    return t_neg, u_neg, phi_neg, phi_shift_neg, u_pos, phi_pos, phi_shift_pos
    
    

            
            
            
def find_tstep_opt(phi0, step=2, discretization = 200):
    """ finds tstar and t_opt (at discretization) for a given phi0"""
    # define a periodic spline for the 
    delta_phis_total = delta_phis_step(phi0, 80, step=step, u_max=0.06)    
    times = delta_phis_total[0]
    
    # get the delta phis
    delta_phi_fs = np.linspace(0,2*np.pi,discretization)
    cross_loc =np.min(
         np.where(delta_phis_total[6]-delta_phis_total[3]>=2*np.pi))             
    t_star = times[cross_loc]
    phi_star_max = delta_phis_total[6][cross_loc]%(2*np.pi)
    phi_star_min = delta_phis_total[3][cross_loc]%(2*np.pi)
    
    t_opts = []
    for phif in delta_phi_fs:
        if phif > phi_star_max:
            time_to_reach = np.min(
                times[np.where(delta_phis_total[3]+2*np.pi<=phif)]
                )
        elif phif<phi_star_min:
            time_to_reach = np.min(
                times[np.where(delta_phis_total[6]>=phif)]
                )
        else:
            time_to_reach = t_star
        t_opts.append(time_to_reach)
    return delta_phi_fs, np.asarray(t_opts), t_star, phi_star_max, phi_star_min


def t_opts_calc(phi0s, step):
    """ calculates t_opts and returns them in a reasonable format """
    results = [find_tstep_opt(phi0, step=step) for phi0 in phi0s]
    delta_phi_fs = []
    t_opts = []
    t_stars = []
    phi_star_maxs = []
    phi_star_mins = []
    #recover the parts
    for result in results:
        delta_phi_fs.append(result[0])
        t_opts.append(result[1])
        t_stars.append(result[2])
        phi_star_maxs.append(result[3])
        phi_star_mins.append(result[4])
    
    delta_phi_fs = np.vstack(delta_phi_fs).T
    t_opts = np.vstack(t_opts).T
    return t_opts, delta_phi_fs
    


# choose the phi0s at which to get results
phi0s = np.linspace(0,2*np.pi,100)
phi0s_plt = np.vstack([phi0s]*200)
t_opts_4h, delta_phi_fs = t_opts_calc(phi0s, 4)


# load reference optimal
t_opts_ref = np.load('Data/t_opts.npy')

#get how l-infty norm, how tstar max changes
linfty = [0]
step_taus = np.arange(0.5,13,0.5)
max_tstar = [np.max(t_opts_ref)]
for step in step_taus:
    t_opts, delta_phi_fs = t_opts_calc(phi0s, step)
    linfty.append(np.max(t_opts-t_opts_ref))
    max_tstar.append(np.max(t_opts))
step_taus = np.hstack([[0], step_taus])




# plot results
plo.PlotOptions(uselatex=True, ticks='in')
plt.figure(figsize=(3.5,4.2))
gs = gridspec.GridSpec(4,2)

ax = plt.subplot(gs[0,:])
img = ax.pcolormesh(phi0s_plt, delta_phi_fs, t_opts_ref, label='Time (h)', 
                    cmap = 'inferno',vmax=60,vmin=0)
cbar = plt.colorbar(img)
img.set_zorder(-20)
cbar.set_label('$t^{opt}$ (h)')
ax.set_ylabel('$\Delta\phi_f$')
ax.set_rasterization_zorder(-10)
plo.format_2pi_axis(ax)
plo.format_4pi_axis(ax,x=False,y=True)
ax.set_ylim(0,2*np.pi)
ax.set_xticklabels([])

bx = plt.subplot(gs[1,:])
img = bx.pcolormesh(phi0s_plt, delta_phi_fs, t_opts_4h, label='Time (h)', 
                    cmap = 'inferno',vmax=60,vmin=0)
cbar = plt.colorbar(img)
img.set_zorder(-20)
cbar.set_label('$t^{opt}_{4h}$ (h)')
bx.set_ylabel('$\Delta\phi_f$')
bx.set_rasterization_zorder(-10)
plo.format_2pi_axis(bx)
plo.format_4pi_axis(bx,x=False,y=True)
bx.set_ylim(0,2*np.pi)
bx.set_xticklabels([])

cx = plt.subplot(gs[2,:])
img = cx.pcolormesh(phi0s_plt, delta_phi_fs, t_opts_4h-t_opts_ref, 
                    label='Lost Time (h)', 
                    cmap = 'Reds',vmin=0)
cbar = plt.colorbar(img)
img.set_zorder(-20)
cbar.set_label('$t^{opt}_{4h}-t^{opt}$ (h)')
cx.set_ylabel('$\Delta\phi_f$')
cx.set_xlabel('Initial Phase $\phi_0$')
cx.set_rasterization_zorder(-10)
plo.format_2pi_axis(cx)
plo.format_4pi_axis(cx,x=False,y=True)
cx.set_ylim(0,2*np.pi)

# L-infty norm
d1x = plt.subplot(gs[3,0])
d1x.plot(step_taus, linfty, 'k')
d1x.set_xlabel(r'$\tau$ (h)')
d1x.set_ylabel(r'L$_\infty$($t^{opt}-t^{opt}_\tau$)')
d1x.set_ylim([0,80])
d1x.set_xlim([0,12])

# tstar(all)
d2x = plt.subplot(gs[3,1])

d2x.plot(step_taus, max_tstar,'k')
d2x.set_xlabel(r'$\tau$ (h)')
d2x.set_ylabel(r'max($t^\star$($\phi_0$))')
d2x.set_ylim([55,80])
d2x.set_xlim([0,12])







plt.tight_layout(**plo.layout_pad)

plt.savefig("Data/fig2_rasterization.svg", dpi=900)



