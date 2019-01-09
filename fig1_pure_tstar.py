"""
j.h.abel 19/7/2016

adjusting single-step for multi-step optimization
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

# # plots to see what the portions look like
#plo.PlotOptions()
#plt.figure()
#plt.plot(times, prc, color='k', label='PRC')
#plt.plot(times, prc_pos, color='h', ls='--', label='PRC+')
#plt.plot(times, prc_neg, color='fl', ls='--', label='PRC-')
#plt.legend()
#plt.tight_layout(**plo.layout_pad)

prc_pos_spl = ut.PeriodicSpline(times, prc_pos, period=pmodel.T)
prc_neg_spl = ut.PeriodicSpline(times, prc_neg, period=pmodel.T)

def dphitot_dt(phis, t):
    [phi_osc_pos, phi_osc_neg, phi_shift_pos, phi_shift_neg] = phis
    dphi_osc_pos_dt = (2*np.pi)/pmodel.T + umax*prc_pos_spl(phi_osc_pos*pmodel.T/(2*np.pi))
    dphi_osc_neg_dt = (2*np.pi)/pmodel.T + umax*prc_neg_spl(phi_osc_neg*pmodel.T/(2*np.pi))
    dphi_shft_pos_dt = umax*prc_pos_spl(phi_osc_pos*pmodel.T/(2*np.pi))
    dphi_shft_neg_dt = umax*prc_neg_spl(phi_osc_neg*pmodel.T/(2*np.pi))
    return dphi_osc_pos_dt, dphi_osc_neg_dt, dphi_shft_pos_dt, dphi_shft_neg_dt
    
    
int_times = np.linspace(0,3*pmodel.T, 10001)

delta_phis_total = integrate.odeint(dphitot_dt, [0,0,0,0], int_times,
                                    hmax=0.001)
time_max = int_times[np.min(
            np.where(delta_phis_total[:,2]-delta_phis_total[:,3]>2*np.pi)
            )]


def find_tstar(phi0, discretization = 200):
    """ finds tstar and t_opt (at discretization) for a given phi0"""
    # define a periodic spline for the 
    start_time = pos_start_root+phi0*23.7/(2*np.pi)
    prc_pos = stats.threshold(-pmodel.pPRC_interp(times+start_time)[:, 15],
                            threshmin=0)
    prc_neg = stats.threshold(-pmodel.pPRC_interp(times+start_time)[:, 15],
                            threshmax=0)
    prc = -pmodel.pPRC_interp(times+start_time)[:, 15]
    
    prc_pos_spl = ut.PeriodicSpline(times, prc_pos, period=pmodel.T)
    prc_neg_spl = ut.PeriodicSpline(times, prc_neg, period=pmodel.T)
    
    def dphitot_dt(phis, t): # integrates everything
        [phi_osc_pos, phi_osc_neg, phi_shift_pos, phi_shift_neg] = phis
        dphi_osc_pos_dt = (2*np.pi)/pmodel.T +\
                        umax*prc_pos_spl(phi_osc_pos*pmodel.T/(2*np.pi))
        dphi_osc_neg_dt = (2*np.pi)/pmodel.T +\
                        umax*prc_neg_spl(phi_osc_neg*pmodel.T/(2*np.pi))
        dphi_shft_pos_dt = umax*prc_pos_spl(phi_osc_pos*pmodel.T/(2*np.pi))
        dphi_shft_neg_dt = umax*prc_neg_spl(phi_osc_neg*pmodel.T/(2*np.pi))
        return (dphi_osc_pos_dt, dphi_osc_neg_dt, 
                dphi_shft_pos_dt, dphi_shft_neg_dt)
        
        
    int_times = np.linspace(0,3*pmodel.T, 10001)
    
    delta_phis_total = integrate.odeint(dphitot_dt, [0,0,0,0], int_times,
                                        hmax=0.001)
    
    # get the delta phis
    delta_phi_fs = np.linspace(0,2*np.pi,discretization)
    cross_loc =np.min(
         np.where(delta_phis_total[:,2]-delta_phis_total[:,3]>=2*np.pi))             
    t_star = int_times[cross_loc]
    phi_star = delta_phis_total[cross_loc,3]%(2*np.pi)
    
    t_opts = []
    for phif in delta_phi_fs:
        if phif > phi_star:
            time_to_reach = np.min(
                int_times[np.where(delta_phis_total[:,3]+2*np.pi<=phif)]
                )
        else:
            time_to_reach = np.min(
                int_times[np.where(delta_phis_total[:,2]>=phif)]
                )
        t_opts.append(time_to_reach)
    return delta_phi_fs, np.asarray(t_opts), t_star, phi_star

# choose the phi0s at which to get results
phi0s = np.linspace(0,2*np.pi,100)
results = [find_tstar(phi0) for phi0 in phi0s]

delta_phi_fs = []
t_opts = []
t_stars = []
phi_stars = []
#recover the parts
for result in results:
    delta_phi_fs.append(result[0])
    t_opts.append(result[1])
    t_stars.append(result[2])
    phi_stars.append(result[3])

delta_phi_fs = np.vstack(delta_phi_fs).T
t_opts = np.vstack(t_opts).T
phi0s_plt = np.vstack([phi0s]*200)
np.save('Data/t_opts.npy', t_opts)


palatte = colorbrewer.get_map('Dark2','Qualitative',3)
colors = palatte.mpl_colors

plo.PlotOptions(uselatex=True, ticks='in')
plt.figure(figsize=(3.5,2.85))
gs = gridspec.GridSpec(3,2, height_ratios=(1,0.5,0.5), width_ratios = (1.5,1))

ax = plt.subplot(gs[0,0])
ts = np.arange(0,23.7,0.01)
ax.plot(times*np.pi/11.85, prc, color='k')
ax.set_ylabel('ipPRC$(\phi)$')
ax.set_xlabel('Phase $\phi$')
plo.format_2pi_axis(ax)

bx = plt.subplot(gs[0,1])
bx.plot(int_times, delta_phis_total[:,3]+2*np.pi,
        color=colors[0], ls=':', label = '$\Delta\phi^-$')
bx.plot(int_times, delta_phis_total[:,2], color=colors[1], ls='--',
        label = '$\Delta\phi^+$')
plo.format_4pi_axis(bx, x=False, y=True)
bx.set_xlim([0,72])
#bx.plot(time_max, np.pi/2-0.27, 'ko')
bx.annotate('$t^\star,\Delta\phi^\star$',xy=[time_max, np.pi/2-0.27],
            xytext=[time_max,2.5],
            arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", 
                            color='k'))
bx.set_ylim([0,2*np.pi+0.1])
bx.set_ylabel('$\Delta\phi$')
bx.set_xticklabels([])
bx.legend()



d1x = plt.subplot(gs[1,1])
d2x = plt.subplot(gs[2,1])
d3x = d1x.twinx()
d4x = d2x.twinx()
u_positive = umax*(prc_pos_spl(delta_phis_total[:,0]*23.7/(2*np.pi)) > 1E-6)
u_negative = umax*(prc_neg_spl(delta_phis_total[:,1]*23.7/(2*np.pi)) < -1E-6)


d1x.plot(int_times, u_negative, color=colors[0], ls=':')
d3x.plot(int_times, -prc_spl(start_time+delta_phis_total[:,1]*23.7/(2*np.pi)),
         'k')
d2x.plot(int_times, u_positive, color=colors[1], ls='--')
d4x.plot(int_times, -prc_spl(start_time+delta_phis_total[:,0]*23.7/(2*np.pi)),
         'k')
d2x.set_ylim([0,0.065])
d2x.set_xlim([0,72])
d2x.set_xlabel('Time (h)')
d2x.set_ylabel('$u^+(t)$')
d1x.set_ylim([0,0.065])
d1x.set_xlim([0,72])
d1x.set_ylabel('$u^-(t)$')
d1x.set_xticklabels([])

plt.tight_layout(**plo.layout_pad)

plt.savefig("Data/fig1_rasterization.svg", dpi=900)


