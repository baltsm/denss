#plot_fit

import saxstats.saxstats as saxs
import saxstats.denssopts as dopts
import numpy as np
import sys, argparse, os
import matplotlib.pyplot as plt
from  matplotlib.colors import colorConverter as cc
import matplotlib.gridspec as gridspec

output = "loop_SASDRX8_map"
q,I,err,ifit,results = saxs.loadFitFile("/Users/maijabalts/Desktop/BioXFEL/denss-master/Loop_map.fit")

# qraw = args.qraw
# Iraw = args.Iraw
# q = args.q
# I = args.I
# sigq = args.sigq

f = plt.figure(figsize=[6,6])
gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])

ax0 = plt.subplot(gs[0])
ax0.errorbar(q, I, fmt='k.', yerr=err, mec='none', mew=0, ms=5, alpha=0.3, capsize=0, elinewidth=0.1, ecolor=cc.to_rgba('0',alpha=0.5),label='Supplied Data',zorder=-1)
ax0.plot(q,ifit,'r-',label="DENSS Map") #label=r'DENSS Map $\chi^2 = %.2f$'%final_chi2)

handles,labels = ax0.get_legend_handles_labels()
handles = [handles[1], handles[0] ]
labels = [labels[1], labels[0] ]
ax0.legend(handles,labels)
ax0.semilogy()
ax0.set_ylabel('I(q)')

ax1 = plt.subplot(gs[1])
ax1.plot(q, I*0, 'k--')
residuals = (I-ifit)/err
ax1.plot(q, residuals, 'r.')
ylim = ax1.get_ylim()
ymax = np.max(np.abs(ylim))
ymax = np.max(np.abs(residuals))
ax1.set_ylim([-ymax,ymax])
ax1.yaxis.major.locator.set_params(nbins=5)
xlim = ax0.get_xlim()
ax1.set_xlim(xlim)
ax1.set_ylabel(r'$\Delta{I}/\sigma$')
ax1.set_xlabel(r'q ($\mathrm{\AA^{-1}}$)')
plt.tight_layout()
plt.savefig(output+'_fit.png',dpi=150)
plt.close()