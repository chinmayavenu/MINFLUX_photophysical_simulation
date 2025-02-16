import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolor
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mcolor.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

new_cmap= truncate_colormap(plt.get_cmap('viridis'),0,1,100)

def plot_figures(psf_all,K,L_list,fov_size,beam_all,dye_pos,beam_dwell_time,k01_all,pop_all,photons_all,est_all,dt):

    fig,axes=plt.subplots(len(L_list),K,figsize=(10, 5),gridspec_kw={'hspace':0.1,'wspace':0.1})
    fig.canvas.manager.set_window_title('Beam_positions')

    mmax=np.max(psf_all)
    mmin=np.min(psf_all)
    for i, ax in enumerate(axes.flat):
        j= int(i / K)
        im= ax.imshow(psf_all[i][:,:] ,cmap=new_cmap,vmax=mmax,vmin=mmin,extent = [-fov_size/2,fov_size/2,-fov_size/2,fov_size/2])
        #ax.scatter(beam_all[i%K][0],beam_all[i%K][1],c='k')
        ax.scatter(dye_pos[0], dye_pos[1],facecolors='gold', edgecolors='k', marker='*', s=100)
        ax.scatter(beam_all[i][0], beam_all[i][1], facecolors='red', marker='o', s=10)
        ax.set_xticks([])
        ax.set_yticks([])


    #plt.colorbar()
    fig.subplots_adjust(right=0.8)
    # put colorbar at desire position
    cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.70])
    cbar= fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(label= '$I\ (kW/cm^2)$', size=30)
    plt.tight_layout()

    #fig=plt.figure('k01',figsize=(10,6)) #enseumble
    fig = plt.figure('k_01', figsize=(10, 6)) #SM
    plt.scatter(range(1, int(beam_dwell_time) * K * len(L_list) + 1, 1), k01_all[::10],
                label='$K_{01}$ at dye position ($\u03bcs^{-1}$)')
    plt.xlabel('Time ($\u03bcs$)', fontsize=30)
    plt.ylabel('$K_{01}$ ($\u03bcs^{-1}$)', fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    #plt.ylim(0,2)
    #plt.xticks(range(1, int(beam_dwell_time / dt) * K * len(L_list) + 1, 1))
    plt.tight_layout()
    plt.legend()
    axs = fig.gca()
    for axis in ['top', 'bottom', 'left', 'right']:
                axs.spines[axis].set_linewidth(2)
    ax = plt.gca()
    ax.tick_params(axis='both', width=5, size=5)
    #plt.savefig(savepath + '_K01_big.tif', bbox_inches='tight', dpi=300)
    #plt.savefig(savepath + '_K01_big.svg', bbox_inches='tight', dpi=300)

    #fig=plt.figure('Evolution of State population',figsize=(10,6)) #ensemble
    fig = plt.figure('Evolution of State population', figsize=(10, 6)) #SM
    axs = fig.gca()
    for axis in ['top', 'bottom', 'left', 'right']:
                axs.spines[axis].set_linewidth(2)
    plt.scatter(range(1, int(beam_dwell_time) * K * len(L_list) + 1, 1), [item[0] for item in pop_all][::10],
                label='Trans')
    plt.scatter(range(1, int(beam_dwell_time) * K * len(L_list) + 1, 1), [item[1] for item in pop_all][::10],
                label='Cis')
    plt.scatter(range(1, int(beam_dwell_time) * K * len(L_list) + 1, 1), [item[2] for item in pop_all][::10],
                label='Triplet')
    plt.scatter(range(1, int(beam_dwell_time) * K * len(L_list) + 1, 1), [item[3] for item in pop_all][::10],
                label='Redox')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel('Time ($\u03bcs$)', fontsize=30)
    plt.ylabel('Photophysical state', fontsize=30)
    ax = plt.gca()
    ax.tick_params(axis='both', width=5,size=5)
    plt.tight_layout()
    plt.legend()
    # plt.savefig(savepath + '_POP_big.tif', bbox_inches='tight', dpi=300)
    # plt.savefig(savepath + '_POP_big.svg', bbox_inches='tight', dpi=300)

    # ax3.set_title('S1 normalized for k01')
    # ax3.plot(range(1, beam_dwell_time *K*len(L_list)+1,1),pop_all,label='S1 pop')
    # ax3.set_xlabel('time ( x '+str(dt)+' us)',fontsize=10)
    # ax3.set_ylabel('Normalized S1 population',fontsize=10)
    # #plt.legend()

    #fig=plt.figure('photons',figsize=(10,5.88)) #last approved_ensemble

    fig=plt.figure('Photons',figsize=(10,6)) #last approved_SM
    axs = fig.gca()
    for axis in ['top', 'bottom', 'left', 'right']:
                axs.spines[axis].set_linewidth(2)
    plt.scatter(range(1, K * len(L_list) + 1, 1), photons_all,s=100, label='photons')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel('Beam position', fontsize=30)
    plt.ylabel('Photons', fontsize=30)
    ax = plt.gca()
    ax.tick_params(axis='both', width=5,size=5)
    plt.tight_layout()


    fig=plt.figure('Evolution of State population-SM',figsize=(10,6))
    axs = fig.gca()
    for axis in ['top', 'bottom', 'left', 'right']:
                axs.spines[axis].set_linewidth(2)
    pop_index = []
    for item in pop_all[::10]:
        pop_index.append(np.where(item==1)[0])

    plt.scatter(range(len(pop_index)),pop_index)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel('Time ($\u03bcs$)', fontsize=30)
    plt.ylabel('Photophysical state', fontsize=30)
    ax = plt.gca()
    ax.tick_params(axis='both', width=5,size=5)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['N', 'P', 'T', 'R'])  # N - Trans, P - Cis, T - Triplet, R - Redox
    plt.tight_layout()
    plt.legend()
    plt.show()
