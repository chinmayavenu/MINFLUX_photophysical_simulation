import matplotlib.pyplot as plt
import numpy as np
import molecular_photophysics as m_p
rng = np.random.default_rng()

rates = { 'exc_sigma':8.404,'lifetime':0.57,'quantum_yield':0.119, 'kiso': 17, 'cs_biso': 0.119, 'ktherm': 0.007,
                      'kisc': 2.08, 'kt': 0.174, 'kred': 0.0609, 'kox': 0.000073, 'Q': 1}  # DL755.Ab GLOX 10mM MEA

K=7
start_t = 0
beam_dwell_time = 150
end_t = beam_dwell_time*K
dt = 0.1
pattern_repeat = 1
t = np.arange(start_t,end_t,dt)
k01_t = np.empty(np.shape(t)[0])
k01_t.fill(1)

ensu_N=[]
ensu_P=[]
ensu_T=[]
ensu_R=[]
plt.figure('ensemble',figsize=(12,8))
for exp in range(1):
    pop_ind = m_p.model_N_P_T_R(t,dt,k01_t,rates,ensemble=True,initial_state=[1,0,0,0])
    ensu_N.append(pop_ind[:,0])
    ensu_P.append(pop_ind[:, 1])
    ensu_T.append(pop_ind[:, 2])
    ensu_R.append(pop_ind[:, 3])

plt.plot(t,[sum(i) for i in zip(*ensu_N)],lw=6,label='Trans_Ensemble')
plt.plot(t,[sum(i) for i in zip(*ensu_P)],lw=6,label='Cis_Ensemble')
plt.plot(t,[sum(i) for i in zip(*ensu_T)],lw=6,label='Triplet_Ensemble')
plt.plot(t,[sum(i) for i in zip(*ensu_R)],lw=6,label='Redox_Ensemble')


ensu_N=[]
ensu_P=[]
ensu_T=[]
ensu_R=[]
samples=100
for exp in range(samples):
    pop_ind = m_p.model_N_P_T_R(t,dt,k01_t,rates,ensemble=False,initial_state=[1,0,0,0])
    ensu_N.append(pop_ind[:,0])
    ensu_P.append(pop_ind[:, 1])
    ensu_T.append(pop_ind[:, 2])
    ensu_R.append(pop_ind[:, 3])

plt.plot(t,[sum(i)/samples for i in zip(*ensu_N)],lw=3,label='Avg.Trans')
plt.plot(t,[sum(i)/samples for i in zip(*ensu_P)],lw=3,label='Avg.Cis')
plt.plot(t,[sum(i)/samples for i in zip(*ensu_T)],lw=3,label='Avg.Triplet')
plt.plot(t,[sum(i)/samples for i in zip(*ensu_R)],lw=3,label='Avg.Redox')

plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel('Time ($\u03bcs$)', fontsize=30)
plt.ylabel('Population', fontsize=30)
ax = plt.gca()
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2)
ax.tick_params(axis='both', width=5, size=5)
plt.tight_layout()
plt.legend(ncol=2,fontsize=20)


plt.figure('SM',figsize=(10,6))
ensu_N=[]
ensu_P=[]
ensu_T=[]
ensu_R=[]
for exp in range(3):
    pop_ind = m_p.model_N_P_T_R(t, dt, k01_t, rates, ensemble=False,initial_state=[1, 0, 0, 0])
    ensu_N.append(pop_ind[:,0])
    ensu_P.append(pop_ind[:, 1])
    ensu_T.append(pop_ind[:, 2])
    ensu_R.append(pop_ind[:, 3])
    pop_dye= np.where(pop_ind==1)[1]
    plt.scatter(t,pop_dye+(exp*5),label= 'Fluorophore '+str(exp+1))
    plt.yticks(ticks=range(15),labels= ['N','P','T','R','']*3,fontsize=25)
    plt.xticks(fontsize=25)
    plt.ylim(-1,15)
    plt.xlabel('Time ($\u03bcs$)', fontsize=30)
    plt.ylabel('Photophysical state', fontsize=30)
    ax = plt.gca()
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    ax.tick_params(axis='both', width=5, size=5)
    plt.tight_layout()
    plt.legend(fontsize=16)
plt.show()
