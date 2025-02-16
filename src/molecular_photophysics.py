import numpy as np
import scipy.linalg as lin  #for ensemble differential equation solution


rng = np.random.default_rng() #random seed generator for single molecule multinomial sampling

def choose_dye(fluorophore,photophys_tag):
    if fluorophore == 'AF647Ab' :
        if photophys_tag == 'ALL':
            rates = { 'exc_sigma': 6.4, 'lifetime': 1,'quantum_yield': 0.33, 'kiso': 15.1, 'cs_biso': 0.0743, 'ktherm': 0.0005,
                      'kisc': 0.248, 'kt': 3.03, 'kred': 0.28, 'kox': 0.00017, 'Q': 1}  # AF647.Ab GLOX 10mM MEA
        elif photophys_tag == 'ISO':
            rates = {'exc_sigma': 6.4, 'lifetime': 1, 'quantum_yield': 0.33, 'kiso': 15.1, 'cs_biso': 0.0743, 'ktherm': 0.0005,
                     'kisc': 0, 'kt': 0, 'kred': 0, 'kox': 0, 'Q': 1}  # AF647.Ab GLOX 10mM MEA
        elif photophys_tag == 'NO':
            rates = {'exc_sigma': 6.4, 'lifetime': 1, 'quantum_yield': 0.33, 'kiso': 0, 'cs_biso': 0, 'ktherm': 0,
                     'kisc': 0, 'kt': 0, 'kred': 0, 'kox': 0, 'Q': 1}  # AF647.Ab GLOX 10mM MEA
        else:
            rates = {}
            print('Wrong fluorophore tag for the fluorophore choosen')
    elif fluorophore == 'DL755Ab' :
        if photophys_tag == 'ALL':
            rates = { 'exc_sigma':8.404,'lifetime':0.57,'quantum_yield':0.119, 'kiso': 17, 'cs_biso': 0.119, 'ktherm': 0.007,
                      'kisc': 2.08, 'kt': 0.174, 'kred': 0.0609, 'kox': 0.000073, 'Q': 1}  # DL755.Ab GLOX 10mM MEA
        elif photophys_tag == 'ISO':
            rates = {'exc_sigma': 8.404, 'lifetime': 0.57, 'quantum_yield': 0.119, 'kiso': 17, 'cs_biso': 0.119,'ktherm': 0.007,
                     'kisc': 0, 'kt': 0, 'kred': 0, 'kox': 0, 'Q': 1}  # DL755.Ab GLOX 10mM MEA
        elif photophys_tag == 'NO':
            rates = {'exc_sigma': 8.404, 'lifetime': 0.57, 'quantum_yield': 0.119, 'kiso': 0, 'cs_biso': 0,'ktherm': 0,
                     'kisc': 0, 'kt': 0, 'kred': 0, 'kox': 0, 'Q': 1}  # DL755.Ab GLOX 10mM MEA
        elif photophys_tag == 'ROXS':
            rates = {'exc_sigma': 8.404, 'lifetime': 1.14, 'quantum_yield': 0.119, 'kiso': 12.2, 'cs_biso': 0.124, 'ktherm': 0.0001,
                     'kisc': 0.3, 'kt': 0.02, 'kred': 0, 'kox': 0, 'Q': 1}  # DL755F1 GLOX ROXS D20 20mM Mg
        else:
            rates = {}
            print('Wrong fluorophore tag for the fluorophore choosen')
    else:
        print('Wrong fluorophore selected')
        rates={}
    return rates



def I_to_k01(intensity, sigma, lamda):
    intensity = np.array(intensity)* 1e7    # in kW/cm^2 Intensity from excitation at a certain location,multiply by 1e7 to get W/m^2
    sigma = sigma * 1e-16 * 1e-4            # in cm^2, multiply by 1e-4 for m^2
    lamda = lamda * 1e-9                    # in nm, multiply by 1e-9 for m
    h = 6.626e-34                           # in J.s
    c = 3e8                                 # in m/s

    k01_mag = (sigma * intensity * lamda) / (h * c)  # in 1/second ( W/m2 * m2 * m * 1/J.s * s/m ~~ J/s * 1/J ~~ 1/s)
    # print(intensity,sigma,lamda,h,c)
    return k01_mag * 1e-6  # in 1/microsecond


def model_N_P_T_R( t, dt, k01_t, rates, ensemble = True, initial_state=[1,0,0,0]):

    k01 = k01_t                                      # in us^-1, k01 value for each sampling time for the duration of simulation t
    k10 =1 / (rates['lifetime'] * 1e-3)              # in us^-1, k10 total singlet excited state depopulation rate
    exc = k01 / (k01 + k10)                          # considering singlets has equilibriated, fraction of molecules in S1 state which contribute to photoinduced processes
    kiso =  rates['kiso']                            # in us^-1, absolute isomerization rate from S1 of N (trans) to P (cis)
    kiso_p = kiso*exc                                # in us^-1, excitation dependent isomerization rate from N to P at each sampling time
    ktherm = rates['ktherm']                         # in us^-1, thermal backisomerization rate from P to N
    kbiso_p = rates['cs_biso'] * (k01 / rates['exc_sigma']) # in us^-1, photoinduced backisomerization rate from P to N at each sampling time
    kisc = rates['kisc']                             # in us^-1, absolute intersystem crossing rate from S1 to triplet (T) within trans
    kisc_p = kisc*exc                                # in us^-1, excitation dependent intersysem crossing rate from N to T at each sampling time
    kt = rates['kt']                                 # in us^-1, triplet relaxation rate
    kred= rates['kred']                              # in us^-1, reduction rate from T to redox (R) state
    kox=rates['kox']                                 # in us^-1, oxidation rate from R to N


    #print(np.shape(k01),np.shape(k10),np.shape(kiso),np.shape(kbiso))# kbiso=k01*(cbiso/6.2) #6.2 is absorption cross-section

    St = np.zeros((np.shape(k01)[0],4))             # State matrix for N(trans), P(cis), T(triplet), R(redox) at each sampling time
    S0 = initial_state                              # Intitial state, this can be different than [1,0,0,0] for iterative MFX where the system did not relax completely back
    S_ = S0
    if ensemble:
        #print('ensemble')
        for i, time in enumerate(t):                    # Checking state evolution between each sampling time

            #print('time = '+ str(time))
            if i!= 0:
                S_ = St[i-1,:]   #starting states in this iteration is the state determined in the last iteration. Starting state is initial state in case of iteration 0

            ## Matrix representing the coefficients and terms of differential equations for the time evolution of states N,P,T,R
            M = np.array([[-1*(kiso_p[i]+kisc_p[i]),     (kbiso_p[i]+ktherm)  ,        kt       ,   kox ] ,
                          [    kiso_p[i]           ,  -1*(kbiso_p[i]+ktherm)  ,         0       ,   0   ],
                          [    kisc_p[i]           ,             0            ,  -1*(kt+kred)   ,   0   ],
                          [        0               ,             0            ,       kred      ,  -kox ]])


            x = lin.expm(M * dt)   # matrix exponent
            St[i, :] = np.dot(x, S_)

    else:
        for i, time in enumerate(t):

            if i!= 0:
                S_ = St[i-1,:]
            ## right Markov Matrix representing the probabilities of switching to states N,P,T,R (columns) from the current state N,P,T,R (rows)
            M = np.array([[1-((kiso_p[i]+kisc_p[i])*dt),        kiso_p[i]*dt          ,     kisc_p[i]*dt     ,     0       ] ,
                          [  (kbiso_p[i]+ktherm)*dt    ,  1-((kbiso_p[i]+ktherm)*dt)  ,           0          ,     0       ],
                          [            kt*dt           ,               0              ,  1- ((kt+kred)*dt)   ,   kred*dt    ],
                          [            kox*dt          ,               0              ,           0          ,   1-(kox*dt) ]])

            curr_prob = np.dot(S_,M) # probability row corresponding to the current state of the single molecule
            St[i, :] = rng.multinomial(1, curr_prob, size=1)[0] #selecting a next state using a random draw from a multinomial function with the current probabilities


    return St



def photons(pop,rates,k01_t,dt,dwell_time=166,pattern_repeat=1): ##Dwell_time here is dwell time /dt

    fluorescence_qy = rates['quantum_yield']  # fluorscence quatum yield of the fluorophore (k10_f/k01_f+k01_nr)
    k10 = 1/(rates['lifetime']*1e-3)          # total depopulation rate k10 = k01_f+k01_nr = 1/lifetime
    k01 = k01_t
    S1_pop = pop[:,0]*(k01/(k01+k10))*dt        # population of S1 population of the trans state calculated from the N population for each sampling time, since rates are in us-1, we need to multiply dt to get population in sampling time dt
    P_pop  = pop[:,1]                        # population of dimly fluorescent cis isomer
    Q = rates['Q']                           # ratio of brightness between dim fluorescent second cis isomer and the fluorescent trans

    F =  (k10*fluorescence_qy*S1_pop)# + Q* P_pop  # instantaneous fluorescence photons at each sampling time
    F_rep = F.reshape(pattern_repeat,-1)
    F_add = np.add.reduceat(F_rep, np.arange(0, np.shape(F_rep)[0],np.shape(F_rep)[0]),axis=0)
    return np.add.reduceat(F_add, np.arange(0, np.shape(F_add)[1],int((dwell_time/dt)/pattern_repeat)),axis=1)[0]


def random_dye_pos_within_circle(R=50):

    theta = np.random.uniform(0, 2 * np.pi, 1)
    radius = R * (np.random.uniform(0, 1, 1) ** 0.5)

    x_ = radius * np.cos(theta)
    y_ = radius * np.sin(theta)

    return (int(x_[0]), int(y_[0]))


