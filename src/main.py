import numpy as np
import beam_position as b_p
import molecular_photophysics as m_p
import MLE_localization as mle_l
import plot_figures as p_f

fov_size=300  #in nm, FOV size
fov_step=1    #in nm, smallest step in the space
x = np.arange(-fov_size/2,fov_size/2,fov_step)
y = np.arange(-fov_size/2,fov_size/2,fov_step)
[xv,yv] = np.meshgrid(x,-y)  #FOV meshgrid
start_t = 0
dt = 0.1 # us , simulation time is 100 ns

sample_nos = 10         # Number of single molecule runs to be simulated

lamda = 750                 # wavelength, in nm
fwhm = 360                  # FWHM of the beam in nm
power_factor_list=[1,2,4,6] # Power factor which scalest with reduced L
L_list = [288,150,75,40]    # List of pattern size L for the different iteration
pattern = 'hexagon'         # Pattern beam numbers, center included default
if pattern == 'hexagon':
    K = 7                   # Total beam positions within the pattern
if pattern == 'triangle':
    K = 4

save_path = r'...'          # CHANGE to a valid folder path
# For each of the conditions, add/remove different conditions in the lists from which each parameter is changed
# in the loops. Possible conditions are :
# different fluorophores (change_fluorophore)- 'DL755Ab' and 'AF647Ab'
# different photophysical condition (change_photophys) - 'ALL' for photophysics rates determined for the dye attached to antibody in GODCAT+10mM MEA buffer
#                                                      - 'ISO' for hypothetical dye without Redox in GODCAT+10mM MEA
#                                                      - 'NO'  for no transient dark state population
#                                                      - 'ROXS' for photophysical rates determined for the dye attached to imager strand in balanced redox buffer (only for DL755)
# different beam dwell times at each point of the TCP (change_dwell) - 500 us, 150 us, 30 us, 5 us
# different pattern repeat numbers (change_repeat) - 1, 5
# different starting laser powers at itr1 (change_power) - 30 uW, 10 uW, 5 uW, 1 uW

for change_fluorophore in ['DL755Ab','AF647Ab']:
    for change_photophys in ['ALL','ISO','NO','ROXS']:
        rates = m_p.choose_dye(fluorophore=change_fluorophore,photophys_tag=change_photophys)
        print(rates)
        for change_dwell in [500,150,30,5]:
            for change_repeat in [1,5]:
                for change_power in [30,10,5,1]:
                    fluorophore = change_fluorophore
                    photophys_tag = change_photophys
                    start_power = change_power  # starting power for the largest L iteration, in uW
                    pattern_repeat = change_repeat
                    beam_dwell_time = change_dwell #total beam dwell at any given beam postion, irrrespective of pattern repeats
                    print('Dye : ' + fluorophore, ' , Photophys_tag : ' + photophys_tag)
                    print('Starting power : ' + str(change_power) + 'uW')
                    print('Pattern repeat :' + str(change_repeat))
                    print('Beam dwell time : '+str(change_dwell))
                    end_t = beam_dwell_time * K
                    t = np.arange(start_t, end_t, dt)
                    dye_pos_sample=[]
                    est_pos_sample=[]
                    photons_all_sample=[]
                    #pop_all_sample = []
                    for sample in range(sample_nos):
                            if sample % 10 == 0:
                                print(sample)
                            last_tcp_center = (0,0)  # The tcp center from the previous iteration, for itr 1, last_tcp_center is the estimated fluorophore position in pre localization
                            dye_pos=(1,1)      # CHANGE IF DIFFERENT DYE POSITION IS TO BE SIMULATED
                            #dye_pos= m_p.random_dye_pos_within_circle(50)
                            pre_pop_dye = [1,0,0,0]    # The photophysical state vector for the fluorophore with N,P,T,R. Assuming fluorophore relaxed back to lowest energy after preloc

                            psf_all  = []
                            beam_all =[]
                            k01_all  =[]
                            pop_all  =[]
                            photons_all=[]
                            est_all=[]

                            for l_idx,l in enumerate(L_list): #Going through individual iterations
                                #print('iteration = ' + str(l_idx))
                                #print('pattern size (L) = '+str(l)+' nm')

                                i_dye=[]        # intensity values the dye experience for each of the beam dwell times
                                psf = np.zeros((K,fov_size,fov_size))
                                beam_pos = b_p.beam_position(pattern, last_tcp_center, l)  # finding the beam centers for the specific iteration
                                for i in range(len(beam_pos)): #Going through individual beam positions within the iteration
                                    x0 = beam_pos[i][0]
                                    y0 = beam_pos[i][1]
                                    rv = np.sqrt((xv - x0) ** 2 + (yv - y0) ** 2)                   #Converting all cordinates into relative coordinates based on beam center
                                    rdye = np.sqrt((dye_pos[0] - x0) ** 2 + (dye_pos[1] - y0) ** 2) #Conversion of dye position

                                    psf[i,:,:] = b_p.donut(rv,power=start_power,power_factor=power_factor_list[l_idx],FWHM=fwhm)
                                    i_dye.append(b_p.donut(rdye,power=start_power,power_factor=power_factor_list[l_idx],FWHM=fwhm))

                                    beam_all.append((x0, y0))
                                    psf_all.append(psf[i,:,:])



                                k01_dye = m_p.I_to_k01(i_dye, sigma=rates['exc_sigma'], lamda=lamda) #Calculating corresponding k01 rate experienced by the dye
                                k01_dye_t = np.tile(np.repeat(k01_dye, int((beam_dwell_time/dt)/pattern_repeat), axis=0),pattern_repeat)

                                pop_dye = m_p.model_N_P_T_R(t, dt, k01_dye_t, rates,ensemble=False,initial_state=pre_pop_dye) # photophysical state evolution
                                pre_pop_dye = pop_dye[-1]   #Taking the last state to be given as the initial state for the next iteration

                                photons_dye = m_p.photons(pop_dye, rates, k01_dye_t,dt, dwell_time=beam_dwell_time, pattern_repeat = pattern_repeat)#Finding the integrated photons for the each beam position
                                est_ind, grid_size, tot_like = mle_l.loc_MLE(photons_dye, psf)
                                est_pos = (int((est_ind[1]*grid_size)-fov_size/2),int(-1*((est_ind[0]*grid_size)-fov_size/2))) #doing exactly what meshgrid (x,-y) will do, kindof
                                last_tcp_center = est_pos  # setting the pattern center for the next iteration on to the estimated position of the fluorophore

                                k01_all.append(k01_dye_t)
                                pop_all.append(pop_dye)
                                photons_all.append(photons_dye)
                                est_all.append(est_pos)

                            k01_all=[item for sublist in k01_all for item in sublist]
                            pop_all=[item for sublist in pop_all for item in sublist]
                            photons_all=[item for sublist in photons_all for item in sublist]
                            #est_all=[item for sublist in est_all for item in sublist]
                            #print(dye_pos,est_all,photons_all,photons_all.count(0.0))
                            dye_pos_sample.append(dye_pos)
                            est_pos_sample.append(est_all)
                            photons_all_sample.append(photons_all)
                            #pop_all_sample.append(pop_all)
                            #INCLUDE THE NEXT LINE IF you want to plot individual dynamics, similar to Figure 2 in main text
                            p_f.plot_figures(psf_all,K,L_list,fov_size,beam_all,dye_pos,beam_dwell_time,k01_all,pop_all,photons_all,est_all,dt)
                    #print(np.array(dye_pos_sample))
                    #print(np.array(est_pos_sample))
                    #print(np.array(photons_all_sample))

                    np.save(save_path + fluorophore +'_GLOX_10mM_MEA_dt0p1_4itr_startingpower' + str(start_power) + 'uW_patternrepeat'
                            + str(pattern_repeat) + '_dwell' + str(beam_dwell_time) + 'us_DYE_POS_SAMPLE_samples' + str(sample_nos)
                            + '_SM' + str(dye_pos[0]) + ',' + str(dye_pos[1]) + '_photophysics' + photophys_tag + '.npy',
                            np.array(dye_pos_sample))

                    np.save(save_path + fluorophore +'_GLOX_10mM_MEA_dt0p1_4itr_startingpower' + str(start_power) + 'uW_patternrepeat'
                            + str(pattern_repeat) + '_dwell' + str(beam_dwell_time) + 'us_EST_POS_SAMPLE_samples' + str(sample_nos)
                            + '_SM' + str(dye_pos[0]) + ',' + str(dye_pos[1]) + '_photophysics' + photophys_tag + '.npy',
                            np.array(est_pos_sample))

                    np.save(save_path + fluorophore +'_GLOX_10mM_MEA_dt0p1_4itr_startingpower' + str(start_power) + 'uW_patternrepeat'
                            + str(pattern_repeat) + '_dwell' + str(beam_dwell_time) + 'us_PHOTONS_ALL_SAMPLE_samples' + str(sample_nos)
                            + '_SM' + str(dye_pos[0]) + ',' + str(dye_pos[1]) + '_photophysics' + photophys_tag + '.npy',
                            np.array(photons_all_sample))
                    # np.save(save_path + fluorophore + '_GLOX_10mM_MEA_dt0p1_4itr_startingpower' + str(start_power) + 'uW_patternrepeat'
                    #         + str(pattern_repeat) + '_dwell' + str(beam_dwell_time) + 'us_POP_ALL_SAMPLE_samples' + str(sample_nos)
                    #         + '_SM' + str(dye_pos[0]) + ',' + str(dye_pos[1]) + '_photophysics' + photophys_tag + '.npy',
                    #         np.array(pop_all_sample))


