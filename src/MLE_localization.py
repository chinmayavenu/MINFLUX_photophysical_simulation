import numpy as np
def loc_MLE(photons, psf_list) :

    K = np.shape(psf_list)[0]  ## Number of photon values
    splice = 1                 ## This is for reduce computation if the grid search for MLE need not be 1nm (as fine as the space pixel, splice two means, MLE with 2nm jump
    intens_field = psf_list[:, ::splice, ::splice] ## Basically intensity at all possible dye positions
    parameter = np.zeros_like(intens_field) ## Parameter function

    for ii in range(np.shape(intens_field)[1]):
        for jj in range(np.shape(intens_field)[2]):
            # parameter value at any position is ratio of intensity at given k with sum of intensity for all k
            # Equation in supplementary materials, section 1, eq S4, Balzarotti_et_al,Science,2017
            parameter[:, ii, jj] = (intens_field[:, ii, jj]) / np.sum(intens_field[:, ii, jj], axis=0) # Calculation of parameter at all possible dye positions

    log_likely = np.zeros_like(intens_field)

    for i in range(K):
        log_likely[i,:,:] = photons[i] * np.log(parameter[i,:,:])
    # Equation in supplementary materials, section 3.1.2, eq S37, Balzarotti_et_al,Science,2017
    total_likely = np.sum(log_likely,axis=0) # Log likelyhood function value to be maximized
    # argmax for the likelihood function
    mle_index = np.unravel_index(np.argmax(total_likely, axis=None),
                                 total_likely.shape)

    return(mle_index,splice,total_likely)

