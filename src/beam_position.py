import numpy as np

# returns Intensity in kW/cm^2 at the radial position r,given measured power in uW and power_factor which is the power multiplier for iterative MINFLUX
#power in uW, FWHM in nm, so to make Intensity is in uW/nm^2 = 10^-9 kW/ 10^-14 cm^2 = 10^5 kW/cm^2, hence multipling the intesity with 1e5 to get the intensity in kW/cm2
#Equation in supplementary materials, section 2, eq S17,S19, Balzarotti_et_al,Science,2017

def gaussian(r,power=1,power_factor=1,FWHM =360):
#f(r') = A.exp(-2*(r'-r0)^2/w0^2); if r= r'-r0,ie r is the relative radial cordinate with respect to the beam center r0, then f(r) = A.exp(-2r^2/w0^2)
# FWHM = sqrt(2.ln2)*w0, which implies std^2 = FHWM^2/2.ln2.
#subtituting, we get f(r) = A.exp(-4.ln2.r^2/FWHM^2)
#for guassian beam I0 = 2P0/pi.w0^2, w0 = FWHM/sqrt(2.ln2), thus I0 = 2P0*2.ln2/pi.FWHM^2
    return power_factor * ((4*power*np.log(2))/(np.pi*(FWHM**2))*1e5) * np.exp((-4*np.log(2)*((r)**2))/(FWHM**2))

def donut(r,power=1,power_factor=1,FWHM=360):
#similar deal about FHWM and beam radius calculations, along with power calculations with f(r) = A.(2r^2/w0^2). exp(1-2r^2/w0^2)
    return power_factor * ((4*power*np.log(2))/(np.pi*(FWHM**2))*1e5) * (4*np.log(2)*((r**2)/(FWHM**2)))* np.exp(1-(4*np.log(2)*((r ** 2)/(FWHM ** 2))))


# Takes in the pattern center position (which is decided based on est_fluorphore_position), pattern size L and the pattern, to give beam center positions for the EOD move
#Equation in supplementary materials, section 2.2, eq S24, Balzarotti_et_al,Science,2017
def beam_position(pattern='hexagon',pattern_center=(0,0),l=288):
    phi= -2*np.pi/12 # small offset to get the beam to go down vertically first instead of an angle
    if pattern == 'hexagon':
        ex_pos=[]

        ex_pos.append(pattern_center)
        for k in np.arange(1,7):
                theta = 2*np.pi* -k / 6 + phi
                ex_pos.append((((l / 2) * np.cos(theta))+pattern_center[0],((l / 2) * np.sin(theta))+pattern_center[1]))
    if pattern == 'triangle':
        ex_pos = []

        ex_pos.append(pattern_center)
        for k in np.arange(1, 4):
                theta = 2 * np.pi * -k / 3 + phi
                ex_pos.append((((l / 2) * np.cos(theta))+pattern_center[0],((l / 2) * np.sin(theta))+pattern_center[1]))
    return ex_pos




