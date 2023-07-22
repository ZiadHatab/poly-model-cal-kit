import numpy as np
import skrf as rf
import matplotlib.pyplot as plt


def open_standard(f, cap=[0], Z0=50, ereff=1, l=0):
    """
        modeling an open standard as polynomial capacitor with an offset

    parameters
    -----------
    f:  frequency array in Hz
    cap: polynomial coefficients in increasing order
    Z0: ref impedance; can be array of same size as frequency
    ereff: relative effective permittivity; can be array of same size as frequency
    l: length of the offset
    """

    c0 = 299792458   # speed of light in vacuum (m/s)
    cap = np.atleast_1d(cap)
    f = np.atleast_1d(f)
    Z0 = np.atleast_1d(Z0)*np.ones(len(f))

    C = np.array([(ff**np.arange(len(cap))*cap).sum() for ff in f])  # polynomial model 

    omega = 2*np.pi*f
    open_reflection = (1 - 1j*omega*C*Z0)/(1 + 1j*omega*C*Z0)    # the reflection

    gamma = 2*np.pi*f/c0*np.sqrt(-ereff*(1+0.0j))
    offset_open_reflection = open_reflection*np.exp(-2*gamma*l)

    freq = rf.Frequency.from_f(f, unit='hz')
    freq.unit = 'ghz'
    return rf.Network(s=offset_open_reflection, frequency=freq, name='open')

def short_standard(f, ind=[0], Z0=50, ereff=1, l=0):
    """
    modeling a short standard as polynomial inductance with an offset

    parameters
    -----------
    f:  frequency array in Hz
    ind: polynomial coefficients in increasing order
    Z0: ref impedance; can be array of same size as frequency
    ereff: relative effective permittivity; can be array of same size as frequency
    l: length of the offset
    """
        
    c0 = 299792458   # speed of light in vacuum (m/s)
    ind = np.atleast_1d(ind)
    f = np.atleast_1d(f)
    Z0 = np.atleast_1d(Z0)*np.ones(len(f))

    L = np.array([(ff**np.arange(len(ind))*ind).sum() for ff in f])  # polynomial model 

    omega = 2*np.pi*f
    short_reflection = (1j*omega*L - Z0)/(1j*omega*L + Z0)    # the reflection

    gamma = 2*np.pi*f/c0*np.sqrt(-ereff*(1+0.0j))
    offset_short_reflection = short_reflection*np.exp(-2*gamma*l)

    freq = rf.Frequency.from_f(f, unit='hz')
    freq.unit = 'ghz'
    return rf.Network(s=offset_short_reflection, frequency=freq, name='short')

if __name__=='__main__':

    # male
    c0 = 3.602e-15
    c1 = -146.3e-27
    c2 = 3.415e-36
    c3 = -0.01284e-45
    '''
    # female
    c0 = 0.5385e-15
    c1 = -12.08e-27
    c2 = 2.056e-36
    c3 = -0.007363e-45
    '''
    c = [c0, c1, c2, c3]  # lower to higher order
    f = np.linspace(0,70, 201)*1e9

    open_kit = open_standard(f, c, l=5e-3)
    short_kit = short_standard(f, l=5e-3)
    plt.figure()
    open_kit.plot_s_deg()
    short_kit.plot_s_deg()

    plt.show()
