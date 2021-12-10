import numpy as np
import math
from tqdm import tqdm
from scipy.signal import fftconvolve
from matplotlib import pyplot as plt

# TODO: Not done yet.

class LandauDistributionExact:
    # Based on:
    # https://journals.aps.org/rmp/pdf/10.1103/RevModPhys.60.663
    def __init__(self):

        # Mean excitaion energy of gold
        # self.I = 790. # eV
        # Density of gold
        # self.RHO = 19.32 # g / cm3
        # Atomic weight of gold
        # self.A = 196.966569 # g / mol

        # Clementi and Raimondi orbital charge coefficenties
        # http://www.knowledgedoor.com/2/elements_handbook/
        # clementi-raimondi_effective_nuclear_charge_part_2.html#gold
        self.XI = np.array([
            #    1s       2s       2p       3s       3p       4s       3d
            77.4761, 29.1849, 37.2566, 18.5876, 18.9010, 11.1033, 21.8361,
            #    4p      5s       4d      5p      6s       4f      5d
            10.8867, 5.4655, 10.3820, 5.0340, 1.8230, 10.1624, 4.0253])
        # Number of electrons in each shell of gold
        #                  1s 2s 2p 3s 3p 4s 3d  4p 5s 4d  5p 6s 4f  5d
        self.N = np.array([2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 1, 14, 10])

        # self.RY = 13.6058 # eV
        # self.D1 = self.RY * self.XI**2 * 4. / 3.
        self.D1 = 18.1410666667 * self.XI**2

        self.DN = (self.N * self.D1).sum()

        self.temp_axis = np.array([
            1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.2, 2.4,
            2.6, 2.8, 3., 3.3, 3.6, 4., 4.4, 4.8, 5., 5.5, 6., 6.5, 7., 7.5,
            8., 8.5, 9., 9.5])[::-1]

        # https://refractiveindex.info/?shelf=main&book=Au&page=Hagemann
        # Refractive index
        self.refractn = np.array([
            1., 1.002, 1.000, 1.000, 1.000, 1.000, 1.000, 1.001, 1.001, 1.000,
            1.001, 9.990e-1, 9.990e-1, 1.000, 1.000, 9.980e-1, 9.970e-1,
            9.960e-1, 9.950e-1, 9.920e-1, 9.910e-1, 9.890e-1, 9.880e-1,
            9.870e-1, 9.860e-1, 9.830e-1, 9.810e-1, 9.790e-1, 9.750e-1,
            9.700e-1, 9.630e-1, 9.550e-1, 9.490e-1, 9.440e-1, 9.370e-1,
            9.290e-1, 9.200e-1, 9.110e-1, 9.000e-1, 8.890e-1, 8.770e-1,
            8.720e-1, 8.680e-1, 8.650e-1, 8.660e-1, 8.650e-1, 8.630e-1,
            8.620e-1, 8.640e-1, 8.650e-1, 8.650e-1, 8.610e-1, 8.600e-1,
            8.570e-1, 8.530e-1, 8.500e-1, 8.470e-1, 8.420e-1, 8.350e-1,
            8.280e-1, 8.270e-1, 8.280e-1, 8.300e-1, 8.330e-1, 8.380e-1,
            8.360e-1, 8.300e-1, 8.200e-1, 8.080e-1, 7.950e-1, 7.870e-1,
            7.780e-1, 7.710e-1, 7.670e-1, 7.670e-1, 7.700e-1, 7.700e-1,
            7.660e-1, 7.600e-1, 7.540e-1, 7.510e-1, 7.490e-1, 7.450e-1,
            7.410e-1, 7.370e-1, 7.300e-1, 7.300e-1, 7.570e-1, 8.180e-1,
            8.440e-1, 8.680e-1, 8.870e-1, 8.960e-1, 8.920e-1, 8.850e-1,
            8.860e-1, 9.050e-1, 9.530e-1, 1.005, 1.080, 1.183, 1.292,
            1.406, 1.499, 1.584, 1.612, 1.565, 1.513, 1.505, 1.540, 1.591,
            1.592, 1.583, 1.569, 1.556, 1.548, 1.572, 1.585, 1.614, 1.666,
            1.727, 1.786, 1.728, 1.665, 1.618, 1.603, 1.638, 1.702, 1.848,
            1.927, 1.995, 2.039, 2.024, 1.967, 1.960, 1.960, 1.950, 1.862,
            1.541, 8.780e-1, 6.130e-1, 4.870e-1, 4.180e-1, 3.750e-1, 1.273,
            2.690e1, 6.550e1, 2.220e2, 3.450e2])
        # Extinction coefficient
        self.refractk = np.array([
            2.56e-9, 1.95e-8, 5.22e-9, 3.05e-8, 1.85e-7, 8.83e-7, 2.28e-6,
            1.94e-6, 2.56e-6, 2.53e-6, 2.87e-5, 1.64e-4, 1.45e-4, 2.11e-4,
            1.20e-4, 9.60e-4, 1.63e-3, 3.40e-3, 4.71e-3, 7.31e-3, 8.95e-3,
            9.76e-3, 1.06e-2, 1.13e-2, 1.18e-2, 1.15e-2, 1.11e-2, 1.07e-2,
            1.03e-2, 1.01e-2, 1.01e-2, 1.09e-2, 1.18e-2, 1.33e-2, 1.52e-2,
            1.78e-2, 2.24e-2, 2.83e-2, 3.53e-2, 4.54e-2, 5.70e-2, 6.36e-2,
            7.18e-2, 8.07e-2, 8.38e-2, 8.44e-2, 8.70e-2, 9.18e-2, 9.56e-2,
            9.62e-2, 9.38e-2, 9.43e-2, 9.63e-2, 1.01e-1, 1.09e-1, 1.17e-1,
            1.24e-1, 1.31e-1, 1.40e-1, 1.55e-1, 1.71e-1, 1.86e-1, 1.99e-1,
            2.11e-1, 2.19e-1, 2.23e-1, 2.30e-1, 2.41e-1, 2.55e-1, 2.72e-1,
            2.84e-1, 2.98e-1, 3.15e-1, 3.35e-1, 3.55e-1, 3.71e-1, 3.84e-1,
            3.98e-1, 4.15e-1, 4.38e-1, 4.61e-1, 4.84e-1, 5.11e-1, 5.40e-1,
            5.73e-1, 6.17e-1, 6.72e-1, 7.47e-1, 7.88e-1, 7.99e-1, 8.04e-1,
            8.03e-1, 8.02e-1, 8.14e-1, 8.55e-1, 9.17e-1, 1., 1.12, 1.18, 1.24,
            1.28, 1.29, 1.27, 1.21, 1.07, 9.50e-1, 8.65e-1, 8.65e-1, 8.98e-1,
            9.34e-1, 9.22e-1, 8.99e-1, 9.08e-1, 9.20e-1, 9.45e-1, 9.90e-1,
            1.05, 1.07, 1.14, 1.17, 1.22, 1.15, 1.13, 1.19, 1.30, 1.45, 1.65,
            1.77, 1.87, 1.89, 1.88, 1.86, 1.81, 1.84, 1.88, 1.89, 1.86, 1.81,
            1.71, 2.01, 2.64, 3.31, 5.13, 8.36, 1.81e1, 8.07e1, 1.24e2, 2.56e2,
            3.79e2])
        # Wavelength
        self.refracte = np.array([
            8.266e-6, 1.494e-5, 1.550e-5, 2.480e-5, 4.133e-5, 6.199e-5,
            8.266e-5, 8.856e-5, 9.537e-5, 1.240e-4, 2.480e-4, 4.275e-4,
            4.350e-4, 4.959e-4, 6.199e-4, 1.240e-3, 1.550e-3, 2.066e-3,
            2.480e-3, 3.542e-3, 4.133e-3, 4.428e-3, 4.769e-3, 5.166e-3,
            5.636e-3, 6.199e-3, 6.525e-3, 6.888e-3, 7.293e-3, 7.749e-3,
            8.266e-3, 8.856e-3, 9.184e-3, 9.537e-3, 9.919e-3, 1.033e-2,
            1.078e-2, 1.127e-2, 1.181e-2, 1.240e-2, 1.305e-2, 1.333e-2,
            1.362e-2, 1.393e-2, 1.409e-2, 1.417e-2, 1.425e-2, 1.442e-2,
            1.459e-2, 1.467e-2, 1.476e-2, 1.485e-2, 1.494e-2, 1.512e-2,
            1.550e-2, 1.590e-2, 1.631e-2, 1.675e-2, 1.722e-2, 1.771e-2,
            1.823e-2, 1.879e-2, 1.937e-2, 2.000e-2, 2.066e-2, 2.138e-2,
            2.214e-2, 2.296e-2, 2.384e-2, 2.480e-2, 2.530e-2, 2.583e-2,
            2.638e-2, 2.695e-2, 2.755e-2, 2.818e-2, 2.883e-2, 2.952e-2,
            3.024e-2, 3.100e-2, 3.179e-2, 3.263e-2, 3.351e-2, 3.444e-2,
            3.542e-2, 3.647e-2, 3.757e-2, 3.875e-2, 3.999e-2, 4.065e-2,
            4.133e-2, 4.203e-2, 4.275e-2, 4.428e-2, 4.592e-2, 4.769e-2,
            4.959e-2, 5.166e-2, 5.276e-2, 5.391e-2, 5.510e-2, 5.636e-2,
            5.767e-2, 5.904e-2, 6.199e-2, 6.525e-2, 6.888e-2, 7.293e-2,
            7.749e-2, 8.266e-2, 8.856e-2, 9.537e-2, 1.033e-1, 1.078e-1,
            1.127e-1, 1.181e-1, 1.240e-1, 1.305e-1, 1.378e-1, 1.459e-1,
            1.550e-1, 1.653e-1, 1.771e-1, 1.907e-1, 2.066e-1, 2.254e-1,
            2.480e-1, 2.610e-1, 2.818e-1, 2.952e-1, 3.100e-1, 3.263e-1,
            3.444e-1, 3.594e-1, 3.757e-1, 3.936e-1, 4.133e-1, 4.428e-1,
            4.769e-1, 5.166e-1, 5.636e-1, 6.199e-1, 8.266e-1, 1.240, 2.480,
            1.240e1, 2.480e1, 1.240e2, 2.480e2])
        # Energy in eV
        self.refracte = 4.135667662e-15 * 2.99792458e8 / (self.refracte * 1e-6)
        self.xaxis = self.refracte

        self.axis_out = np.append(self.temp_axis[:-6] * 1e5, self.xaxis)

        #self.xaxis = np.array([0.])
        for i in range(6):
            self.axis_out = np.append(
                self.temp_axis * np.power(10, i + 6), self.axis_out)

        self.xaxis2 = self.xaxis**2
        self.xaxis3 = self.xaxis**3

        self.K = 510998.928 * np.sqrt(self.xaxis / 13.6058) / 137.035999046
        self.K *= self.K

        self.Q = np.zeros(len(self.xaxis)) + 13904280.8309
        for i in range(len(self.Q)):
            if self.xaxis[i] >= 100:
                break
            self.Q[i] = 8690.1755193
        #self.Q = np.sqrt(
        #    510998.928**2 + self.xaxis * 510998.928 * 2.) - 510998.928

        self.e1 = self.refractn**2 - self.refractk**2
        self.e2 = 2. * self.refractn * self.refractk

        self.im1e = self.e2 / (self.e1**2 + self.e2**2)
        self.e = self.e1**2 + self.e2**2

        self.x27 = self.xaxis[0]**3.3637

        plt.plot(self.xaxis, self.im1e)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()



    def sigma(self, T):
        TM = T * 211316742. # = T * 2 * M (= Mass of muon: 105658371 eV)
        T2 = T**2
        T2M = T2 + TM
        # beta**2 = (T**2 + 2 * T * M) / (T + M)**2
        b2 = T2M / (T2M + 1.1163691e16)
        # gamma = (T + M) / M
        g = 1. + T * 9.46446543e-9
        # beta**2 * gamma**2
        b2g2 = (T2 / 1.1163691e16) + (T / 105658371) + 2.


        # me * c**2 = 510998.928 eV
        # K = z**2 * 2 * pi * e**4 / (me * c**2)
        # With z (charge of muon) = 1: K = 2.5496e-19 eV * cm**2
        rho = 1.033976e-7 / (b2 * self.xaxis2)
        Em = 211316742. * b2g2 / (206.773119415 + g)

        # Atomic number (charge) of gold Z = 79
        out1_u = 79. * (1. - b2 * self.xaxis / Em)

        # s = 0.5 * E / me * c**2
        # s = E * 9.78475634e-7
        out2_u = self.DN  / (self.xaxis + self.xaxis2 * 9.78475634e-7)

        out_u = rho * (out1_u + out2_u)

        # https://link.springer.com/content/pdf/10.1007/BF03214934.pdf
        # width = 8 eV
        # Er = 8 eV

        # e1 = 1. - (Er**2 / E**2) * (1. / (1. + (width / E)**2))
        # e2 = width * (Er**2 / E**3) * (1. / (1. + (width / E)**2))
        # x8 = 1. / (1. + (width / E)**2)
        # x8 = 1. / (1. + (8. / self.xaxis)**2)
        # e1 = 1. - (64. / self.xaxis2) * x8
        # e2 = (512. / self.xaxis3) * x8

        # e1e2 = e1**2 + e2**2
        b2e1 = 1. - b2 * self.e1
        b2e2 = b2 * self.e2

        # Elementary charge
        # e = 0.30282212

        # With z (charge of muon) = 1: out *= z2
        # out_t = e**2 / (3.1415926536 * b2)
        out_ = 0.0291894101 / b2

        out_l = np.log(Em * self.Q * b2 / self.xaxis2)
        out_l *= self.im1e * out_

        # out1_t = np.log(np.sqrt(b2e1**2 + b2e2**2))
        out1_t = self.im1e * np.log(1. / np.sqrt(b2e1**2 + b2e2**2))
        # out1_t *= (self.e2 / self.e)
        #print(out1_t)
        out2_t = (b2 - (self.e1 / self.e)) * np.arctan(b2e2 / b2e1)
        out_t = out_ * (out1_t + out2_t)

        #out1_t
        #plt.plot(self.xaxis, out_u, label="u")
        #plt.plot(self.xaxis, out_l, label="l")
        #plt.plot(self.xaxis, out_t, label="t")
        #plt.legend()
        #return out_u

        out = out_t + out_l + out_u
        first_e = out[0]
        # delta_s = out[1] - out[0]
        out = np.append(
            self.x27 * first_e / (self.temp_axis[:-6] * 1e5)**3.3637, out)

        for i in range(6):
            out = np.append(self.x27 * first_e / \
                        (self.temp_axis * np.power(10, i + 6))**3.3637, out)

        #print((first_e + delta_s / first_e))
        #print((self.xaxis[0] / self.xaxis[1])**3.3637)
        return out * (self.axis_out <= T) #first_e**-0.0005


    def sigma_n(self, n, T):
        if n == 1:
            return self.sigma(T)
        else:
            out = fftconvolve(
                self.sigma(T), self.sigma_n(n - 1, T),
                mode="full")[359:]
            return out * (self.axis_out <= T)


    #def f(self, T):
    #    for n in range(5):



class MuonDistribution:
    def __init__(self):
        # Energies in GeV
        self.e = 854 # GeV
        self.Emax = 5e3 # GeV

        self.eV = 1e9
        self.keV = self.eV * 1e-3
        self.MeV = self.keV * 1e-3
        self.GeV = self.MeV * 1e-3

    def muon_energy(self, E, E0, I0=70., n=3., norm=1.):
        Emax = self.Emax
        e = self.e
        N = (n - 1.) * math.pow(E0 + Emax, n - 1.)
        out = I0 * N * np.power(E0 + E, -n) * (e / (e + E))
        return  (e * E) * out / norm

    def muon_angle(self, theta):
        theta *= math.pi / 180.
        return math.pow(math.cos(theta), 2.)

    def muon_angle_normed(self, theta):
        norm = 0.
        for i in range(90):
            norm += self.muon_angle(i)
        return 200. * self.muon_angle(theta) / norm

    def get_probs_per_keV(self, xaxis):
        return self.muon_energy(
            (xaxis / self.keV), 4.298, norm=1084901707741.9655)

l = LandauDistributionExact()
plt.plot(l.axis_out, l.sigma_n(1, 2.5e6) * l.axis_out**2)
#plt.plot(xaxis, l.sigma(2e9))# * xaxis**2)
#plt.plot(xaxis, l.sigma(2e10))# * xaxis**2)
#plt.plot(xaxis, l.sigma(2e11))# * xaxis**2)
#plt.plot(xaxis, l.sigma(2e12))# * xaxis**2)
plt.xlim((10, 1e5))
plt.xscale('log')
plt.yscale('log')
plt.show()
