import numpy as np
import math
import sys
import os
from tqdm import tqdm
import multiprocessing as mp
import pandas as pd
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join("..", "Read_Data")))
import global_parameters as gl


def multi_func(theta_list, l5, l10, l15, l20, axis, md, getL,
        lock, queue, t0, out_str, muon):
    """ Calculate energy loss probabilities for particles with different
        energies and arriving angles.

    Args:
        theta_list (numpy.array): List of zenith angles.
        l5 (mp.Array): Array, which will be filled with the energy loss
                       propabilities for an absorber thickness of 5 um.
        l10 (mp.Array): Array, which will be filled with the energy loss
                        propabilities for an absorber thickness of 10 um.
        l15 (mp.Array): Array, which will be filled with the energy loss
                        propabilities for an absorber thickness of 15 um.
        l20 (mp.Array): Array, which will be filled with the energy loss
                        propabilities for an absorber thickness of 20 um.
        axis (numpy.array): List of deposited energies.
        md (MuonDistribution): The energy and angular distributions of the particles.
        getL (func): Function giving the path length for a given theta and position.
        lock (mp.Lock): Lock for datawriting to the arrays.
        queue (mp.Queue): Queue for parallel threads.
        t0 (datetime): Start time of this task.
        out_str (string): Output which should be printed during processing.
        muon (boolean): IF muons or electrons should be considered.
    """

    # Load the landau distribution.
    ld = LandauDistribution(muon=muon)

    # The dimensions of the absorbers are 180 um * 180 um.
    # It can be splitted in four identical parts a 90 um * 90 um.
    # The 90 um * 90 um quarter is symmetrical around the diagonal.

    #         <----   180 um  ---->
    #        |----------|----------|  ^
    #        |          |          |  |
    #        |          |          |  |
    #        |          |          |
    #     ^  |----------|----------| 180 um
    #     |  |        / |          |
    # 90 um  |     /    |          |  |
    #     |  |  /       |          |  |
    #     v  |----------|----------|  v
    #        <- 90 um ->

    # The eighth part is then pixelized in NMAX * NMAX pixels.
    # NMAX should be an odd number.
    NMAX = 3

    # Number the rows and columns (e.g. for NMAX = 5):

    #    5     4     3     2     1
    # |-----|-----|-----|-----|-----|
    # |     |     |     |     |  /  | 1
    # |-----|-----|-----|-----|-----|
    # |     |     |     |  /  |     | 2
    # |-----|-----|-----|-----|-----|
    # |     |     |  /  |     |     | 3
    # |-----|-----|-----|-----|-----|
    # |     |  /  |     |     |     | 4
    # |-----|-----|-----|-----|-----|
    # |  /  |     |     |     |     | 5
    # |-----|-----|-----|-----|-----|

    # Determine the positions of each pixel.
    #  The top right corner is at (0, 0).
    #  y
    #  ^
    #  |
    #  |
    #  |
    #  -----------> x

    def get_p0(n, nmax=5):
        """ Get the x-values of the pixels of the n-th row.

        Args:
            n (int): The row number.
            nmax (int, optional): The sqrt of the pixel numbers. Defaults to 5.

        Returns:
            numpy.array: The x-positions.
        """

        # Define the pixel size.
        d = 90.e-6 / nmax
        d2 = d / 2.

        # n = 1: [-d/2]
        # n = 2: [-d - d/2, -d/2]
        # n = 3: [-2d - d/2, -d - d/2, -d/2]
        return (d * np.arange(int(n)) + d2) - n * d


    def get_p1(n, nmax=5):
        """ Get the y-values of the pixels of the n-th row.

        Args:
            n (int): The row number.
            nmax (int, optional): The sqrt of the pixel numbers. Defaults to 5.

        Returns:
            numpy.array: The y-positions.
        """

        d = 90.e-6 / nmax
        d2 = d / 2.
        # n = 1: [-d/2]
        # n = 2: [-d - d/2, -d - d/2]
        # n = 3: [-2d - d/2, -2d - d/2, -2d - d/2]
        return (np.zeros(int(n)) + d2) - n * d


    def get_max_d(n, nmax=5):
        """ Get the distance to the edge for pixels of a row.

        Args:
            n (int): The row number.
            nmax (int, optional): The sqrt of the pixel numbers. Defaults to 5.

        Returns:
            numpy.array: The (shortest) distance to the edge.
        """

        d = 90.e-6 / nmax # <-> 90e-6 = nmax * d
        d2 = d / 2.
        # n = 1: [90e-6 - d/2]
        # n = 2: [90e-6 - d - d/2, 90e-6 - d - d/2]
        # n = 3: [90e-6 - 2d - d/2, 90e-6 - 2d - d/2, 90e-6 - 2d - d/2]
        return np.zeros(int(n)) + d2 + (nmax - n) * d


    def get_weight(n):
        """ The pixels are symmetrical, but the pixels at the diagonal only
            count half. Thus, weights have to be defined.

        Args:
            n (int): The row number.

        Returns:
            numpy.array: The weights.
        """

        out = np.ones(int(n))
        out[0] *= 0.5
        return out

    # Create the pixel postions, weights and minimum distances to the edge.
    p0_list = None
    p1_list = None
    x_list = None
    w_list = None

    # Iterate over all rows (pixels).
    for n in np.arange(NMAX)[::-1] + 1:
        if p0_list is None:
            p0_list = get_p0(n, nmax=NMAX)
            p1_list = get_p1(n, nmax=NMAX)
            x_list = get_max_d(n, nmax=NMAX)
            w_list = get_weight(n)
        else:
            p0_list = np.append(p0_list, get_p0(n, nmax=NMAX))
            p1_list = np.append(p0_list, get_p1(n, nmax=NMAX))
            x_list = np.append(x_list, get_max_d(n, nmax=NMAX))
            w_list = np.append(p0_list, get_weight(n))

    # Normalize the weights to one.
    w_list /= w_list.shape[0]

    # Create azimuth angles.
    # Number of azimuth angles.
    NUM_PHI = 90
    pi2 = np.pi / 180.
    # Divide the number of angles over 360 degrees.
    phi_list = np.arange(NUM_PHI) * 360 / NUM_PHI
    # Convert the angles in radians.
    phi_list *= pi2
    # Get the cos and sin of each azimuth angle.
    sinp = np.sin(phi_list)
    cosp = np.cos(phi_list)

    # Define the normalization factor. It is given as the product of number of
    # zenith angles and number of azimuth angles.
    norm = 1. / (180. * NUM_PHI)


    # Determine the maximum path length, which is connected to the arriving
    # angle alpha, for each position (by considering the nearest edge). Alpha
    # is given by the arctan of the distance to the edge x divided by the
    # thickness d:

    #        \  |
    #   alpha-> |
    #          \|<-x->
    #    |------o----| ^
    #    |       \   | |
    #    | 90 -   \  | d
    #    | alpha-> \ | |
    #    |-----------| v

    # For arriving angles theta (zenith angles), which are larger then alpha,
    # the path length has to be differnet calculated as with d / cos(theta).

    # Define the absorber thicknesses.
    d_list = np.array([5.e-6, 10.e-6, 15.e-6, 20.e-6])
    # Calculate the alpha angles for each position and thickness.
    atan_dict = {}
    # Initialize the deposited energy distributions for each thickness.
    t_dict = {}
    for i in range(4):
        atan_dict[i] = np.arctan(x_list / d_list[i]) / pi2
        t_dict[i] = np.zeros(len(axis))

    # Iterate over all zenith angles.
    for theta in theta_list:
        # Get the cos and sin of the zenith angle.
        sint = np.sin(pi2 * theta)
        cost = np.cos(pi2 * theta)

        # The direction vector (spherical coordinates) is given by (v0, v1, v2)
        # = (sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)).
        tmp_v0 = sint * cosp
        tmp_v1 = sint * sinp

        # If theta is smaller than alpha, the azimuth angle does not matter.
        # The path length can be calculated with l = d / cos(theta), thus define
        # a direction vector with phi = 0.
        _v = [sint, 0., cost]

        # Iterate over all positions.
        for p_index, p0 in enumerate(p0_list):
            # Get the position vector.
            pos = [p0, p1_list[p_index]]
            # Determine, if theta is smaller than alpha for each thickness.
            use_l = np.zeros(4, dtype=np.bool)
            all_true = True
            for i in range(4):
                # Check, if the path length can be calculated with d / cos(theta).
                # Meaning: check, if theta is smaller then alpha.
                if theta <= atan_dict[i][p_index]:
                    use_l[i] = True
                else:
                    all_true = False

            # Iterate over all muon energies for the current zenith angle.
            # The energies in Theta_Energies are given in multiples of 100 eV,
            # but be needed in keV = 100 eV / 10.
            for e_index, energy in enumerate(md.Theta_Energies[theta] / 10.):
                # Get the intensity of the muon energy defined in
                # Theta_Energy_Dis and scale it with the pixel (position) weight.
                w = md.Theta_Energy_Dis[theta][e_index] * w_list[p_index]

                # Iterate over all thicknesses.
                for i in range(4):
                    if use_l[i]:
                        # The azimuth angle does not matter.
                        # Get the landau distribution for a single phi and
                        # scale it with the weight.
                        t_dict[i] += ld.f(
                            energy, axis, getL(pos, _v, d=d_list[i])) * w

                if all_true:
                    # The azimuth angle does not matter for any thickness.
                    continue

                # The path length has to be calculated for each azimuth angle.
                # Scale the weight with the number of angles.
                w *= norm

                # Iterate over all azimuth angles.
                for v_index, v0 in enumerate(tmp_v0):
                    v = [v0, tmp_v1[v_index], cost]

                    # Iterate over all thicknesses.
                    for i in range(4):
                        # Get the landau distribution for the given zenith
                        # and azimuth angle.
                        t_dict[i] += ld.f(
                            energy, axis, getL(pos, v, d=d_list[i])) * w

            # Print the progess.
            if (theta * 2) % 4 == 0:
                lock.acquire()
                position = queue.get()
                if position % 3 == 0:
                    gl.show_progress(
                        position, 45 * len(p0_list), t0, out_str=out_str)
                queue.put(position + 1)
                lock.release()

    # Add the accumulated landau distributions to the output arrays.
    lock.acquire()
    for i in range(axis.shape[0]):
        l5[i] += t_dict[0][i]
        l10[i] += t_dict[1][i]
        l15[i] += t_dict[2][i]
        l20[i] += t_dict[3][i]
    lock.release()


class LandauDistribution:
    """ Calculates the energy loss of electrons or muons in gold following
        https://doi.org/10.1016/S0168-9002(96)00774-7.

        Methods:
            xi(Z/A, beta, rho, x): Xi for a material with thickness x and density rho.
            xi_gold(beta, x): Xi of gold absorbers with a thickness x.
            xi_10um(beta): Xi of gold with a thickness of 10 um.

            get_beta(E_kin): The velocity of the particle.
            gamma(beta): The Lorentz factor for a given velocity.

            lamb(beta, E_loss, x): Lambda, the Landau's universal variable.
            Lamb(l): ln(lambda).

            small_lamb(l): Phi of Eq. (14).
            large_lamb(l): Phi of Eq. (12).

            small_phi(l): Phi of Eq. (15). For -3.4 < lambda < -1.
            middle_phi(l): Phi of Eq. (10). For -1 < lambda < 3.
            high_phi(l): Phi of Eq. (11). For 3 < lambda < 150.
            large_phi(l): Phi of Eq. (13). For lambda > 150.

            phi(beta, E_loss, x): Phi.
            get_prob(beta, phi, x): Eq (4).
            f(E_kin, energies, x): Load/Create data following Eq. (4).
    """

    def __init__(self, muon=True):
        """
        Args:
            muon (bool, optional): If the energy loss should be calculated for
                                   muon or electrons. Defaults to True.
        """

        # The electron mass.
        self.mec = 0.510998928 # MeV
        self.mec *= 1e3 # keV
        # The muon mass.
        self.mmu = 206.7682826 * self.mec

        # Set the muon flag.
        self._Muon = muon
        if not self._Muon:
            # Calculate the landau distribution for electrons.
            self.mmu = self.mec

        # The mean excitation energy of gold.
        self.I = 790. # eV
        self.I *= 1e-3 # keV

        # Define some parameterization parameters from
        # https://doi.org/10.1016/S0168-9002(96)00774-7
        # For -1 < lambda < 3; Eq (10):
        self.coeff_0 = np.array([0.17885481, -0.015464468, -0.030040882,
                                 0.013781358, -0.0011491556, -0.0012835837,
                                 0.00062395162, -0.0001262613, 0.000010108918])
        # For 3 < lambda < 150; Eq (11):
        self.coeff_1 = np.array([-1.5669876, -1.5811676, 1.677088,
                                 -1.4972908, 0.57062974, -0.11777036,
                                 0.01343737, -0.00076315158, 0.000014881108])


    def xi(self, za, beta, rho, x):
        """ Xi of Eq (3) in https://doi.org/10.1016/S0168-9002(96)00774-7.

        Args:
            za (float): The atomic number divided by the atomic mass of the
                        absorber material: Z / A.
            beta (float): Velocity of the particle.
            rho (float): Density of the absorber material in mg / cm3.
            x (float): Thickness of the absorber material in cm.

        Returns:
            float: Xi.
        """

        return 0.1536 * za * rho * x / beta**2


    def xi_gold(self, beta, x):
        """ Xi of gold.

        Args:
            beta (float): Velocity of the particle.
            x (float): Thickness of the absorber material.

        Returns:
            float: Xi.
        """

        za = 0.40108 # = 79 / 196.97
        rho = 19.32 # g / cm**3
        rho *= 1e3 # mg / cm**3
        return self.xi(za, beta, rho, x)


    def xi_10um(self, beta):
        """ Xi of gold with a thickness of 10 um.

        Args:
            beta (float): Velocity of the particle.

        Returns:
            float: Xi.
        """

        x = 1e-3 # cm
        return self.xi_gold(beta, x)


    def gamma(self, beta):
        """ Get the Lorentz factor.

        Args:
            beta (float): The particle velocity.

        Returns:
            float: The Lorentz factor.
        """

        return 1. / np.sqrt(1. - beta**2)


    def lamb(self, beta, E_loss, x):
        """ Lambda, Landaus's universal variable. See Eq. (5) in
            https://doi.org/10.1016/S0168-9002(96)00774-7.

        Args:
            beta (float): The particle velocity.
            E_loss (float): The energy loss.
            x (float): The absorber thickness.

        Returns:
            float: Lambda.
        """

        first = E_loss / self.xi_gold(beta, x)
        second = np.log(
            self.xi_gold(beta, x) * 2. * self.mec * (
                beta * self.gamma(beta))**2 / self.I**2)
        third = -1. + beta**2 + 0.2 # 0.577
        return first - second + third


    def Lamb(self, l):
        """ Natural logarithmus of lambda. Compare to paragraph above Eq. (11)
            in https://doi.org/10.1016/S0168-9002(96)00774-7.

        Args:
            l (float): Lambda.

        Returns:
            float: ln(lambda)
        """

        return np.log(l)


    def small_lamb(self, l):
        """ Phi of Eq. (14) in https://doi.org/10.1016/S0168-9002(96)00774-7.

        Args:
            l (float): Lambda

        Returns:
            float: Phi
        """

        a = 1. / np.sqrt(2. *np.pi)
        b = np.abs(l) - 1.
        return a * np.exp((b / 2.) - np.exp(b))


    def large_lamb(self, l):
        """ Phi of Eq. (12) in https://doi.org/10.1016/S0168-9002(96)00774-7.

        Args:
            l (float): Lambda

        Returns:
            float: Phi
        """

        return (1. / l**2) - (
            3. - 2. * 0.577 - 2. * self.Lamb(l)) * (1. / l**3)


    def small_phi(self, l):
        """ Phi of Eq. (15) in https://doi.org/10.1016/S0168-9002(96)00774-7.
            For -3.4 < lambda < -1.

        Args:
            l (float): Lambda

        Returns:
            float: Phi
        """

        phi = self.small_lamb(l)
        return phi * (1 + 0.01 * (
            6.7853 + 4.884 * l +
            1.4488 * l**2 +
            0.20802 * l**3 +
            0.012057 * l**4))

    def high_phi(self, l):
        """ Phi of Eq. (11) in https://doi.org/10.1016/S0168-9002(96)00774-7.
            For 3 < lambda < 150.

        Args:
            l (float): Lambda

        Returns:
            float: Phi
        """

        out = 0.
        L = self.Lamb(l)
        for i, c in enumerate(self.coeff_1):
            out += c * L**i

        # Note the ln(phi) in Eq. (11).
        return np.exp(out)


    def middle_phi(self, l):
        """ Phi of Eq. (10) in https://doi.org/10.1016/S0168-9002(96)00774-7.
            For -1 < lambda < 3.

        Args:
            l (float): Lambda.

        Returns:
            float: Phi.
        """

        out = 0.
        for i, c in enumerate(self.coeff_0):
            out += c * l**i
        return out


    def large_phi(self, l):
        """ Phi of Eq. (13) in https://doi.org/10.1016/S0168-9002(96)00774-7.
            For lambda > 150.

        Args:
            l (float): Lambda

        Returns:
            float: Phi
        """

        phi = self.large_lamb(l)
        return phi / (1. - 0.01 * np.exp(5.157 - 1.42 * self.Lamb(l)))


    def phi(self, beta, E_loss, x):
        """ Calculate phi.

        Args:
            beta (float): Velocity of the particle.
            E_loss (float or numpy.array): Energy loss of the particle.
            x (float): Thickness of the absorber.

        Returns:
            float: Phi
        """

        # Get lambda.
        l = self.lamb(beta, E_loss, x)

        # Check if phi should be calculated for a single energy loss.
        if type(l) == np.float:
            if l < -1:
                return self.small_phi(l), l
            elif l < 3:
                return self.middle_phi(l), l
            elif l < 150:
                return self.high_phi(l), l
            else:
                return self.large_phi(l), l

        # Categorize lambda in four regions.
        small_ = (l < -1)
        middle_ = ((l < 3) & (l >= -1))
        high_ = ((l < 150) & (l >= 3))
        large_ = (l >= 150)

        cuts = [small_, middle_, high_, large_]
        funcs = [self.small_phi, self.middle_phi, self.high_phi, self.large_phi]

        # Initialize the output.
        out_phi = np.zeros(l.shape[0], dtype=np.float64)

        # Iterate over all regions.
        for i, cut in enumerate(cuts):
            # Check if there are any lambdas in this category.
            if cut.sum() > 0:
                # Get the first and last lambda of this category.
                first = cut.astype(np.int).argmax()
                last = cut.astype(np.int)[::-1].argmax()

                # Get the corresponding phi.
                if first > 0:
                    temp = np.zeros(first, dtype=np.float)
                    temp = np.append(temp, funcs[i](l[cut]))
                else:
                    temp = funcs[i](l[cut]).copy()

                if last > 0:
                    temp = np.append(temp, np.zeros(last, dtype=np.float))
                out_phi += temp

        return out_phi, l


    def get_prob(self, beta, phi, x):
        """ Get the propability for an energy loss corresponding to phi.
            Eq. (4) in https://doi.org/10.1016/S0168-9002(96)00774-7.

        Args:
            beta (float): The velocity of the particle.
            phi (float): The phi.
            x (float): The thickness of the absorber.

        Returns:
            float: the probability for an energy loss.
        """

        return (1. / self.xi_gold(beta, x)) * phi


    def get_beta(self, E_kin):
        """ Get the velocity of a particle with a given energy.

        Args:
            E_kin (float): The kinetic energy of the particle.

        Returns:
            float: The velocity of the particle.
        """

        # E_kin = E - m = (gamma - 1) * m
        # => gamma = 1 + E_kin / m
        g = 1. + (E_kin / self.mmu)

        # (E_kin / m)**2 + 2 * E_kin / m + 1 = gamma**2
        # gamma**2 - 1 = beta**2 / (1 - beta**2) = (beta * gamma)**2
        # => beta * gamma = sqrt((E_kin / m)**2 + 2 * E_kin / m)
        # bg = np.sqrt((E_kin / self.mmu)**2 + (2. * E_kin / self.mmu))
        # b = bg / g

        # gamma**2 = 1 /(1 - beta**2)
        # => beta = sqrt(1 - (1 / gamma**2))
        return np.sqrt(1. - (1. / g**2))


    def f(self, E_kin, energies, x):
        """ Calculates the propability for energy losses of energies of particles
            with E_kin passing an absorber with thickness x.

        Args:
            E_kin (float): The kinetic energy of the particle.
            energies (numpy.array): List of energy losses.
            x (float): Absorber thickness.

        Returns:
            numpy.array: List of energy loss propabilities.
        """

        # Get the magnitude of the kinetic energy and convert this to a string.
        if E_kin < 1e3:
            e_str = str(E_kin)
            e_str = e_str.split(".")[0] + "keV"
        elif E_kin < 1e4:
            e_str = str(E_kin / 1e3)
            e_str = e_str.split(".")[0] + "MeV" + e_str.split(".")[1][:2]
        elif E_kin < 1e5:
            e_str = str(E_kin / 1e3)
            e_str = e_str.split(".")[0] + "MeV" + e_str.split(".")[1][:1]
        elif E_kin < 1e6:
            e_str = str(E_kin / 1e3)
            e_str = e_str.split(".")[0] + "MeV"
        elif E_kin < 1e7:
            e_str = str(E_kin / 1e6)
            e_str = e_str.split(".")[0] + "TeV" + e_str.split(".")[1][:2]
        elif E_kin < 1e8:
            e_str = str(E_kin / 1e6)
            e_str = e_str.split(".")[0] + "TeV" + e_str.split(".")[1][:1]
        else:
            e_str = str(E_kin / 1e6)
            e_str = e_str.split(".")[0] + "TeV"

        # Get the absorber thickness.
        x_str = str(x * 1e4)
        _x_str = x_str.split(".")
        x_str = _x_str[0] + "um"

        # Create a name of the file, which contains the list of probabilities.
        p_name = e_str + "_" + x_str + ".npy"

        # Select the directory based on the particle type.
        if self._Muon:
            p = os.path.join(gl.LANDAU_PATH, p_name)
        else:
            p = os.path.join(gl.LANDAU_E_PATH, p_name)

        # If the file exits, load the data.
        if os.path.exists(p):
            try:
                return np.load(p)
            except ValueError:
                print(p_name)

        # Get the velocity of the particle.
        beta = self.get_beta(E_kin)

        # Get the list of phis from the energy losses.
        phi_, l = self.phi(beta, energies, x)
        # Get the propability for each phi.
        prob = self.get_prob(beta, phi_, x).astype(np.float64)
        # The propability can be very tiny and result in NaN.
        prob = np.nan_to_num(prob).astype(np.float64)
        # Save the list.
        np.save(p, prob)

        # norm = np.trapz(prob, x=energies)
        return prob


# %matplotlib widget
class MuonDistribution:
    """ Calculates the energy loss of electrons or muons with a given
        distribution of energies and angles.

        Attributes:
            e (float): Epsilon, an energy parameter in GeV accounting the
                       finite life time of pions and kaons.
            Emax (float): The maximum energy of muons in GeV.

            eV (float): A modifier to get energies in eV.
            keV (float): A modifier to get energies in keV.
            MeV (float): A modifier to get energies in MeV.
            GeV (float): A modifier to get energies in GeV.

            Zetas (numpy.array): Zeta = p * cos(theta).
            Energy_Dis (numpy.array): Intensity distribution of zetas.
            Thetas (numpy.array): zenith angles in degree.
            Theta_Dis (numpy.array): Angular distribution of thetas.
            Theta_Energies (dict): Energies for different thetas.
            Theta_Energy_Dis (dict): Energy distribution for different thetas.

        Methods:
            energy_dis(zeta): Determine the energy distribution following
                              https://arxiv.org/pdf/hep-ph/0604145.pdf.
            get_probs_per_keV(xaxis): Determine the energy distribution of
                                      muon_energy for energies given in keV.
            muon_energy(E, E0): Determine the energy distribution following
                                https://arxiv.org/pdf/1606.06907.pdf.
    """

    def __init__(self, muon=True):
        """
        Args:
            muon (bool, optional): Defines if the angular and energy
                                   distribution of muons should be used or
                                   uniform distributed distributions.
                                   Defaults to True.
        """

        # A parameter, which modifies the power in the high energy part and
        # that should account for the finite life time of pions and kaons.
        # Comapre to the sentence above Eq. (2) in
        # https://arxiv.org/pdf/1606.06907.pdf.
        # The value is identified there for a zenith angle of 0 degree. Compare
        # to the paragraph below Figure 4 and to Table 1.
        self.e = 854 # GeV

        # The maximum energy of muons (cut-off value).
        self.Emax = 5e3 # GeV

        # Some modifiers, which can be used to change the unit of an energy.
        # The deafult unit is GeV.
        self.eV = 1e9
        self.keV = self.eV * 1e-3
        self.MeV = self.keV * 1e-3
        self.GeV = self.MeV * 1e-3

        # Generate some values for zeta (compare to the caption of Figure 3 in
        # https://arxiv.org/pdf/hep-ph/0604145.pdf). Zeta = p * cos(theta).
        # Define 90 values for each magnitude: (10, 11, ..., 99) * 10**n,
        # for n = 0, 1, ..., 11. Note: 100**n = 10**(n+1)
        self._temp_axis = np.arange(90, dtype=np.uint64) + 10 # in 100 eV

        # Xaxis will store these values.
        self._xaxis = np.copy(self._temp_axis).astype(np.uint64)

        # Deltas will give the differences between two neighboring x-values.
        # deltas[i] = xaxis[i] - xaxis[i-1]
        self._deltas = np.ones(90, dtype=np.uint64) # in 100 eV

        # Iterate over 11 magnitudes
        # (actually 12, the zero-th case is the initialed arrays)
        for i in range(11):
            # The array which will be appended to xaxis.
            tmp = self._temp_axis.copy()
            # The array, which will be appended to deltas.
            tmp_d = np.ones(90, dtype=np.uint64)
            # Get the magnitude.
            for j in range(i + 1):
                tmp *= 10
            for j in range(i):
                tmp_d *= 10
            # Append the arrays.
            self._xaxis = np.append(self._xaxis, tmp)
            self._deltas = np.append(self._deltas, tmp_d)

        # Assign zeta.
        self._Zetas = self._xaxis # in 100 eV

        # Remove the last entries: Why? Has to be checked.
        self.Zetas = self._Zetas[:-1]
        self._deltas = self._deltas[:-1]

        # Get the intensity for each zeta.
        self._Energy_Dis = self.energy_dis(self._Zetas)
        # Normalize the total instensity to one. Note that the zeta axis is
        # logarithmical, thus not evenly spaced.
        self._Energy_Dis = self._Energy_Dis / np.trapz(
            self._Energy_Dis, x=self._Zetas)

        # Create a binned intensity distribution. Initialize the array.
        self.Energy_Dis = np.zeros(
            self._Energy_Dis.shape[0] - 1, dtype=np.double)
        # Iterate over all intensities.
        # TODO: Use enumerate.
        for i in range(self.Energy_Dis.shape[0]):
            # Get the mean intensity of two neighboring entries.
            self.Energy_Dis[i] = \
                    (self._Energy_Dis[i] + self._Energy_Dis[i + 1]) / 2.
            # Scale it with the distance of the entries.
            self.Energy_Dis[i] *= self._deltas[i]

        # Normalize the total intensity to one.
        self.Energy_Dis = self.Energy_Dis / self.Energy_Dis.sum()

        # Generate 180 zenith angles between 0 and 90 degree.
        self.Thetas = np.arange(180) / 2. # in degree
        # Generate the distribution of thetas: cos(theta)**2.
        # Note: Numpy needs angles in radians.
        self.Theta_Dis = np.power(np.cos(math.pi * self.Thetas / 180.), 2)
        # Normalize the total spectrum to one.
        self.Theta_Dis = self.Theta_Dis / self.Theta_Dis.sum()

        # Convert the zetas to kinetic energies and calculate the intensity
        # for these energies.
        self.Theta_Energies = {}
        self.Theta_Energy_Dis = {}
        if muon:
            # Use the cos2 angular distribution.
            # Iterate over all zenith angles and convert the zetas to energies.
            # Note: Theta is given in degrees, but numpy needs radians.
            for theta in self.Thetas:
                self.Theta_Energies[theta] = self.Zetas.astype(np.double) / \
                        np.cos(math.pi * theta / 180.) # in 100 eV
                # Scale the intensity distribution with the angular distribution.
                self.Theta_Energy_Dis[theta] = self.Theta_Dis[int(theta * 2)] * self.Energy_Dis
        else:
            for theta in self.Thetas:
                # Only consider an electron with 500 keV for each angle.
                self.Theta_Energies[theta] = np.array([0, 5000]) # in 100 eV
                self.Theta_Energy_Dis[theta] = np.array([0, 1. / 180.])


    def muon_energy(self, E, E0, I0=70., n=3., norm=1.):
        """ Get the energy distribution of muons. Eq. (2) in
            https://arxiv.org/pdf/1606.06907.pdf.

        Args:
            E (float or numpy.array): The muon energy in GeV.
            E0 (float): An energy parameter in GeV.
            I0 (float, optional): The intensity amplitude. Defaults to 70.
            n (int, optional): The power of the angular distribution (cos2) +1.
                               Defaults to 3.
            norm (float, optional): Normalizationtion. Defaults to 1.

        Returns:
            float or numpy.array: The energy distribution.
        """

        N = (n - 1.) * math.pow(E0 + self.Emax, n - 1.)
        out = I0 * N * np.power(E0 + E, -n) * (self.e / (self.e + E))
        return  (self.e * E) * out / norm


    def get_probs_per_keV(self, xaxis):
        """ Calculate the muon energy distribution for energies given in keV.
        Args:
            xaxis (float or numpy.array): The muon energy in keV.

        Returns:
            float or numpy.array: The energy distribution.
        """
        # E0 = 4.298 is taken from https://arxiv.org/pdf/1606.06907.pdf.
        # The norm is determined for the used xaxis in Landaus.

        return self.muon_energy(
            (xaxis / self.keV), 4.298, norm=1084901707741.9655)


    def energy_dis(self, zeta):
        """ Get the intensity for a given energy. Eq. (3) in
            https://arxiv.org/pdf/hep-ph/0604145.pdf.

        Args:
            zeta (float or numpy.array): A zeta.

        Returns:
            float or numpy.array: Intensities for each zeta.
        """

        c1 = 0.00253
        c2 = 0.2455
        c3 = 1.288
        c4 = -0.2555
        c5 = 0.0209

        # Zeta = p * cos(theta). Zeta is given in multiples of 100 eV, but has
        # to be given in GeV: 100 eV = 1 GeV / 1e7
        p_cos = np.double(zeta) / 1.e7
        lE = np.log10(p_cos)
        lE2 = lE * lE
        exponent = c2 + c3 * lE + c4 * lE2 + c5 * lE2 * lE

        return (c1 / np.power(p_cos, exponent))


class Landaus:
    """ Calculates the energy loss of electrons or muons in gold.

        Attributes:
            xaxis (numpy.array): List of deposited energies in eV.
            landaus_5um (numpy.array): Distribution for 5 um thick absorbers.
            landaus_10um (numpy.array): Distribution for 10 um thick absorbers.
            landaus_15um (numpy.array): Distribution for 15 um thick absorbers.
            landaus_20um (numpy.array): Distribution for 20 um thick absorbers.

        Methods:
            get_length(self, pos, v): Calculates the path length.
    """

    def __init__(self, muon=True):
        """
        Args:
            muon (bool, optional): If the energy loss should be calculated for
                                   muon or electrons. Defaults to True.
        """

        # Generate the values of deposited energies. Define 31 energies for each
        # magnitide: (1, 1.1, ..., 9.5) * 10**n, for n = 0, 1, ..., 11
        # The energies are given in keV.
        self._temp_axis = np.array([
            1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.2, 2.4,
            2.6, 2.8, 3., 3.3, 3.6, 4., 4.4, 4.8, 5., 5.5, 6., 6.5, 7., 7.5,
            8., 8.5, 9., 9.5])
        self.xaxis = np.copy(self._temp_axis)

        # Iterate over 11 magnitudes
        # (actually 12, the zero-th case is the initialed arrays).
        for i in range(11):
            self.xaxis = np.append(
                self.xaxis, self._temp_axis * np.power(10, i + 1))

        # self.probs_per_keV = MuonDistribution().get_probs_per_keV(self.xaxis)

        # Get the energy and angular distribution of muons (or an electron).
        md = MuonDistribution(muon=muon)


        # Get the number of energy points.
        LENGTH = self.xaxis.shape[0]
        # Initialize the arrays consisting the landau distributions for absorbers
        # with different thicknesses.
        self._landaus_5um = mp.Array('d', LENGTH, lock=False)
        self._landaus_10um = mp.Array('d', LENGTH, lock=False)
        self._landaus_15um = mp.Array('d', LENGTH, lock=False)
        self._landaus_20um = mp.Array('d', LENGTH, lock=False)


        # Check how many cpu cores are available. But limit them to five.
        # More would increase the computing time due to the reading/saving of
        # calculated data. It is limited by the bandwidth of the drive.
        num_of_cores = mp.cpu_count() - 2
        if num_of_cores > 4:
            num_of_cores = 4

        # Split the list of zenith angles in smaller list.
        # Create as many lists as parallel threads.
        theta_list = np.arange(180) / 2.
        theta_dict = {}
        for i in range(num_of_cores):
            theta_dict[i] = None

        # TODO: Split the lists by using slices. E.g. with len(list) // num_of_cores.

        while theta_list.shape[0] > 0:
            for i in range(num_of_cores):
                if theta_list.shape[0] > 0:
                    if theta_dict[i] is None:
                        theta_dict[i] = np.array([theta_list[0]])
                    else:
                        theta_dict[i] = np.append(theta_dict[i], theta_list[0])

                    theta_list = theta_list[1:]
                else:
                    break

        # Initialize iterating values.
        procs = list()
        lock = mp.Lock()
        queue = mp.Queue()
        queue.put(0)

        # Set the start time.
        t0 = datetime.now()

        # Define the out, which will be printed during computing.
        out_str = "Calculating"

        # Iterate over all traces to calculate the energy loss of particles with
        # different energies and different arriving angles.
        for i in range(num_of_cores):
            if theta_list is not None:
                # Create a process.
                p = mp.Process(target=multi_func, args=(
                    theta_dict[i], self._landaus_5um, self._landaus_10um,
                    self._landaus_15um, self._landaus_20um, self.xaxis, md,
                    self.get_length, lock, queue, t0, out_str, muon))

                # Add this process to the list of processes.
                procs.append(p)
                # Start the process.
                p.start()

        # Wait until all processes are completed.
        for p in procs:
            p.join()

        # Close the processes and free the memory.
        for p in procs:
            p.kill()
            del p

        # Convert the data to numpy.arrays.
        self.landaus_5um = np.array(self._landaus_5um)
        self.landaus_10um = np.array(self._landaus_10um)
        self.landaus_15um = np.array(self._landaus_15um)
        self.landaus_20um = np.array(self._landaus_20um)

        # Free up memory.
        del self._landaus_5um
        del self._landaus_10um
        del self._landaus_15um
        del self._landaus_20um


        # Normalize the distributions to one.
        self.landaus_5um /= np.trapz(self.landaus_5um, x=self.xaxis)
        self.landaus_10um /= np.trapz(self.landaus_10um, x=self.xaxis)
        self.landaus_15um /= np.trapz(self.landaus_15um, x=self.xaxis)
        self.landaus_20um /= np.trapz(self.landaus_20um, x=self.xaxis)

        # Change the unit of the deposited energies to eV.
        self.xaxis *= 1e3


    def get_length(self, pos, v, d=10.e-6):
        """ Determines the path length of a particle with velocity vector v.

        Args:
            pos (list): The 2d arriving position vector (z = 0).
            v (list): The 3d velocity vector of the particle.
            d (float, optional): The absorber thickness in m. Defaults to 10e-6.

        Returns:
            float: The path length in cm.
        """

        # The dimensions of the absorbers are 180 um * 180 um.
        # It can be splitted in four identical parts a 90 um * 90 um.

        # Define the distance to the edges by considering the velocity direction.
        _d0 = 0.
        _d1 = 0.

        # The position coordinates are both negative.

        #  The top right corner of the absorber quarter is at (0, 0).
        #  y
        #  ^
        #  |
        #  |
        #  |
        #  -----------> x

        #         <----   180 um  ---->
        #        |----------|----------|  ^
        #        |          |          |  |
        #        |          |          |  |
        #        |          |          |
        #     ^  |----------|----------| 180 um
        #     |  |   \    / |          |
        # 90 um  |     X    |          |  |
        #     |  |  /    \  |          |  |
        #     v  |----------|----------|  v
        #        <- 90 um ->


        if v[0] < 0:
            # The distance to the left edge.
            _d0 = 90.e-6 + pos[0]
        elif v[0] > 0:
            # The distance to the right edge.
            _d0 = 90.e-6 - pos[0]
        if v[1] < 0:
            # The distance to the bottom edge.
            _d1 = 90.e-6 + pos[1]
        elif v[0] > 0:
            # The distance to the top edge.
            _d1 = 90.e-6 - pos[1]

        # Note: If v[0] is zero, _d0 is zero. If v[1] is zero, _d1 is zero.

        # Get the x- and y-coordinates of the 'exit' point.
        if v[2] != 0:
            # Meaning after passing the thickness d.
            bound_x = pos[0] + d * v[0] / np.abs(v[2])
            bound_y = pos[1] + d * v[1] / np.abs(v[2])
        else:
            # The z component of the velocity is zero.
            if _d0 != 0:
                # Meaning after passing the left/right edge.

                # pos[0] + _d0 * v[0] / np.abs(v[0])
                # = pos[0] + 90e-6 * v[0] / np.abs(v[0])
                #   - (pos[0] * v[0] / np.abs(v[0]) * (v[0] / np.abs(v[0]))
                # = pos[0] + 90e-6 * v[0] / np.abs(v[0]) - pos[0]
                # = 90e-6 * v[0] / np.abs(v[0])
                bound_x = 90.e-6 * v[0] / np.abs(v[0])
                bound_y = pos[1] + _d0 * v[1] / np.abs(v[0])
            else:
                # v[0] is zero.

                # Meaning after passing the top/bottom edge.
                bound_x = pos[0] # + _d1 * v[0] / np.abs(v[1])
                bound_y = 90.e-6 * v[1] / np.abs(v[1])

        # Only the absolute values matter.
        a_bound_x = np.abs(bound_x)
        a_bound_y = np.abs(bound_y)

        # Note: The path length is returned in cm = 100 * m.
        if a_bound_x >= 89.9e-6:
            # The left/right edge is reached before the thickness is passed.
            if a_bound_y >= 89.9e-6:
                # Also the top/bottom edge is reached before the thickness is passed.
                # Check, which edge is passed before.
                if a_bound_x > a_bound_y:
                    # The left/right edge is passed first.
                    return 1.e2 * _d0 / np.abs(v[0])
                else:
                    # The top/bottom edge is passed first.
                    return 1.e2 * _d1 / np.abs(v[1])
            else:
                # -90e-6 < bound_y < 90e-6 and v[2] * _d0 / np.abs(v[0]) < d.
                return 1.e2 * _d0 / np.abs(v[0])
        elif a_bound_y >= 89.9e-6:
            # The top/bottom edge is reached before the left/right and before
            # passing the thickness:
            # -90e-6 < bound_x < 90e-6 and v[2] * _d1 / np.abs(v[1]) < d.
            return 1.e2 * _d1 / np.abs(v[1])
        else:
            # If the thickness is passed before the edges are reached:
            # -90e-6 < bound_x/y < 90e-6.
            return 1.e2 * d / np.abs(v[2])


