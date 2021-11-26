import os
import sys
import gc
import warnings
import numpy as np
import multiprocessing as mp
import pandas as pd
import holoviews as hv
from scipy import signal
from datetime import datetime

import global_parameters as gl
import pulse_generator as pg

sys.path.append(gl.PATH_TO_PLOTTING)
import PlottingTool as ptool
hv.extension('bokeh')


def multi_func(
        name_list, path_list, queue, data0, data1, lock, amount, out_str, t0):
    """ Fills data0 with pulse numbers and data1 with corresponding pulse amplitudes.
        Can be processed by parallel running threads.

    Args:
        name_list (list like): List of parent directory names of pulse files.
                               E.g. ["1_1000", "1001_2000"]
        path_list (list like): List of pathes to parent directories of pulse files.
                               E.g. ["home/some_path/ADC15/NEGP"]
        queue (mp.Queue): Queue for parallel threads.
        data0 (mp.Array): Array which will be filled with pulse numbers.
        data1 (mp.Array): Array which will be filled with pulse amplitudes.
        lock (mp.Lock): Lock for datawriting to the arrays.
        amount (int): The length of the arrays.
        out_str (string): Output which should be printed during processing.
        t0 (datetime): Start time of this task.
    """
    # TODO 'amount' can be dropped. It can be assigned with data0.shape[0] at the beginning.

    # Postion in data arrays, which should be filled.
    position = 0

    # Loop through all directories
    for i, pulse_path in enumerate(path_list):
        pulse_name = name_list[i]
        # Check if the current directory contains pulses.
        in_loop = gl.is_pulses_dir_(pulse_path, pulse_name)
        if not in_loop:
            continue

        # Loop over all pulses in current directory.
        # Memory can be overflowed due to the break statement.
        # Thus, manually close the iterator at the end.
        pulse_itr = os.scandir(pulse_path)
        for pulse in pulse_itr:
            # Check if current file is a pulse file.
            if pulse.name[-5:] == gl.SRAW:
                try:
                    # Load the pulse.
                    tmp_pulse = pg.PulseFromPath(pulse.path)
                    # Apply a median filter before determining the maximum.
                    data = signal.medfilt(
                        tmp_pulse.Data[gl.FIRST_E:gl.FIRST_E + 1000],
                        kernel_size=3)
                    # Determine the maximum.
                    d = np.float32(data.max())

                    # The pulse are only added to the array, if the maximum
                    # is larger than a threshold.
                    if d > 30:
                        # Start saving the pulse information in the arrays.
                        lock.acquire()
                        try:
                            # Get the last filled wrote position in arrays.
                            position = queue.get()
                        finally:
                            # Check if arrays are already fully filled.
                            if position < amount:
                                data0[position] = tmp_pulse.Pulse
                                data1[position] = d
                                # Print the progress.
                                gl.show_progress(
                                    position, amount, t0, out_str=out_str)
                                # Mark current entry as filled.
                                queue.put(position + 1)
                            else:
                                queue.put(position)
                            # End of data saving.
                            lock.release()
                            # Break if arrays are fully filled.
                            if position >= amount:
                                break

                    # Delete temporary arrays
                    del tmp_pulse
                    del data

                # If pulse file is corrupted skip this file.
                except pg.PulseReadError:
                    continue

        # Close the iterator. Important!
        pulse_itr.close()
        # Break if arrays are fully filled.
        if position >= amount:
            break


class CalibrationEvents:
    """ Determine which pulses should be used to build the template pulse and
        identify dominant lines, which will be used later for energy calibartion.

        Attributes:
            Channel (uint8): The ADC channel.
            Histogram (2d numpy.array): The spectrum of pulse amplitudes.
            Lines (numpy.array): List of five booleans.
                                 First and second entry: If Mn55 K lines are detected.
                                 Third entry: If Ho163 M1 line is detected:
                                 Fourth entry: If Ho163 M2 line is detected.
                                 Fifth entry: If Ho163 N1 line is detected.
            Path (string): The path to the directory containing ADC channel directories.
            Peaks (numpy.array): List of peak positions.
            Pixel (uint8): The pixel number.
            Polarity (string): The polarity of the pulse. Either 'NEGP' or 'POSP'.
            Pulses (pandas.DataFrame): The list of calibration pulses.

        Methods:
            plot(): Save the spectrum of pulse amplitudes.
    """
    def __init__(
        self, path, number, polarity=None, amount=15000, width=(2, 20),
            threshold=160, new=False, verbose=False, rec=0):
        """
        Args:
            path (string): Path to the directory containing ADC channel directories.
            number (int): Either a pixel number or the ADC channel.
            polarity (string, optional): The pixel polarity. Either 'NEGP' or
                                         'POSP'. Defaults to None.
            amount (int, optional): Defines how much pulses should be histogrammed.
                                    Defaults to 15000.
            width (tuple, optional): Defines the range of peak widths.
                                     Defaults to (2, 20).
            threshold (int, optional): Defines the minimum amplitude of peaks.
                                       Lower values means higher amplitudes.
                                       Defaults to 160.
            new (bool, optional): If data should be recalculated instead of
                                  being loaded. Defaults to False.
            verbose (bool, optional): Set to True if more peak information
                                      should be printed. Defaults to False.
            rec (int, optional): Level of recursions. Defaults to 0.
                                 Should not be changed.
        """

        # Determines the pixel properties.
        __pulse = pg.RandomPulse(path, number, polarity=polarity)
        self.Path = __pulse.Path
        self.Channel = __pulse.Channel
        self.Polarity = __pulse.Polarity
        self.Pixel = __pulse.Pixel
        del __pulse

        # Determines the root directory of this pixels
        self.__PathToPulses = os.path.join(
            self.Path, gl.ADC + str(self.Channel), self.Polarity)

        # Defines how many pulses should be histogrammed
        self.__Amount = amount
        # Defines the range of peak widths. If width is not in range, it is no peak.
        self.__Width = width
        # Defines the minimum amplitude of peaks.
        self.__Threshold = threshold

        # Initialize the attributes
        # The list of pulses used for the template
        self.Pulses = None
        # Spectrum of pulse amplitudes
        self.Histogram = None
        # List of peak positions
        self.Peaks = None
        # List of detected lines (Mn55 and Ho163 lines)
        self.Lines = None


        # The maximum pulse amplitude, which should be shown in the histogram
        self.__Max_Length = 1500
        # The bining of the histogram
        self.__Bins = 1000
        self.__generate(new=new, verbose=verbose, rec=rec)

        if rec == 0:
            self.plot()


    def __generate(self, new=False, verbose=False, rec=0):
        """ Searches for pulses of the Mn55 K-alpha and Dy163 M1 line. Returns a pandas
        frame with the numbers of pulses containing of the lines. If both lines,
        the Dy M1 and the Mn K-alpha lines, are found, the pulses of the Dy M1 line
        will be returned.

        Args:
            new (bool, optional): If data should be recalculated instead of
                                  being loaded. Defaults to False.
            verbose (bool, optional): Set to True if more peak information
                                      should be printed. Defaults to False.
            rec (int, optional): Level of recursions. Defaults to 0.
                                 Should not be changed.

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """

        # Column names of self.Pulses
        names = [gl.COLUMN_SIGNAL_NUMBER, gl.COLUMN_MAX_VALUE]

        # Pathes where the calculated data should be saved and loaded from
        pathToIndex = os.path.join(
            self.__PathToPulses, gl.FILE_CALIBRATION_INDEX)
        pathToSignals = os.path.join(
            self.__PathToPulses, gl.FILE_CALIBRATION_SIGNALS)
        pathToHistY = os.path.join(
            self.__PathToPulses, gl.FILE_CALIBRATION_HIST_Y)
        pathToHistX = os.path.join(
            self.__PathToPulses, gl.FILE_CALIBRATION_HIST_X)
        pathToPeaks = os.path.join(
            self.__PathToPulses, gl.FILE_CALIBRATION_PEAKS)
        pathToDetectedLines = os.path.join(
            self.__PathToPulses, gl.FILE_CALIBRATION_DETECTED_LINES)
        pathToLines = os.path.join(
            self.__PathToPulses, gl.FILE_CALIBRATION_LINES)
        pathToDecayTime = os.path.join(
            self.__PathToPulses, gl.FILE_CALIBRATION_DECAY_TIME)

        # Check if data already exists or if it should be recreated
        if os.path.exists(pathToHistY) and (not new):
            # Load saved spectrum of pulse amplitudes
            # The histogram does not change even if the peaks should be recalculated
            self.Histogram = np.array(
                [np.load(pathToHistY), np.load(pathToHistX)])

            if rec == 1:
                # Other attributes will be loaded in the first recursion
                return

        if (os.path.exists(pathToIndex) and (not new)):
            # The histogram could be generated without the determination of the peaks
            # Load the peaks and template pulses
            if pd.read_csv(pathToIndex)[gl.COLUMN_CAL_INDEX][0] == 0:
                # No lines are identified, thus no template pulses exist.
                self.Pulses = None
            else:
                self.Pulses = pd.DataFrame(
                    data={names[0]: np.load(pathToSignals)})
            self.Peaks = np.load(pathToPeaks)
            self.Lines = np.load(pathToDetectedLines)
            return

        # Use old theoretical spectrum (without scattering with electron continuum)
        # It shows better numbers compared to measurement
        # Use energy resolution of ds = 10 eV
        ENERGY_M1 = 2015.7  # eV
        ENERGY_M2 = 1808.9  # eV
        ENERGY_N1 =  402.0  # eV

        ENERGY_MN55_A = 5894  # eV
        ENERGY_MN55_B = 6515  # eV


        def get_dis(e_h, e_l, e_d):
            """ Get the relative differences between two uncertain values.
                Distance_xy = (Energy_y - Energy_x) / Energy_y
                            = Energy_y / Energy_y - Energy_x / Energy_y
                With error:
                    Make left quotient large and right quotient small
                    (Energy_y + de) / (Energy_y - de) - (Energy_x - de) / (Energy_y + de)

                    Or differnet way de -> -de

            Args:
                e_h (number): Higher value.
                e_l (number): Lower value.
                e_d (number): Error of the values.

            Returns:
                number: Difference of the values.
            """

            return ((e_h + e_d) / (e_h - e_d)) - ((e_l - e_d) / (e_h + e_d))

        # Energy resolution = 10 eV -> 30 eV = 3 * 10 eV (3 sigma range)
        def get_dis_max(e_h, e_l):
            return get_dis(e_h, e_l, 30.)


        def get_dis_min(e_h, e_l):
            return get_dis(e_h, e_l, -30.)

        # Calculate bounds of relative distances between lines
        dis_m2m1_max = get_dis_max(ENERGY_M1, ENERGY_M2)
        dis_m2m1_min = get_dis_min(ENERGY_M1, ENERGY_M2)
        dis_n1m1_max = get_dis_max(ENERGY_M1, ENERGY_N1)
        dis_n1m1_min = get_dis_min(ENERGY_M1, ENERGY_N1)
        dis_n1m2_max = get_dis_max(ENERGY_M2, ENERGY_N1)
        dis_n1m2_min = get_dis_min(ENERGY_M2, ENERGY_N1)

        dis_mn55_max = get_dis_max(ENERGY_MN55_B, ENERGY_MN55_A)
        dis_mn55_min = get_dis_min(ENERGY_MN55_B, ENERGY_MN55_A)

        # Values calculated with theory: Ration of counts in peaks +- 3 ds
        RATIO_M2M1 = 12.052
        RATIO_N1M2 = 0.037
        RATIO_N1M1 = 0.452
        RATIO_MN55 = 0.271

        # Modifier of distance bounds
        MULTIPLY_L = 0.9
        MULTIPLY_H = 1.1


        def get_ratios(ratio, mode="/"):
            """ Calculate the bounds of for ratios.
                Example: With measured Counts_m2 = 4 -> Expect Counts_m1 = 12 * 4 = 48
                         ratio_m2m1_min = (48 - 1.5 * sqrt(48)) / (4 + 3.0 * sqrt(4))
                         ratio_m2m1_max = (48 + 3.0 * sqrt(48)) / (4 - 1.5 * sqrt(4))

                         Four counts is here the threshold for peaks. The error
                         is coming from poisson statistics (sqrt(N)). If the
                         threshold is higher, the bounds will be smaller.

            Args:
                ratio (float): A peak-peak ratio.
                mode (str, optional): Defines if peak including 4 counts is the
                                      larger or smaller peak. Either '*' or
                                      '/'. If ratio > 1 use '*'. Defaults to "/".

            Raises:
                ValueError: If mode is not one of '*' and '/'.

            Returns:
                tuple: Ratio bounds.
            """

            # TODO: Drop argument 'mode'. Use ratio > 1 instead.
            # Calculate the height of the second peak.
            if mode == "*":
                # If ratio > 1
                c_l = 4.
                c_h = c_l * ratio
            elif mode == "/":
                c_h = 4.
                c_l = c_h / ratio
            else:
                raise ValueError(
                    "Argument 'mode' has to be either '*' or '/'.")

            r_min = (c_h - 1.5 * np.sqrt(c_h)) / (c_l + 3.0 * np.sqrt(c_l))
            r_max = (c_h + 3.0 * np.sqrt(c_h)) / (c_l - 1.5 * np.sqrt(c_l))

            return (r_min * MULTIPLY_L, r_max * MULTIPLY_H)


        # Calcultate bounds for distances and ratios.
        distance_m1_m2 = (dis_m2m1_min, dis_m2m1_max)
        amplitude_m1_m2 = get_ratios(RATIO_M2M1, mode="*")

        distance_n1_m2 = (dis_n1m2_min, dis_n1m2_max)
        amplitude_n1_m2 = get_ratios(RATIO_N1M2)

        distance_n1_m1 = (dis_n1m1_min, dis_n1m1_max)
        amplitude_n1_m1 = get_ratios(RATIO_N1M1)

        distance_kalpha_kbeta = (dis_mn55_min, dis_mn55_max)
        amplitude_kalpha_kbeta = get_ratios(RATIO_MN55)

        # Define minimum signal heights and distances in mV for different lines
        # A modifier
        # delta_f = 0.1 # 0.8
        min_n1 = 50         # 40 * delta_f
        min_m1 = 250        # 330 * delta_f
        min_mn55_a = 640    # 875 * delta_f
        min_delta_mm = 20   # 30 * delta_f
        min_delta_nm1 = 170 # 260 * delta_f
        min_delta_nm2 = 150 # 230 * delta_f
        min_delta_mn55 = 60 # 600 * delta_f

        def line_exists(ratio_pos, ratio_counts, limits_pos, limits_counts):
            """ Check if a set of two lines characterized by ratio_pos and
                ratio_counts fullfills the bound conditions.

            Args:
                ratio_pos (number): The distance ratio of two peaks.
                ratio_counts (number): The amplitude ratio of two peaks.
                limits_pos (tuple): Bounds of the distance ratio.
                limits_counts (tuple): Bounds of the amplitude ratio.

            Returns:
                boolean: True if the two peaks fullfill the bound conditions.
            """

            out = (limits_pos[0] <= ratio_pos)
            out = out and (limits_pos[1] >= ratio_pos)
            out = out and (limits_counts[0] <= ratio_counts)
            out = out and (limits_counts[1] >= ratio_counts)
            return out


        # Initialize the arrays containing the signal numbers and signal amplitudes.
        temp_data_0 = mp.Array('Q', self.__Amount, lock=False)
        temp_data_1 = mp.Array('f', self.__Amount, lock=False)

        # Initialize iterating values
        # TODO: The initialization of position should not be needed.
        position = 0
        procs = list()
        queue = mp.Queue()
        queue.put(0)
        lock = mp.Lock()

        # Define the out, which will be printed during computing.
        out_str = "Pixel " + str(self.Pixel)
        out_str += ": Looking for calibration lines"

        # Set the start time.
        t0 = datetime.now()

        print(out_str + (80 - len(out_str)) * " ", end="\r")

        # Create a list of all pulse files and their pathes.
        path_list = None
        name_list = None
        for pulses in os.scandir(self.__PathToPulses):
            if path_list is None:
                path_list = np.array([pulses.path])
                name_list = np.array([pulses.name])
            else:
                path_list = np.append(path_list, pulses.path)
                name_list = np.append(name_list, pulses.name)

        # Split the list of pulse files in smaller list.
        # Create as many lists as parallel threads.
        path_dict = {}
        name_dict = {}

        # Check how many cpu cores are available. But limit them to five.
        # More would increase the computing time due to the reading of the pulses.
        # It is limited by the bandwidth of the drive.
        num_of_cores = mp.cpu_count() - 2
        if num_of_cores > 5:
            num_of_cores = 5
        for i in range(num_of_cores):
            path_dict[i] = None
            name_dict[i] = None

        # TODO: Split the lists by using slices. E.g. with len(list) // num_of_cores.

        while path_list.shape[0] > 0:
            for i in range(num_of_cores):
                if path_list.shape[0] > 0:
                    if path_dict[i] is None:
                        path_dict[i] = np.array([path_list[0]])
                        name_dict[i] = np.array([name_list[0]])
                    else:
                        path_dict[i] = np.append(path_dict[i], path_list[0])
                        name_dict[i] = np.append(name_dict[i], name_list[0])

                    path_list = path_list[1:]
                    name_list = name_list[1:]
                else:
                    break

        # Iterate over all traces get the maximum value of the pulses and their
        # pulse number.
        for i in range(num_of_cores):
            if path_dict[i] is not None:
                # Create a process.
                p = mp.Process(target=multi_func, args=(
                    name_dict[i], path_dict[i], queue, temp_data_0, temp_data_1,
                    lock, self.__Amount, out_str, t0))
                # Add this process to the list of processes.
                procs.append(p)
                # Start the process.
                p.start()

        # Wait until all processes are completed.
        for p in procs:
            p.join()

        # Close the processes and free the memory.
        for p in procs:
            p.close()
            del p

        # Get the last entry of the data arrays, which is filled.
        position = queue.get()
        queue.close()
        print(out_str + " -- Done", end="\r")

        # Remove all not filled entries. This could be the case if
        # self.__Amount is larger than the number of the recorded pulses.
        if position < self.__Amount:
            temp_data_0 = temp_data_0[:position]
            temp_data_1 = temp_data_1[:position]

        # Convert the arrays to a dictionary.
        temp_data = {names[0]: np.array(temp_data_0[:]),
                     names[1]: np.array(temp_data_1[:])}

        # Convert the dictionary in a pandas.DataFrame.
        temp_frame = pd.DataFrame(data=temp_data)
        # Histogrammize the pulse amplitudes.
        hist_data = np.histogram(
            temp_frame[gl.COLUMN_MAX_VALUE], bins=self.__Bins,
            range=(0, self.__Max_Length))

        # Create the voltage axis (x-axis):
        tmp_arr = np.arange(self.__Bins).astype(np.float32) + 1
        tmp_arr *= hist_data[1][1] - hist_data[1][0]

        # Assign the histogram to the attribute and save it.
        self.Histogram = np.array(
            [hist_data[0].astype(np.float32), tmp_arr])
        np.save(pathToHistY, self.Histogram[0])
        np.save(pathToHistX, self.Histogram[1])

        # Both pixels of a channel should be analzed together.
        # Thus, get the spectrum of the pulse amplitdes of the partner pixel.
        if rec == 0:
            other_pol = gl.NEGP
            if self.Polarity == gl.NEGP:
                other_pol = gl.POSP
            if os.path.exists(os.path.join(
                    self.Path, gl.ADC + str(self.Channel), other_pol)):
                other_cal = CalibrationEvents(
                    self.Path, self.Channel, polarity=other_pol, new=new,
                    verbose=verbose, rec=1)
                # Add the histogram.
                self.Histogram[0] += other_cal.Histogram[0]
            else:
                other_cal = None
        else:
            return


        # Find the positions of the local maxima/peaks.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            peak_locs = signal.find_peaks_cwt(
                self.Histogram[0],
                np.arange(self.__Width[0], self.__Width[1]),
                min_snr=0)


        # Calculate the total number of counts of the peak.
        # The width of the peaks.
        width = int(4)

        # Iterate over all peak positions.
        # TODO: This loop does not do anything. Probably a left over from the past.
        # for peak_pos in peak_locs:
        #    # Initialize the maximum peak amplitude.
        #    tmp_amp = None

        #    # Look in the -+ width range for the maximum.
        #    # Due to the amplitude threshold, peak_pos should not be smaller then width.
        #    for w in range(2 * width):
        #        pos = int(peak_pos - width + w)
        #        if pos >= len(self.Histogram[0]):
        #            # Out of range.
        #            # TODO: break statement should be better.
        #            continue

        #        # Assign tmp_amp to the maximum amplitude.
        #        if tmp_amp is None:
        #            tmp_amp = self.Histogram[0][pos]
        #        else:
        #            if tmp_amp < self.Histogram[0][pos]:
        #                tmp_amp = self.Histogram[0][pos]

        # The list of peak areas.
        amplitudes = np.zeros(1).astype(np.float32)

        # Iterate over all peak positions.
        for i in peak_locs:
            # i - width has to be larger than 0.
            # TODO: Change condition to i > width.
            # In the past slices were [i - 2 * width:i]
            if i >= 2 * width:
                tmp_amps = np.copy(
                    self.Histogram[0][int(i - width):int(i + width + 1)])
                tmp_amps = tmp_amps.sum()
            else:
                tmp_amps = 0

            # Define a threshold for peak amplitudes.
            if tmp_amps < 4:
                tmp_amps = 0

            amplitudes = np.append(amplitudes, tmp_amps)
        amplitudes = amplitudes[1:]

        # Peaks have to have a height of threshold or more.
        # Threshold is defined by the heighest peak.
        # Check if there are nay peaks detected.
        if amplitudes.shape[0] > 0:
            _threshold = amplitudes.max() / self.__Threshold
            self.Peaks = peak_locs[amplitudes > _threshold].astype(np.float32)
            amplitudes = amplitudes[amplitudes > _threshold]
            no_peak = False
        else:
            self.Peaks = np.zeros(1)
            no_peak = True
        # Save the peak positions.
        np.save(pathToPeaks, self.Peaks)

        # 5 Known lines: Mn55-K_Alpha, Mn55-K_Beta, Ho-M1, Ho-M2 and Ho-N1
        self.Lines = np.zeros(5).astype(np.float32)

        # Generate lists for these five lines. Fill them with peaks, which
        # could correspond to the lines.

        # Mn-55 K_Alpha and K_Beta pair.
        # The peak energies.
        energy_iron_0 = []
        energy_iron_1 = []
        # The peak positions.
        index_iron_0 = []
        # The sum of counts of both peaks.
        iron_counts = []

        # Ho-163 M1 and M2 pair.
        energy_mm_0 = []
        energy_mm_1 = []
        index_mm_0 = []
        index_mm_1 = []
        mm_counts = []

        # Ho-163 M1 and N1 pair.
        energy_nm1_0 = []
        energy_nm1_1 = []
        index_nm1_0 = []
        index_nm1_1 = []
        nm1_counts = []

        # Ho-163 M2 and N1 pair.
        energy_nm2_0 = []
        energy_nm2_1 = []
        index_nm2_0 = []
        index_nm2_1 = []
        nm2_counts = []

        # Go through all peaks.
        # TODO: Change range with enumerate.
        for this_peak_id in range(self.Peaks.shape[0] - 1):
            if no_peak:
                break
            # The peak position.
            this_index = int(self.Peaks[this_peak_id])
            # The peak energy.
            this_energy = self.Histogram[1][this_index]
            # The peak amplitude.
            this_counts = amplitudes[this_peak_id]

            # Loop over all other peaks at higher energies and check if the
            # pair of two peaks fullfills some line-conditions.
            # TODO: Change range with enumerate.
            for next_peak_id in range(self.Peaks.shape[0] - this_peak_id - 1):
                # The other peak position.
                next_index = int(self.Peaks[next_peak_id + this_peak_id + 1])
                # The other peak energy.
                next_energy = self.Histogram[1][next_index]
                # The other peak amplitude.
                next_counts = amplitudes[next_peak_id + this_peak_id + 1]

                # Calculate the relative distance of the two peaks.
                energy_distance = (next_energy - this_energy) / next_energy
                # Calculate the peak amplitude ratio.
                counts_ratio = next_counts / this_counts
                if verbose:
                    print("Peak " + str(this_peak_id) + " - Peak " +
                          str(next_peak_id + this_peak_id + 1) + ": ")
                    print("Energy distance: " + str(energy_distance)[:5])
                    print("Counts ratio: " + str(counts_ratio)[:5])

                # Check for each line, if the pair of peaks can be assigned to
                # the line. Also, check if the minimum properties are fullfilled:
                # The minimum absolute distance between the peaks and minimum
                # absolute voltage.

                # For the Mn-55 lines.
                iron_line = line_exists(
                    energy_distance, counts_ratio,
                    distance_kalpha_kbeta, amplitude_kalpha_kbeta)
                if iron_line and \
                        (next_energy - this_energy > min_delta_mn55) and \
                        (this_energy > min_mn55_a):
                    if verbose:
                        print("KAlpha and KBeta lines are found.")
                        # There are other peaks, which could also be assigned to
                        # these lines.
                        if energy_iron_0 != []:
                            print("Warning: KAlpha and KBeta lines are " +
                                  "not well defined!")

                    if energy_iron_0 is None:
                        energy_iron_0 = [this_energy]
                        energy_iron_1 = [next_energy]
                        index_iron_0 = [this_index]
                        iron_counts = [this_counts + next_counts]
                    else:
                        energy_iron_0.append(this_energy)
                        energy_iron_1.append(next_energy)
                        index_iron_0.append(this_index)
                        iron_counts.append(this_counts + next_counts)

                # For the M1 and M2 lines.
                mm_line = line_exists(
                    energy_distance, counts_ratio,
                    distance_m1_m2, amplitude_m1_m2)
                if mm_line and \
                        (next_energy - this_energy > min_delta_mm) and \
                        (next_energy > min_m1):
                    if verbose:
                        print("M1 and M2 lines are found.")
                        if energy_mm_0 != []:
                            print("Warning: M1 and M2 lines are " +
                                  "not well defined!")
                    energy_mm_0.append(this_energy)
                    energy_mm_1.append(next_energy)
                    index_mm_0.append(this_index)
                    index_mm_1.append(next_index)
                    mm_counts.append(this_counts + next_counts)

                # For the N1 and M1 lines.
                nm1_line = line_exists(
                    energy_distance, counts_ratio,
                    distance_n1_m1, amplitude_n1_m1)
                if nm1_line and \
                        (next_energy - this_energy > min_delta_nm1) and \
                        (next_energy > min_m1) and (this_energy > min_n1):
                    if verbose:
                        print("N1 and M1 lines are found.")
                        if energy_nm1_0 != []:
                            print("Warning: N1 and M1 lines are " +
                                  "not well defined!")
                    energy_nm1_0.append(this_energy)
                    energy_nm1_1.append(next_energy)
                    index_nm1_0.append(this_index)
                    index_nm1_1.append(next_index)
                    nm1_counts.append(this_counts + next_counts)

                # For the N1 and M2 lines.
                nm2_line = line_exists(
                    energy_distance, counts_ratio,
                    distance_n1_m2, amplitude_n1_m2)
                if nm2_line and \
                        (next_energy - this_energy > min_delta_nm2) and \
                        (this_energy > min_n1):
                    if verbose:
                        print("N1 and M2 lines are found.")
                        if energy_nm2_0 != []:
                            print("Warning: N1 and M2 lines are " +
                                  "not well defined!")
                    energy_nm2_0.append(this_energy)
                    energy_nm2_1.append(next_energy)
                    index_nm2_0.append(this_index)
                    index_nm2_1.append(next_index)
                    nm2_counts.append(this_counts + next_counts)

            # TODO: Why is this commented?
            # amplitudes will not be used later.
            # del amplitudes

            # Define which peaks should be assigned to the lines.
            energy_m1 = None
            index_m1 = None
            energy_m2 = None
            energy_n1 = None
            index_n1 = None

            break_loop = False

            max_nm_index = -1
            max_nm_count = 0

            # Iterate over all N1/M1 candidates.
            # TODO: General, change range with enumerate.
            for i in range(len(energy_nm1_0)):
                # Get the current set of properties of the N1 and M1 candidates.
                # The energy of the M1 peak.
                tmp_m1 = energy_nm1_1[i]
                # The energy of the N1 peak.
                tmp_n1 = energy_nm1_0[i]
                # The position of the M1 peak.
                tmp_im = index_nm1_1[i]
                # The position of the N1 peak.
                tmp_in = index_nm1_0[i]
                # The sum of counts of both peaks.
                tmp_max_nm = nm1_counts[i]

                # Check if the M2 line is also detected.
                # If yes, check if the N1 and M1 peaks are compatible with the
                # M2 line.
                # Iterate over all N1/M2 candidates.
                if energy_nm2_0 != []:
                    for j in range(len(energy_nm2_0)):
                        # The energy of the M2 peak.
                        tmp_m2 = energy_nm2_1[j]
                        # The energy of the second N1 peak.
                        second_n1 = energy_nm2_0[j]

                        # If a pair of N1/M2 is detected, the pair of M1/M2
                        # should also be detected.
                        if energy_mm_0 != []:
                            # Iterate over all M1/M2 candidates.
                            for k in range(len(energy_mm_0)):
                                # The energy of the second M2 peak.
                                second_m2 = energy_mm_0[k]
                                # The energy of the second M1 peak.
                                second_m1 = energy_mm_1[k]

                                # If all three N1, M1 and M2 candidates fullfill
                                # all conditions, the probability that these peaks
                                # correspond to the lines is very high.
                                if (tmp_m2 == second_m2) and \
                                        (tmp_m1 == second_m1) and \
                                        (tmp_n1 == second_n1):
                                    energy_m1 = tmp_m1
                                    index_m1 = tmp_im
                                    energy_n1 = tmp_n1
                                    index_n1 = tmp_in
                                    energy_m2 = tmp_m2
                                    # The N1, M1 and M2 lines are identified.
                                    break_loop = True
                                    break
                                # TODO: Check order of following conditions.
                                # TODO: Check if the sollowing assumption is correct.
                                # If two peaks are the same, the third should
                                # also be the same.
                                # If only the M2 line is the same.
                                elif (tmp_m2 == second_m2):
                                    energy_m1 = second_m1
                                    index_m1 = index_mm_1[k]
                                    energy_n1 = second_n1
                                    index_n1 = index_nm2_0[j]
                                    energy_m2 = tmp_m2
                                # If only the N1 line is the same.
                                elif (tmp_n1 == second_n1):
                                    energy_m1 = tmp_m1
                                    index_m1 = tmp_im
                                    energy_n1 = tmp_n1
                                    index_n1 = tmp_in
                                    energy_m2 = tmp_m2
                                # If only the M1 line is the same.
                                elif (tmp_m1 == second_m1):
                                    energy_m1 = tmp_m1
                                    index_m1 = tmp_im
                                    energy_n1 = tmp_n1
                                    index_n1 = tmp_in
                                    energy_m2 = second_m2

                            if break_loop:
                                # All lines are identified.
                                break
                        # No M1/M2 pairs are detected.
                        # Check if the two N peaks are the same.
                        else:
                            if second_n1 == tmp_n1:
                                energy_m1 = tmp_m1
                                index_m1 = tmp_im
                                energy_n1 = tmp_n1
                                index_n1 = tmp_in
                                energy_m2 = tmp_m2
                                break_loop = True
                                break

                # The N line is not detected, but maybe the M2 line.
                elif energy_mm_0 != []:
                    for j in range(len(energy_mm_0)):
                        # The energy of the M2 peak.
                        tmp_m2 = energy_mm_0[j]
                        # The energy of the second M1 peak.
                        second_m1 = energy_mm_1[j]

                        # Check if the two M1 peaks are identical.
                        if second_m1 == tmp_m1:
                            energy_m1 = tmp_m1
                            index_m1 = tmp_im
                            energy_n1 = tmp_n1
                            index_n1 = tmp_in
                            energy_m2 = tmp_m2
                            break_loop = True
                            break

                else:
                    # The M2 line is not detected.
                    # Look for the highest line candidates.
                    if tmp_max_nm > max_nm_count:
                        max_nm_count = tmp_max_nm
                        max_nm_index = i

                if break_loop:
                    break

            # Assign the peaks with highest amplitudes to Ho-163 lines, if none
            # of the peaks are identified by at least two conditions.
            if max_nm_index >= 0:
                energy_m1 = energy_nm1_1[max_nm_index]
                index_m1 = index_nm1_1[max_nm_index]
                energy_n1 = energy_nm1_0[max_nm_index]
                index_n1 = index_nm1_0[max_nm_index]

            # The Mn-55 line candidates with highest energies are assigned to
            # the lines.
            if energy_iron_0:
                self.Lines[0] = energy_iron_0[-1]
                self.Lines[1] = energy_iron_1[-1]
                # Define the width of the line.
                delta_iron = self.Histogram[1][index_iron_0[-1] + width] - \
                        self.Lines[0]

                # If the Ho-163 M1 line is identified, the Mn-55 lines have to
                # have higher energies than the M1 line.
                # At low energies, many peaks could be occur and could be assigned
                # to Mn-55 lines else.
                if energy_m1 is not None:
                    if energy_m1 > energy_iron_0[-1]:
                        energy_iron_0 = None
                        self.Lines[0] = 0
                        self.Lines[1] = 0

            # Assign the Ho-163 lines.
            if energy_m1 is not None:
                self.Lines[2] = energy_m1
                # Define the width of the line.
                delta_m = self.Histogram[1][index_m1 + width] - self.Lines[2]

            if energy_m2 is not None:
                self.Lines[3] = energy_m2

            if energy_n1 is not None:
                self.Lines[4] = energy_n1
                # Define the width of the line.
                delta_n = self.Histogram[1][index_n1 + width] - self.Lines[4]

            # End of line identification.
            if verbose:
                print("##########################################")


        def get_frame(position, delta):
            """ Get the pulse numbers of pulses with amplitudes in a given range.

            Args:
                position (number): The mean peak amplitude.
                delta (number): The width of the peak.

            Returns:
                array like: List of pulses.
            """

            return temp_frame[
                    (temp_frame[gl.COLUMN_MAX_VALUE] > position - delta) &
                    (temp_frame[gl.COLUMN_MAX_VALUE] < position + delta)][
                        gl.COLUMN_SIGNAL_NUMBER]


        # Pulses of the lines will be used to build the template. cindex defines
        # which line should be used.
        cindex = 0

        # Get pulses of the M1 line.
        if self.Lines[2] != 0.:
            pulse_frame = get_frame(self.Lines[2], delta_m)
            if pulse_frame.shape[0] > 0:
                cindex = 1
            if verbose:
                print("The M1 line will be used to built the template.")

        # Get pulses of the K alpha line.
        elif self.Lines[0] != 0.:
            pulse_frame = get_frame(self.Lines[0], delta_iron)
            if pulse_frame.shape[0] > 0:
                cindex = 2
            if verbose:
                print("The KAlpha line will be used to built the template.")

        # Get pulses of the N1 line.
        elif self.Lines[4] != 0.:
            pulse_frame = get_frame(self.Lines[4], delta_n)
            if pulse_frame.shape[0] > 0:
                cindex = 3
            if verbose:
                print("The N1 line will be used to built the template.")
        else:
            if verbose:
                print("Warning: Did not either find the KAlpha line" +
                      " nor the M1 line nor the N1 line!")
                print("The first 1000 pulses will be used to built " +
                      "the template.")
        # Save the calibration index.
        pd.DataFrame(data=[(cindex)], columns=[gl.COLUMN_CAL_INDEX]).to_csv(
            pathToIndex, index=False)

        # If a line is used (and identified) for the template. Save pulses of
        # this line.
        if cindex > 0:
            np.save(pathToSignals, np.array(pulse_frame).astype(np.int))
        # Save the list of identified lines.
        np.save(pathToDetectedLines, self.Lines)

        # Save if Ho-163 and Mn-55 is detected.
        lines_frame = pd.DataFrame(
            data=np.zeros(shape=(1, 2)), columns=[gl.COLUMN_K, gl.COLUMN_M])
        if self.Lines[0] != 0:
            lines_frame[gl.COLUMN_K][0] = 1
        if self.Lines[2] != 0:
            lines_frame[gl.COLUMN_M][0] = 1
        lines_frame.to_csv(pathToLines, index=False)

        # Assign the others attributes.
        if cindex > 0:
            self.Pulses = pd.DataFrame(
                data={gl.COLUMN_SIGNAL_NUMBER: np.array(pulse_frame)})

        # IF the partner pixel was used, remove his data.
        if (rec == 0) and (other_cal is not None):
            self.Histogram[0] -= other_cal.Histogram[0]
            del other_cal


        # Free up some memory.
        del lock
        del queue
        del temp_data_0
        del temp_data_1
        del hist_data
        del peak_locs
        del lines_frame[gl.COLUMN_K]
        del lines_frame[gl.COLUMN_M]
        del lines_frame
        if cindex > 0:
            del pulse_frame

        del temp_data[names[0]]
        del temp_data[names[1]]
        del temp_data


    def plot(self):
        """ Save the spectrum of pulse amplitudes and the detected lines.
        """

        # Set some labels.
        ylabel = "Counts per " + \
            str(float(self.__Max_Length / self.__Bins))[:3] + " mV"
        xlabel = "Amplitude in mV"
        title = "Pixel " + str(self.Pixel)

        # Generate the spectrum.
        my_hist = ptool.Histogram(
            self.Histogram[1], ydata=self.Histogram[0],
            ylabel=ylabel, xlabel=xlabel, title=title)

        # Define some location propertiers for the line markings and labels
        # Distance between marking and label of the Ho-163 lines.
        delta_pos = self.__Max_Length * 0.15 / 8
        # Distance between marking and label of the Mn-55 lines.
        delta_pos_k = self.__Max_Length * 0.15 / 5

        # Picture height.
        max_height = self.Histogram[0].max()
        # Y-position of the labels.
        height_pos = max_height * 0.6
        height_pos_k = max_height * 0.9

        # Generate markings and labels for the markings.
        vlines = None
        texts = None

        # The transparency of the markings.
        p_alpha = 0.5
        # The rotation of the labels.
        p_angle = 0.

        # Draw all detected peaks.
        if self.Peaks is not None:
            for i in self.Peaks:
                if vlines is None:
                    vlines = ptool.VLine(
                        self.Histogram[1][int(i)], color="red", alpha=p_alpha)
                else:
                    vlines *= ptool.VLine(
                        self.Histogram[1][int(i)], color="red", alpha=p_alpha)

        # Draw markings at the position of the caracteristic lines
        # TODO: This should be replaced by a loop.
        # The Mn-55 K lines.
        if self.Lines[0] != 0:
            color = "purple"
            if vlines is None:
                vlines = ptool.VLine(
                    self.Lines[0], color=color, alpha=p_alpha)
            else:
                vlines *= ptool.VLine(
                    self.Lines[0], color=color, alpha=p_alpha)
            if texts is None:
                texts = ptool.Text(
                    self.Lines[0] - delta_pos_k, height_pos_k, "K_Alpha",
                    color=color, angle=p_angle)
            else:
                texts *= ptool.Text(
                    self.Lines[0] - delta_pos_k, height_pos_k, "K_Alpha",
                    color=color, angle=p_angle)

            vlines *= ptool.VLine(
                self.Lines[1], color=color, alpha=p_alpha)
            texts *= ptool.Text(
                self.Lines[1] + delta_pos_k, height_pos_k, "K_Beta",
                color=color, angle=p_angle)

        # The Ho-163 M1 line.
        if self.Lines[2] != 0:
            color = "green"
            if vlines is None:
                vlines = ptool.VLine(
                    self.Lines[2], color=color, alpha=p_alpha)
            else:
                vlines *= ptool.VLine(
                    self.Lines[2], color=color, alpha=p_alpha)
            if texts is None:
                texts = ptool.Text(
                    self.Lines[2] + delta_pos, height_pos, "M1",
                    color=color, angle=p_angle)
            else:
                texts *= ptool.Text(
                    self.Lines[2] + delta_pos, height_pos, "M1",
                    color=color, angle=p_angle)

        # The Ho-163 N1 line.
        if self.Lines[4] != 0:
            color = "brown"
            if vlines is None:
                vlines = ptool.VLine(
                    self.Lines[4], color=color, alpha=p_alpha)
            else:
                vlines *= ptool.VLine(
                    self.Lines[4], color=color, alpha=p_alpha)
            if texts is None:
                texts = ptool.Text(
                    self.Lines[4] + delta_pos, height_pos, "N1",
                    color=color, angle=p_angle)
            else:
                texts *= ptool.Text(
                    self.Lines[4] + delta_pos, height_pos, "N1",
                    color=color, angle=p_angle)

        # The Ho-163 M2 line.
        if self.Lines[3] != 0:
            color = "orange"
            vlines *= ptool.VLine(
                self.Lines[3], color=color, alpha=p_alpha)
            texts *= ptool.Text(
                self.Lines[3] - delta_pos, height_pos, "M2",
                color=color, angle=p_angle)

        out = my_hist

        if vlines is not None:
            out *= vlines
        if texts is not None:
            out *= texts

        out.save( os.path.join(
            self.Path, "Calibration_" + str(self.Pixel) + ".png"))

        # TODO: Add option to show plot.


    def __del__(self):
        """ Free up memory. """

        if self.Pulses is not None:
            del self.Pulses[gl.COLUMN_SIGNAL_NUMBER]
            del self.Pulses
        del self.Histogram
        del self.Peaks
        del self.Lines


def generate_calibrations(path, new=False, verbose=False):
    """ Determines the calibration lines for all pixels in path.

    Args:
        path (string): Path to the parent directory containing ADC channel directories.
        new (bool, optional): If True, recreate the calibration lines instead
                              of loading. Defaults to False.
        verbose (bool, optional): Set to True if more peak information
                                  should be printed. Defaults to False.
    """

    # Path to the file containing the ADC channel and polarity of each pixel.
    pathToCSV = os.path.join(path, gl.FILE_PIXEL_LIST)
    # The current data set.
    day = path.split(gl.PATH_SEP)[-1]
    print(day)

    # If this file does not exist, create it by loading a pulse.
    if not os.path.exists(pathToCSV):
        try:
            _ = pg.RandomPulse(path, 1)
            del _
        except pg.PixelNotFoundError:
            pass

    # Load the file.
    frame = pd.read_csv(pathToCSV)

    # Iterate over all pixels and create/load the calibration pulses for each.
    for i in range(len(frame[gl.COLUMN_ADC_CHANNEL])):
        print(day + ": Pixel " +
              str(frame[gl.COLUMN_PIXEL_NUMBER].iloc[i]))

        _ = CalibrationEvents(
            path, frame[gl.COLUMN_ADC_CHANNEL].iloc[i],
            polarity=frame[gl.COLUMN_POLARITY].iloc[i],
            new=new, verbose=verbose)

        _.plot()
        del _

        gc.collect()

    frame = None
    print("Finished " + day)


def generate_calibrations_days(path, days, new=False, verbose=False):
    """ Generate the calibration files for all pixels of different data sets.

    Args:
        path (string): Path to the parent directory of data sets.
        days (list like): List of names of data sets.
        new (bool, optional): If True, recreate the calibration lines instead
                              of loading. Defaults to False.
        verbose (bool, optional): Set to True if more peak information
                                  should be printed. Defaults to False.
    """

    for day in days:
        generate_calibrations(
            os.path.join(path, day), new=new, verbose=verbose)

    print("Finished all days")
