import os
import sys
import gc
import numpy as np
import multiprocessing as mp
import pandas as pd
from scipy.optimize import curve_fit
from scipy import signal
from datetime import datetime

import global_parameters as gl
import pulse_generator as pg
import calibration_generator as cg

sys.path.append(gl.PATH_TO_PLOTTING)
import PlottingTool as ptool
from TexToUni import tex_to_uni
from tqdm import tqdm

# TODO: The assignment of tasks for the cpu cores is identical to the one in
# calibration_generator. Maybe it could be generalized in global_parameters.

# TODO: Change name of data_x
def multi_func(
    path_list, name_list, queue, lock,
        data2, data3, data4, data5, data6, data7, data8, data9,
        decay_time, template, full_template, filt, last, lon,
        length, out_str, t0, average_filt):
    """ Calculate pulse shape parameters of each pulse.

    Args:
        path_list (list like): List of pathes to parent directories of pulse files.
                               E.g. ["home/some_path/ADC15/NEGP"]
        name_list (list like): List of parent directory names of pulse files.
                               E.g. ["1_1000", "1001_2000"]
        queue (mp.Queue): Queue for parallel threads.
        lock (mp.Lock): Lock for datawriting to the arrays.
        data2 (mp.Array): Array, which will be filled with pulse numbers.
        data3 (mp.Array): Array, which will be filled with trigger times.
        data4 (mp.Array): Array, which will be filled with the amplitude of the
                          response function of the auto-correlation.
        data5 (mp.Array): Array, which will be filled with template fit amplitudes.
        data6 (mp.Array): Array, which will be filled with the maximum gradient of the pulses.
        data7 (mp.Array): Array, which will be filled with integral of the pulses.
        data8 (mp.Array): Array, which will be filled with template fit Chi2.
        data9 (mp.Array): Array, which will be filled with pulse decay times.
        decay_time (number): Decay time of the pulses.
        template (numpy.array): The first 1000 entries of the template pulse.
        full_template (numpy.array): The template pulse.
        filt (numpy.array): The kernel for matched filtering.
        last (int): The length of the pulse fraction, which should be analyzed.
        lon (int): The maximal length of the pulse fraction, which should be analyzed.
        length (int): The length of the arrays.
        out_str (string): Output which should be printed during processing.
        t0 (datetime): Start time of this task.
        average_filt (numpy.array): The kernel for averaged filtering.
    """

    # TODO: 'length' can be dropped. It can be assigned with data0.shape[0] at the beginning.

    # Loop through all directories
    for i, pulse_path in enumerate(path_list):
        pulse_name = name_list[i]
        # Check if the current directory contains pulses.
        if gl.is_pulses_dir_(pulse_path, pulse_name):
            for pulse in os.scandir(pulse_path):
                try:
                    # Load the pulse.
                    # USe PulseFromPath, since it is faster than Pulse.
                    the_pulse = pg.PulseFromPath(
                        pulse.path, decay_time=decay_time)

                    # Ignore the first firstE entries, they only contain baseline.
                    data = the_pulse.Data[gl.FIRST_E:]

                    # Fit the template to the pulse.
                    # Use lambda, since it is faster than pointing to a function.
                    try:
                        popt, pcov = curve_fit(
                            lambda x, amp: amp * template,
                            None, data[3000:4000], p0=[1.])

                        popt_chi, pcov = curve_fit(
                            lambda x, amp: amp * full_template,
                            None, data[:4000], p0=[1.])

                        # Calculate the reduced chi squared. Use the full
                        # template, so that the Chi2 is more meaningfull.
                        template_chi = \
                                (data[:4000] - full_template * popt_chi[0])**2
                        template_chi = template_chi.sum()
                        if the_pulse.Sigma**2 > 0:
                            template_chi /= the_pulse.Sigma**2

                        # == template_chi /= (len(data) - 2)
                        template_chi /= 3998

                        # Covariance matrix will not be used.
                        del pcov

                    except RuntimeError:
                        # If the pulse can not be described by the template.
                        popt = np.array([1e8])
                        popt_chi = np.array([1e8])
                        template_chi = -1e3

                    # Apply the matched filter.
                    corr_array = signal.fftconvolve(
                        data[:last], filt, mode='full')

                    # Smooth the pulse with the average filter.
                    smooth_data = signal.fftconvolve(
                        average_filt, data[:lon], mode='full')[:-21]
                    smooth_rise = smooth_data[:last]

                    # Get the derivation of the smoothed pulse.
                    # But only around the rising edge.
                    gradient = np.gradient(smooth_rise)

                    # Start the saving process.
                    lock.acquire()
                    try:
                        # Get the last filled entry of the arrays.
                        position = queue.get()
                        # Set the pulse number.
                        data2[position] = np.uint64(pulse.name[1:-5])
                        # Set the trigger time.
                        data3[position] = the_pulse.Time
                        # Set the maximum of the response function of the auto-correlation.
                        data4[position] = np.max(corr_array)
                        # Set the template fit amplitude.
                        data5[position] = popt[0]
                        # Set the maximum gradient.
                        data6[position] = gradient.max()
                        # Set the integral of the pulse.
                        data7[position] = smooth_data.sum()
                        # Set the reduced Chi2
                        data8[position] = template_chi
                        # Set the decay time of the pulse.
                        data9[position] = the_pulse.Decay
                    finally:
                        # Mark this entry as filles.
                        queue.put(position + 1)
                        gl.show_progress(
                            position, length, t0, out_str=out_str)
                        lock.release()

                    # Free up memory.
                    del the_pulse
                    del corr_array
                    del popt
                    del popt_chi
                    del gradient
                    del smooth_data
                    del smooth_rise
                    del data

                except (pg.PulseReadError, pg.NonValidPathError):
                    # If there is no pulse file or a corrupted file, skip this.
                    pass


class Filter:
    """ The Filter/template will be calculated by averaging signals.

        Attributes:
            Channel (uint8): The ADC channel.
            Data (numpy.array): The template pulse.
            Path (string): The path to the directory containing ADC channel directories.
            Pixel (uint8): The pixel number.
            Polarity (string): The polarity of the pulse. Either 'NEGP' or 'POSP'.
            Norm (number): The integral of the template pulse.

        Methods:
            plot(): Save a picture of the time trace of template.
    """

    # TODO: Drop verbose argument.
    def __init__(
        self, path, number, polarity=None, default_path=gl.DEFAULT_PATH,
            default_pixel=gl.DEFAULT_PIXEL, rec=1, verbose=False, new=False):
        """
        Args:
            path (string): The path to the directory containing ADC channel directories.
            number (int): Either a pixel number or the ADC channel.
            polarity (string, optional): The pixel polarity. Either 'NEGP' or
                                         'POSP'. Defaults to None.
            default_path (string, optional): Path to a data set containing a
                                             pixel, which should be used, if no
                                             calibration lines are detected.
                                             Defaults to gl.DEFAULT_PATH.
            default_pixel (string, optional): A pixel, which should be used, if
                                              no calibration lines are detected.
                                              Defaults to gl.DEFAULT_PIXEL.
            rec (int, optional): Level of recursions. Defaults to 1.
                                 Should not be changed.
            verbose (bool, optional): Does not do anything. Defaults to False.
            new (bool, optional): If data should be recalculated instead of
                                  being loaded. Defaults to False.
        """

        # Determines the pixel properties.
        __pulse = pg.RandomPulse(path, number, polarity=polarity)
        self.Path = __pulse.Path
        self.Channel = __pulse.Channel
        self.Polarity = __pulse.Polarity
        self.Pixel = __pulse.Pixel
        del __pulse

        self.Data = None

        # TODO: The following list should be used as arguments and defined in
        # global_parameters.
        # List of data sets consiting coincident events
        coin_pathes = ["Run24-Coincidences", "Run25-Coincidences"]
        # List of data sets consiting background pixels.
        noncoin_pathes = ["Run24-Asymmetric", "Run25"]

        # Check if the current pixel is included in data sets of the two lists
        # from above. There are some special cases. It is hard to identify
        # Ho-163 lines in coincidental events. In those cases, use non-coincidence
        # data of the same pixel (if existing).
        for i, coin_path in enumerate(coin_pathes):
            if coin_path in path:
                if i == 0:
                    tmp_path_0 = path.split("only_coincidences_corr")[0]
                    tmp_path_1 = path.split("only_coincidences_corr")[1]
                    tmp_path = tmp_path_0 + "asymmetric_channels" + tmp_path_1
                else:
                    tmp_path = path

                # Generate the path to the other pixel.
                tmp_path_0 = tmp_path.split(coin_path)[0]
                tmp_path_1 = tmp_path.split(coin_path)[1]
                tmp_path = tmp_path_0 + noncoin_pathes[i] + tmp_path_1

                # Check if the same pixel exists in the other path.
                # If yes, load this pixel.
                if os.path.exists(os.path.join(
                        tmp_path, "ADC" + str(self.Channel), self.Polarity)):

                    _ = pg.RandomPulse(tmp_path, number, polarity=polarity)
                    temp_filter = Filter(
                        tmp_path, number, polarity=polarity)
                    self.Data = temp_filter.Data
                    self.Norm = temp_filter.Norm
                    return

                # The pixel does not exist in a different data set. Data
                # selection rules are defined for differnet pixels in different
                # data sets.
                if i == 0:
                    change_dir = False
                    splitname = "191223_10mK_0"
                    fallback = "191223_10mK_3"

                    if splitname in path:
                        if self.Pixel in [21, 22, 23, 24]:
                            change_dir = True
                            if self.Pixel in [21, 22]:
                                fallback = "191223_10mK_1"

                    if not change_dir:
                        splitname = "191223_10mK_2"
                        fallback = "191223_10mK_0"

                        if splitname in path:
                            if self.Pixel in [7, 8, 23, 24, 27, 28]:
                                change_dir = True
                                if self.Pixel in [23, 24]:
                                    fallback = "191223_10mK_3"

                    if not change_dir:
                        splitname = "191223_10mK_3"
                        fallback = "191223_10mK_0"

                        if splitname in path:
                            if self.Pixel in [9, 10, 21, 22, 27, 28]:
                                change_dir = True
                                if self.Pixel in [21, 22]:
                                    fallback = "191223_10mK_1"

                    if change_dir:
                        tmp_path_0 = path.split(splitname)[0]
                        tmp_path_1 = path.split(splitname)[1]
                        tmp_path = tmp_path_0 + fallback + tmp_path_1
                        temp_filter = Filter(
                            tmp_path, self.Channel, polarity=self.Polarity)
                        self.Data = temp_filter.Data
                        self.Norm = temp_filter.Norm
                        return

        # Check which line should be used for the template.
        pathToDirectory = os.path.join(
            self.Path, gl.ADC + str(self.Channel), self.Polarity)
        pathToIndex = os.path.join(pathToDirectory, gl.FILE_CALIBRATION_INDEX)
        if not os.path.exists(pathToIndex):
            # The file does not exist. It will be created.
            _ = cg.CalibrationEvents(
                self.Path, self.Channel, polarity=self.Polarity)#, new=new)
            del _

        # Pulses of the partner pixel will be used too.
        other_pol = gl.NEGP
        if self.Polarity == gl.NEGP:
            other_pol = gl.POSP

        pathToOtherDirectory = os.path.join(
            self.Path, gl.ADC + str(self.Channel), other_pol)
        pathToOtherIndex = os.path.join(
            pathToOtherDirectory, gl.FILE_CALIBRATION_INDEX)
        use_other = False
        if os.path.exists(pathToOtherDirectory):
            if not os.path.exists(pathToOtherIndex):# or new:
                _ = cg.CalibrationEvents(
                    self.Path, self.Channel, polarity=other_pol)#, new=new)
                del _

            # Only use pulses of the partner pixel, if it has pulses in the lines.
            use_other = pd.read_csv(pathToOtherIndex)[gl.COLUMN_CAL_INDEX][0] != 0

        # Save path of the template data.
        pathToTemplate = os.path.join(
            pathToDirectory, gl.FILE_TEMPLATE)

        # Save path of the decay time.
        pathToDecayTime = os.path.join(
            pathToDirectory, gl.FILE_CALIBRATION_DECAY_TIME)

        # Check if data already exists and load it, or if it should be recreated.
        if os.path.exists(pathToTemplate) and not new:
            self.Data = np.load(pathToTemplate)
            self.Norm = self.Data.sum()
            return

        # If there are no detected lines, use either the partner pixel or the
        # default pixel.
        if pd.read_csv(pathToIndex)[gl.COLUMN_CAL_INDEX][0] == 0:
            if not use_other:
                temp_filter = Filter(default_path, int(default_pixel))
            else:
                temp_filter = Filter(
                    self.Path, self.Channel, polarity=other_pol)
            self.Data = np.copy(temp_filter.Data)
            self.Norm = temp_filter.Norm
            del temp_filter
            return

        # Get the list of pulses which should be used.
        cali_events = cg.CalibrationEvents(
            self.Path, self.Channel, polarity=self.Polarity)
        template_pulses = cali_events.Pulses

        # Get the list of pulses of the partner pixel.
        other_cali_events = None
        if os.path.exists(pathToOtherDirectory):
            other_cali_events = cg.CalibrationEvents(
                self.Path, self.Channel, polarity=other_pol)
            other_template_pulses = other_cali_events.Pulses

        # Initialize a list, which will contain the decay times of the pulses.
        # Only pulses with the mean decay time will be used for the template.
        if (rec == 0) or (not use_other):
            decay_array = np.zeros(np.array(template_pulses).shape[0])
        else:
            decay_array = np.zeros(
                np.array(template_pulses).shape[0] +
                np.array(other_template_pulses).shape[0])

        # Fill the list of decay times.
        counter = 0
        # Iterate over pulses of this pixel.
        for pulse in tqdm(np.array(template_pulses)):
            tmp_pulse = pg.Pulse(
                self.Path, pulse=pulse, number=self.Channel,
                polarity=self.Polarity)
            decay_array[counter] = tmp_pulse.Decay
            counter += 1
            del tmp_pulse

        # Iterate over pulses of the partner pixel.
        if (rec != 0) and use_other:
            for pulse in tqdm(np.array(other_template_pulses)):
                tmp_pulse = pg.Pulse(
                    self.Path, pulse=pulse, number=self.Channel,
                    polarity=other_pol)
                decay_array[counter] = tmp_pulse.Decay
                counter += 1
                del tmp_pulse

        # Save the mean decay times.
        np.save(pathToDecayTime, np.array([decay_array.mean()]))
        decay_time = decay_array.mean()

        # Initialize the template.
        filter_array = None
        # Define how much the decay time is allowed to vary.
        lim = 0

        # Increase lim until there are pulses fullfilling the decay time condition.
        while filter_array is None:
            # The number of pulses used for the template.
            counter = 0
            lim += 0.03

            # Break condition. There are no pulses fullfilling the decay time condition.
            # Use the default pixel.
            if lim > 0.2:
                temp_filter = Filter(default_path, int(default_pixel))
                self.Data = np.copy(temp_filter.Data)
                self.Norm = temp_filter.Norm
                del temp_filter
                np.save(pathToTemplate, self.Data)
                return

            # Iterate over all pulses of current pixel.
            for i in template_pulses[gl.COLUMN_SIGNAL_NUMBER]:
                # Load the pulse.
                this_pulse = pg.Pulse(
                    self.Path, pulse=int(i), number=self.Channel,
                    polarity=self.Polarity, decay_time=decay_time)

                # The decay time of this pulse should not differ much
                # from the mean decay time.
                if np.abs(this_pulse.Decay - decay_time) <= (lim * decay_time):
                    counter += 1

                    # Add this pulse to the template.
                    if filter_array is None:
                        filter_array = np.copy(this_pulse.Data)
                    else:
                        filter_array += this_pulse.Data
                del this_pulse

            # Iterate over the pulses of the partner pixel.
            if (use_other and rec != 0):
                for i in other_template_pulses[gl.COLUMN_SIGNAL_NUMBER]:
                    # Load the pulse.
                    this_pulse = pg.Pulse(
                        self.Path, pulse=int(i), number=self.Channel,
                        polarity=other_pol, decay_time=decay_time)

                    # Check if the decay time is nearly the mean decay time.
                    if np.abs(this_pulse.Decay - decay_time) <= \
                            (lim * decay_time):
                        counter += 1

                        # Add this pulse to the template.
                        if filter_array is None:
                            filter_array = np.copy(this_pulse.Data)
                        else:
                            filter_array += this_pulse.Data
                    del this_pulse


        # Calculate the mean pulse.
        filter_array /= counter

        # Transform the template: f(x) -> f(-x).
        # This is done for the "matched filter".
        self.Data = filter_array[::-1]
        # Calculate the area of the template.
        self.Norm = filter_array.sum()
        # Save the template.
        np.save(pathToTemplate, self.Data)
        del template_pulses
        del cali_events

        # Save a picture of the template.
        self.plot()


    def plot(self, show=False):
        """ Plot/Save the time trace of the template.

        Args:
            show (bool, optional): If True plot the picture, else save it.
                                   Defaults to False.
        """

        # Define some figure properties.
        title = "Template for Pixel " + str(self.Pixel)
        xlabel = tex_to_uni("Time in \mus")
        ylabel = "Voltage in mV"
        # Create the figure.
        #             Sampling rate
        #                 |
        # 128 ns = 1 / (125 MHz / 16)
        #                         |
        #                    Oversampling
        curve = ptool.Curve(
            np.arange(len(self.Data)) * 128. / 1000., self.Data[::-1],
            title=title, xlabel=xlabel, ylabel=ylabel)
        if show:
            curve.plot()
        else:
            curve.save(os.path.join(
                self.Path, "Template_" + str(self.Pixel) + ".png"))


    def __del__(self):
        del self.Data


class PixelDay:
    """ Calculate the pulse shape parameters for each pulse of the pixel.

        Attributes:
            Cal_Derivative (numpy.array): Calibration parameters of the derivative.
            Cal_Filter (numpy.array): Calibration parameters of the matched filter.
            Cal_Full_Integral (numpy.array): Calibration parameters for the integral.
            Cal_Template (numpy.array): Calibration parameters for the template fit.
            Channel (uint8): The ADC channel.
            Chi (array like): The reduced Chi2 values of the pulses.
            Data (pandas.DataFrame): Table containing the pulse shape parameters
                                     of each pulse.
            Decay (array like): The decay times of the pulses.
            E_Derivative (array like): The calibrated derivative parameters.
            E_Filter (array like): The calibrated filter parameters.
            E_Integral (array like): The calibrated integral parameters.
            E_Template (array like): The calibrated template fit parameters.
            Length (int): The amount of pulses.
            Path (string): The path to the directory containing ADC channel directories.
            Pixel (uint8): The pixel number.
            Polarity (string): The polarity of the pulse. Either 'NEGP' or 'POSP'.
            Signal_Numbers (array like): The pulse numbers of each pulse.
            Timestamps (array like): The trigger time of each pulse.
    """

    def __init__(self, path, number, polarity=None,
            new=False, rec=0, verbose=False):
        """
        Args:
            path (string): The path to the directory containing ADC channel directories.
            number (int): Either a pixel number or the ADC channel.
            polarity (string, optional): The pixel polarity. Either 'NEGP' or
                                         'POSP'. Defaults to None.
            new (bool, optional): If data should be recalculated instead of
                                  being loaded. Defaults to False.
            rec (int, optional): Level of recursions. Defaults to 0.
                                 Should not be changed.
            verbose (bool, optional): Defines if debug information should be shown.
                                      Defaults to False.
        """

        # Determines the pixel properties.
        __pulse = pg.RandomPulse(path, number, polarity=polarity)
        self.Path = __pulse.Path
        self.Channel = __pulse.Channel
        self.Polarity = __pulse.Polarity
        self.Pixel = __pulse.Pixel
        del __pulse


        # Define the path to the pixel.
        pathToDirectory = os.path.join(
            self.Path, gl.ADC + str(self.Channel), self.Polarity)

        # Define the path to the partner pixel.
        if self.Polarity == gl.NEGP:
            other_pol = gl.POSP
        else:
            other_pol = gl.NEGP
        pathToOtherDirectory = os.path.join(
            self.Path, gl.ADC + str(self.Channel), other_pol)

        # A list of files, which (will) include the pulse shape parameters.
        file_names = {}
        # A list of pathes of the files from above.
        pathes = {}
        # Fill these two lists.
        for name in gl.PIXEL_NAMES:
            file_names[name] = gl.PIXEL_EVENTS + "_" + name + ".npy"
            pathes[name] = os.path.join(pathToDirectory, file_names[name])

        # Path to the files containing the detected lines.
        pathToPixelCalibration = os.path.join(
            pathToDirectory, gl.FILE_PIXEL_CALIBRATION)

        # Path to the file containing the mean decay time of pulses of this pixel.
        pathToDecayTime = os.path.join(
            pathToDirectory, gl.FILE_CALIBRATION_DECAY_TIME)

        # Check if data already exists and load it, or recreate it.
        if os.path.exists(pathes[gl.PIXEL_NAMES[0]]) and not new:
            # Initialize the dictionary consisting the pulse shape parameters.
            temp_data = {}
            # Load all data columns.
            for name in gl.PIXEL_NAMES:
                temp_data[name] = np.load(pathes[name])
            # Convert the dictionary to a pandas.DataFrame.
            self.Data = pd.DataFrame(data=temp_data)
            self.Length = self.Data.shape[0]

            self.Signal_Numbers = self.Data[gl.COLUMN_SIGNAL_NUMBER]
            self.Timestamps = self.Data[gl.COLUMN_TIMESTAMP]
            self.E_Filter = self.Data[gl.COLUMN_FILTER_AMP]
            self.E_Template = self.Data[gl.COLUMN_TEMPLATE_AMP]
            self.E_Derivative = self.Data[gl.COLUMN_DERIVATIVE_AMP]
            self.E_Integral = self.Data[gl.COLUMN_FULL_INTEGRAL]
            self.Chi = self.Data[gl.COLUMN_TEMPLATE_CHI]
            # self.Decay = self.Data[gl.COLUMN_DECAY_TIME]

            # Load the calibration parameters.
            calibration_array = np.load(pathToPixelCalibration)
            self.Cal_Template = calibration_array[0]
            if np.isnan(self.Cal_Template):
                self.Cal_Template = None
            self.Cal_Filter = np.array(
                [calibration_array[1], calibration_array[2]])
            if np.isnan(calibration_array[1]):
                self.Cal_Filter = None
            self.Cal_Derivative = np.array(
                [calibration_array[3], calibration_array[4]])
            if np.isnan(calibration_array[3]):
                self.Cal_Derivative = None
            self.Cal_Full_Integral = np.array(
                [calibration_array[5], calibration_array[6]])
            if np.isnan(calibration_array[5]):
                self.Cal_Full_Integral = None

            # Check if this data set can be calibrated.
            self.__can_be_calibrated = calibration_array[7].astype(bool)
            self.__all_None = calibration_array[8].astype(bool)

            # Load the mean decay time.
            # TODO: Check if this is necessary. It conflicts with the assignment
            # of self.Decay at the end of this method.
            self.Decay = None
            if os.path.exists(pathToDecayTime):
                self.Decay = np.load(pathToDecayTime)[0]
            elif os.path.exists(os.path.join(
                    pathToOtherDirectory, gl.FILE_CALIBRATION_DECAY_TIME)):
                self.Decay = np.load(os.path.join(
                    pathToOtherDirectory, gl.FILE_CALIBRATION_DECAY_TIME))[0]

            del calibration_array
            return

        # Initialize the data frame.
        self.Data = None

        # Load the filter/template of this pixel.
        the_filter = Filter(
            self.Path, self.Channel, polarity=self.Polarity,
            verbose=verbose, new=new)

        filter_array = the_filter.Data
        normalized = the_filter.Norm

        # Load the mean decay time.
        self.Decay = None
        if os.path.exists(pathToDecayTime):
            self.Decay = np.load(pathToDecayTime)[0]
        elif os.path.exists(os.path.join(
                pathToOtherDirectory, gl.FILE_CALIBRATION_DECAY_TIME)):
            self.Decay = np.load(os.path.join(
                pathToOtherDirectory, gl.FILE_CALIBRATION_DECAY_TIME))[0]

        # Not the full pulse trace will be analyzed. This reduce the amount of
        # pile-up events. Define the length of the fraction of the pulse, which
        # should be analyzed.
        delta_e = filter_array.shape[0] - np.argmax(filter_array) - gl.FIRST_E
        LAST = 4 * delta_e

        # In case of the template fit, the reduced Chi2 is more meaningfull, if
        # more data points are used.
        if LAST > 4000:
            LONG = LAST
        else:
            LONG = 4000

        # Normalize and assign the "matched filter".
        convolve_array = filter_array / normalized
        FILTER = convolve_array[-(gl.FIRST_E + LAST):-gl.FIRST_E]
        # Create the template fit. A short one for the energy reconstruction
        # and a longer one for the Chi2.
        TEMPLATE = filter_array[::-1][gl.FIRST_E + 3000:gl.FIRST_E + 4000]
        FULL_TEMPLATE = filter_array[::-1][gl.FIRST_E:gl.FIRST_E + 4000]

        # Create the empty arrays, which will be the columns of the data frame.
        # Determine the length of the final table.
        entries = int(0)
        for pulses in os.scandir(pathToDirectory):
            # Count the pulses.
            if gl.is_pulses_dir(pulses):
                entries += len(os.listdir(pulses.path))

        LENGTH = entries
        # Column of ADC channel numbers. The channel is obviously the same for
        # all pulses of the same pixel.
        temp_data_0 = np.zeros(LENGTH).astype(np.uint8) + self.Channel
        # Column of pixel numbers. Also the same for all pulses.
        temp_data_1 = np.zeros(LENGTH).astype(np.uint8) + self.Pixel
        # Column of pulse numbers.
        temp_data_2 = mp.Array('Q', LENGTH, lock=False)
        # Column of trigger times.
        temp_data_3 = mp.Array('Q', LENGTH, lock=False)
        # Column of amplitudes of the response function of the auto-correlation (matched filter)
        temp_data_4 = mp.Array('d', LENGTH, lock=False)
        # Column of amplitudes of the template fit.
        temp_data_5 = mp.Array('d', LENGTH, lock=False)
        # Column of maximal gradients (rise time).
        temp_data_6 = mp.Array('d', LENGTH, lock=False)
        # Column of the integral of the pulses (area).
        temp_data_7 = mp.Array('d', LENGTH, lock=False)
        # Column of the reduced Chi2 resulting from the template fit.
        temp_data_8 = mp.Array('d', LENGTH, lock=False)
        # Column of decay times.
        temp_data_9 = mp.Array('d', LENGTH, lock=False)


        # Initialize iterating values
        # TODO: The initialization of position should not be needed.
        position = 0
        procs = list()
        queue = mp.Queue()
        queue.put(0)
        lock = mp.Lock()

        # Define the out, which will be printed during computing.
        out_str = "Pixel " + str(self.Pixel)
        out_str += ": Loading " + str(LENGTH) + " pulses"

        # Set the start time.
        t0 = datetime.now()

        # Define the kernel for the average filter.
        average_filt = np.ones(21)

        print(out_str + (80 - len(out_str)) * " ", end="\r")

        # Create a list of all pulse files and their pathes.
        path_list = None
        name_list = None
        for pulses in os.scandir(pathToDirectory):
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
        if num_of_cores > 4:
            num_of_cores = 4

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

        # Iterate over all traces to calculate their pulse shape parameters.
        for i in range(num_of_cores):
            if path_dict[i] is not None:
                # Create a process.
                p = mp.Process(target=multi_func, args=(
                    path_dict[i], name_dict[i], queue, lock, temp_data_2,
                    temp_data_3, temp_data_4, temp_data_5, temp_data_6,
                    temp_data_7, temp_data_8, temp_data_9, self.Decay,
                    TEMPLATE, FULL_TEMPLATE, FILTER, LAST, LONG, LENGTH,
                    out_str, t0, average_filt))

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

        # Get the last entry of the data arrays, which is filled.
        position = queue.get()
        queue.close()


        # Print how long it took to computing.
        out_str = "Pixel " + str(self.Pixel)
        out_str += ": Loading " + str(position) + " pulses"

        time_passed = datetime.now() - t0
        time_passed = time_passed.total_seconds()
        minutes_passed = int(time_passed / 60)
        if minutes_passed > 0:
            minutes_passed = str(minutes_passed) + " min "
        else:
            minutes_passed = ""
        seconds_passed = int(time_passed % 60)
        if seconds_passed > 0:
            seconds_passed = str(seconds_passed) + " s"
        else:
            seconds_passed = ""
        time_passed = minutes_passed
        time_passed += seconds_passed
        if time_passed != "":
            time_passed = " -- Done in " + time_passed
        else:
            time_passed = " -- Done"

        prog = out_str + time_passed
        print(prog + (80 - len(prog)) * " ")#, end="\r")

        # Convert the data arrays to a dictionary.
        temp_data = {}
        data_container = (
            temp_data_0, temp_data_1, temp_data_2, temp_data_3, temp_data_4,
            temp_data_5, temp_data_6, temp_data_7, temp_data_8, temp_data_9)

        # Remove all not filled entries.
        for i in range(len(gl.PIXEL_NAMES)):
            temp_data[gl.PIXEL_NAMES[i]] = np.array(
                data_container[i][:position])

        # Free up memory.
        for td in data_container:
            del td

        del temp_data_0
        del temp_data_1
        del temp_data_2
        del temp_data_3
        del temp_data_4
        del temp_data_5
        del temp_data_6
        del temp_data_7
        del temp_data_8
        del temp_data_9

        del data_container

        # Calibrate the pulse shape parameters. This will calibrated to the
        # energy spectrum.
        # Initialize the calibration parameters.
        self.Cal_Filter = None
        self.Cal_Template = None
        self.Cal_Derivative = None
        self.Cal_Full_Integral = None
        self.Cal_Rise_Integral = None
        self.__can_be_calibrated = True
        self.__all_None = True

        # This will be the path to the calibration lines/index.
        pathToIndex = None

        # TODO: The following list should be used as arguments and defined in
        # global_parameters.
        # List of data sets consiting coincident events
        coin_pathes = ["Run24-Coincidences", "Run25-Coincidences"]
        # List of data sets consiting background pixels.
        noncoin_pathes = ["Run24-Asymmetric", "Run25"]

        # Coincidental events can not be calibrated, since no lines can be detected.
        # Use other data sets instead.
        for i, coin_path in enumerate(coin_pathes):
            # TODO: This is the same bloch as in Filter class.
            if coin_path in path:
                if i == 0:
                    tmp_path_0 = path.split("only_coincidences_corr")[0]
                    tmp_path_1 = path.split("only_coincidences_corr")[1]
                    tmp_path = tmp_path_0 + "asymmetric_channels" + tmp_path_1
                else:
                    tmp_path = path

                # Generate the path to the other pixel.
                tmp_path_0 = tmp_path.split(coin_path)[0]
                tmp_path_1 = tmp_path.split(coin_path)[1]
                tmp_path = tmp_path_0 + noncoin_pathes[i] + tmp_path_1

                # Check if the same pixel exists in the other path.
                # If yes, load the calibration parameters of this data set.
                if os.path.exists(os.path.join(
                    tmp_path, "ADC" + str(self.Channel), self.Polarity)):
                    pathToIndex = 1
                    _ = pg.RandomPulse(tmp_path, number, polarity=polarity)
                    tmp_Event = PixelDay(
                        tmp_path, self.Channel, polarity=self.Polarity,
                        verbose=verbose)
                    self.Cal_Filter = tmp_Event.Cal_Filter
                    self.Cal_Template = tmp_Event.Cal_Template
                    self.Cal_Derivative = tmp_Event.Cal_Derivative
                    self.Cal_Full_Integral = tmp_Event.Cal_Full_Integral
                    self.__can_be_calibrated = tmp_Event.__can_be_calibrated
                    self.__all_None = tmp_Event.__all_None

                    cal_index = 1
                    if not self.__can_be_calibrated:
                        cal_index = 0

        # Get the index defining which line should be used for calibration.
        if pathToIndex is None:
            pathToIndex = os.path.join(
                pathToDirectory, gl.FILE_CALIBRATION_INDEX)
            cal_index = pd.read_csv(pathToIndex)[gl.COLUMN_CAL_INDEX][0]

        # There isw no line.
        if (cal_index == 0):
            # Load the calibration parameters of the partner pixel.
            if (rec == 0):
                other_polarity = gl.NEGP
                if self.Polarity == gl.NEGP:
                    other_polarity = gl.POSP
                if os.path.exists(os.path.join(
                        self.Path, gl.ADC + str(self.Channel), other_pol)):
                    tmp_Event = PixelDay(
                        self.Path, self.Channel, polarity=other_polarity,
                        new=new, rec=1, verbose=verbose)
                    self.Cal_Filter = tmp_Event.Cal_Filter
                    self.Cal_Template = tmp_Event.Cal_Template
                    self.Cal_Derivative = tmp_Event.Cal_Derivative
                    self.Cal_Full_Integral = tmp_Event.Cal_Full_Integral
                    self.__can_be_calibrated = tmp_Event.__can_be_calibrated
                    self.__all_None = tmp_Event.__all_None
                else:
                    # There is no partner pixel.
                    self.__can_be_calibrated = False
                    for name in gl.PIXEL_NAMES:
                        np.save(pathes[name], temp_data[name])
                    self.Data = pd.DataFrame(data=temp_data)
                    np.save(pathToPixelCalibration, np.array(
                        [None, None, None, None, None,
                        None, None, False, True], dtype=np.float32))
                    return
            else:
                # The partner pixel tries to get the information of this pixel,
                # but this pixel can also not be calibrated.
                self.__can_be_calibrated = False
                for name in gl.PIXEL_NAMES:
                    np.save(pathes[name], temp_data[name])
                self.Data = pd.DataFrame(data=temp_data)
                np.save(pathToPixelCalibration, np.array(
                    [None, None, None, None, None,
                     None, None, False, True], dtype=np.float32))
                return

        # Define a second degree polynomial, which will be used to calibrate
        # the pulse shape parameters.
        def pol2(xaxis, a, b):
            """ A second degree polynomial.

            Args:
                xaxis (number or numpy.array): The x-values.
                a (number): a * x
                b (number): b * x**2

            Returns:
                number or numpy.array: The y-values of the polynomial.
            """

            return a * xaxis + b * xaxis**2

        # Calculate the calibration parameters if the data can be calibrated and
        # the parameters are not set yet.
        if self.__can_be_calibrated:
            if self.__all_None:
                # Load the detected lines.
                calibration_lines = pd.read_csv(os.path.join(
                    pathToDirectory, gl.FILE_CALIBRATION_LINES))

                # If the Mn-55 K lines are detected.
                if (calibration_lines[gl.COLUMN_K][0] != 0):
                    # If also the Ho-163 M1 line is detected.
                    if (calibration_lines[gl.COLUMN_M][0] != 0):
                        # Use the Ho-163 EC spectrum for calibration.
                        calibration_energies = np.copy(gl.CALIBRATION_ENERGY_HO)
                        # Define the amplitude of the template.
                        self.Cal_Template = 2053. # eV
                    else:
                        # No Ho-163 lines are detected, thus use the Mn-55 K lines.
                        calibration_energies = np.copy(gl.CALIBRATION_ENERGY_FE)
                        # Define the amplitude of the template.
                        self.Cal_Template = 5890. # eV
                # If the Ho-163 M1 line is detected.
                elif (calibration_lines[gl.COLUMN_M][0] != 0):
                    # Use the Ho-163 EC spectrum for calibration.
                    calibration_energies = np.copy(gl.CALIBRATION_ENERGY_HO)
                    # Define the amplitude of the template.
                    self.Cal_Template = 2053. # eV
                else:
                    # Only the Ho-163 N1 and M2 lines are detected.
                    # This is strange, thus skip the calibration.
                    calibration_energies = None

                # Get all events located in the calibration line.
                in_peak = (temp_data[gl.COLUMN_TEMPLATE_AMP] > 0.98) & \
                        (temp_data[gl.COLUMN_TEMPLATE_AMP] < 1.02)
                # Check if there are any events in the calibration line.
                self.__can_be_calibrated = in_peak[in_peak].shape[0] > 0

                # Can not calibrate the pulse shape parameters.
                # Save them without calibration.
                if (not self.__can_be_calibrated) or \
                        (self.Cal_Template is None):
                    for name in gl.PIXEL_NAMES:
                        np.save(pathes[name], temp_data[name])
                    self.Data = pd.DataFrame(data=temp_data)
                    np.save(pathToPixelCalibration, np.array(
                        [self.Cal_Template, None, None, None, None, None, None,
                         self.__can_be_calibrated, True],
                        dtype=np.float32))
                    return

                # The pulse shape paramters can be calibrated.
                self.__all_None = False
                # Calibrate the template amplitudes.
                temp_data[gl.COLUMN_TEMPLATE_AMP] *= self.Cal_Template

                # Create lists containing the pulse shape parameters of the
                # calibration lines.
                CALIBRATION_LENGTH = len(calibration_energies)
                tmp_filter = np.zeros(CALIBRATION_LENGTH)
                tmp_derivative = np.zeros(CALIBRATION_LENGTH)
                tmp_full_integral = np.zeros(CALIBRATION_LENGTH)

                # TODO: Use enumerate: counter, energy in enumerate()
                counter = 0
                # Iterate over all calibration energies and look for the
                # corresponding pulse shape parameters.
                for energy in calibration_energies:
                    # Check which events correspond to the current energy.
                    positions = \
                            (temp_data[gl.COLUMN_TEMPLATE_AMP] >
                             (energy - 20.)) & \
                            (temp_data[gl.COLUMN_TEMPLATE_AMP] <
                             (energy + 20.))

                    # Only use events with small reduced Chi2 for the calibration.
                    tmp_chi = temp_data[gl.COLUMN_TEMPLATE_CHI][positions]
                    tmp_chi = tmp_chi < 3.

                    # Only use pulses with decay times similar to the mean decay time.
                    tmp_decay = temp_data[gl.COLUMN_DECAY_TIME][positions]
                    tmp_decay -= self.Decay
                    tmp_decay = np.abs(tmp_decay) < 0.03 * self.Decay

                    # Get the median filter parameter at this energy.
                    tmp_filter[counter] = np.median(
                        temp_data[gl.COLUMN_FILTER_AMP][positions][
                            tmp_chi & tmp_decay])

                    # Get the median derivative parameter at this energy.
                    tmp_derivative[counter] = np.median(
                        temp_data[gl.COLUMN_DERIVATIVE_AMP][positions][
                            tmp_chi & tmp_decay])

                    # Get the median the integral parameter at this energy.
                    tmp_full_integral[counter] = np.median(
                        temp_data[gl.COLUMN_FULL_INTEGRAL][positions][
                            tmp_chi & tmp_decay])

                    counter += 1

                # Define the start value for the identification of the
                # calibration parameters.
                p0 = np.array([1., 0.])

                # Calibrate the filter parameters.
                try:
                    # Determine the calibration paramters for the filter parameter.
                    self.Cal_Filter, _ = curve_fit(
                        pol2, tmp_filter, calibration_energies, p0=p0)
                    # Apply the calibration.
                    temp_data[gl.COLUMN_FILTER_AMP] = pol2(
                        temp_data[gl.COLUMN_FILTER_AMP], *self.Cal_Filter)
                except (RuntimeError, ValueError):
                    self.Cal_Filter = None

                # Calibrate the derivative parameters.
                try:
                    # Determine the calibration paramters for the derivative parameter.
                    self.Cal_Derivative, _ = curve_fit(
                        pol2, tmp_derivative, calibration_energies, p0=p0)
                    # Apply the calibration.
                    temp_data[gl.COLUMN_DERIVATIVE_AMP] = pol2(
                        temp_data[gl.COLUMN_DERIVATIVE_AMP], *self.Cal_Derivative)
                except (RuntimeError, ValueError):
                    self.Cal_Derivative = None

                # Calibrate the integral parameters.
                try:
                    # Determine the calibration paramters for the integral parameter.
                    self.Cal_Full_Integral, _ = curve_fit(
                        pol2, tmp_full_integral, calibration_energies, p0=p0)
                    # Apply the calibration.
                    temp_data[gl.COLUMN_FULL_INTEGRAL] = pol2(
                        temp_data[gl.COLUMN_FULL_INTEGRAL],
                        *self.Cal_Full_Integral)
                except (RuntimeError, ValueError):
                    self.Cal_Full_Integral = None
            # The calibration parameters are already determined, but need to be applied.
            else:
                if self.Cal_Template is not None:
                    temp_data[gl.COLUMN_TEMPLATE_AMP] *= self.Cal_Template
                if self.Cal_Filter is not None:
                    temp_data[gl.COLUMN_FILTER_AMP] = pol2(
                        temp_data[gl.COLUMN_FILTER_AMP], *self.Cal_Filter)
                if self.Cal_Derivative is not None:
                    temp_data[gl.COLUMN_DERIVATIVE_AMP] = pol2(
                        temp_data[gl.COLUMN_DERIVATIVE_AMP], *self.Cal_Derivative)
                if self.Cal_Full_Integral is not None:
                    temp_data[gl.COLUMN_FULL_INTEGRAL] = pol2(
                        temp_data[gl.COLUMN_FULL_INTEGRAL],
                        *self.Cal_Full_Integral)


        # Save the calibration parameters in a single file.
        calibration_array = np.array(
            [self.Cal_Template, None, None, None, None, None, None,
             self.__can_be_calibrated, self.__all_None], dtype=np.float32)
        if self.Cal_Filter is not None:
            calibration_array[1] = self.Cal_Filter[0]
            calibration_array[2] = self.Cal_Filter[1]
        if self.Cal_Derivative is not None:
            calibration_array[3] = self.Cal_Derivative[0]
            calibration_array[4] = self.Cal_Derivative[1]
        if self.Cal_Full_Integral is not None:
            calibration_array[5] = self.Cal_Full_Integral[0]
            calibration_array[6] = self.Cal_Full_Integral[1]
        np.save(pathToPixelCalibration, calibration_array)

        # Free up memory.
        del calibration_array
        del lock
        del queue

        # Save the columns (calibrated pulse shape parameters) of the data frame.
        for name in gl.PIXEL_NAMES:
            np.save(pathes[name], temp_data[name])

        # Assign the attributes.
        self.Data = pd.DataFrame(data=temp_data)
        self.Signal_Numbers = self.Data[gl.COLUMN_SIGNAL_NUMBER]
        self.Timestamps = self.Data[gl.COLUMN_TIMESTAMP]
        self.E_Filter = self.Data[gl.COLUMN_FILTER_AMP]
        self.E_Template = self.Data[gl.COLUMN_TEMPLATE_AMP]
        self.E_Derivative = self.Data[gl.COLUMN_DERIVATIVE_AMP]
        self.E_Integral = self.Data[gl.COLUMN_FULL_INTEGRAL]
        self.Chi = self.Data[gl.COLUMN_TEMPLATE_CHI]
        self.Decay = self.Data[gl.COLUMN_DECAY_TIME]


    def __del__(self):
        """ Free up memory """

        # TODO: What happens with self.Chi etc?
        try:
            if self.Data is not None:
                for name in gl.PIXEL_NAMES:
                    del self.Data[name]
                del self.Data
        except AttributeError:
            pass

        del self.Cal_Template
        del self.Cal_Filter
        del self.Cal_Derivative
        del self.Cal_Full_Integral


def generate_pixel_days(path, new=False, verbose=False):
    """ Generate the pulse shape parameters for all pixels of a data set.

    Args:
        path ([type]): Path to the parent directory containing ADC channel directories.
        new (bool, optional): True if the data should be recreated instead of
                              being loaded. Defaults to False.
        verbose (bool, optional): Defines if debug information should be shown.
                                  Defaults to False.
    """

    # TODO: The function starts identical as in calibration_generator.
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

    # Iterate over all pixels and create/load the pulse shape parameters.
    for i in range(len(frame[gl.COLUMN_ADC_CHANNEL])):
        #if frame[gl.COLUMN_PIXEL_NUMBER].iloc[i] not in [
        #        7, 8, 9, 10, 21, 22, 23, 24, 27, 28]:
        #    continue
        print(day + ": Pixel " +
              str(frame[gl.COLUMN_PIXEL_NUMBER].iloc[i]), end="\r")
        _ = PixelDay(
            path, frame[gl.COLUMN_ADC_CHANNEL].iloc[i],
            polarity=frame[gl.COLUMN_POLARITY].iloc[i],
            new=new, verbose=verbose)

        del _
        gc.collect()


def generate_all_pixel_days(path, days, new=False, verbose=False):
    """ Generate the pulse shape parameters for all pixels of different data sets.

    Args:
        path ([type]): Path to the parent directory of data sets.
        days ([type]): List of names of data sets.
        new (bool, optional): True if the data should be recreated instead of
                              being loaded. Defaults to False.
        verbose (bool, optional): Defines if debug information should be shown.
                                  Defaults to False.
    """

    # TODO: Same syntax as in calibration_generator.
    for day in days:
        generate_pixel_days(
            os.path.join(path, day), new=new, verbose=verbose)

    print("Finished all days")
