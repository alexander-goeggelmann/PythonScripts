import os
import sys
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from tqdm import tqdm

import global_parameters as gl

sys.path.append(gl.PATH_TO_PLOTTING)
import PlottingTool as ptool
from TexToUni import tex_to_uni

def linear(xaxis, a, b):
    """A linear function a * x + b.

    Args:
        xaxis (number or numpy.array of numbers): X-values
        a (number): The slope of the curve.
        b (number): Y-shift of the curve.

    Returns:
        number or numpy.array of numbers: Y-values
    """

    return a * xaxis + b


class PulseNotFoundError(Exception):
    def __init__(self, expression):
        self.message = "Passed pulse is not found in " + expression + "."
        super().__init__(self.message)


class PixelNotFoundError(Exception):
    def __init__(self, expression):
        self.message = "Pixel " + expression + " is not found."
        super().__init__(self.message)


class PulseReadError(Exception):
    def __init__(self, expression, message):
        self.message = "Error while loading pule in " + expression + "."
        self.message += " " + message
        super().__init__(self.message)


class NonValidPathError(Exception):
    def __init__(self, expression_0, expression_1, appendix="directory"):
        self.message = expression_0 + " is no valid path. " + expression_1
        self.message += " " + appendix + " is not found."
        super().__init__(self.message)


class NumberNotPassedError(Exception):
    def __init__(self, expression):
        self.message = expression + " has to be passed by passing 'number'"
        super().__init__(self.message)


class RawData:
    """ Load the data of pulse files.

        Attributes:
            Data (numpy.array): Array of int32 consiting the time trace in mV.
            Decay (float): The decay time of the pulse.
            Pixel (uint8): The pixel number.
            Sigma (float): The noise of the baseline.
            Time (uint64): The trigger time in multiples of 4 ns.
            Title (string): A title for plots. Available if init is True.

        Methods:
            plot(): Plot the time trace.
    """

    def __init__(self, path, lentry=3800, init=True):
        """
        Args:
            path (string): Path to pulse file.
            lentry (int, optional): Data[:lentry] is used to estimate noise. Defaults to 3800.
            init (bool, optional): If True load data. Defaults to True.
        """

        self._Path = os.path.abspath(path)
        self._entry = lentry

        self.Data = None
        self.Sigma = None
        self.Time = None
        self.Pixel = None
        self.Title = None
        self.Decay = None

        if init:
            self._get_data()
            self.Title = "Pixel " + str(self.Pixel)


    def _get_data(self):
        """ Load data from self._Path """

        try:
            # Load the pulse.
            _file = open(self._Path, 'rb')
            # Time trace begins after line starting with 'S'. Above is the header.
            data = _file.read().split(b'\nS ')
            header = data[0]
            # Ignore last entry of trace.
            temp = data[1][:-1]

            try:
                # Covert the uint16 binary data into an array of int32.
                # A value of 1 corresponds to 10 mV.
                # Last 4 bytes can not be used (one bit is already removed above).
                self.Data = np.fromstring(
                    temp, dtype=np.uint16).astype(np.int32)[:-35] / 10.
            except ValueError:
                # It can occur, that one bit is missing.
                try:
                    self.Data = np.fromstring(
                        data[1], dtype=np.uint16).astype(np.int32)[:-35] / 10.
                except ValueError:
                    _file.close()
                    raise PulseReadError(
                        self._Path, "File size does not match.")
        except FileNotFoundError:
            raise PulseNotFoundError(self._Path)
        except IndexError:
            _file.close()
            raise PulseReadError(
                self._Path, "Can not separate header and raw data.")

        # Calculate the alignment of the base line.
        X_AXIS = np.arange(self._entry)
        Y_DATA = self.Data[:self._entry]
        # Fit an exponential function to the baseline.
        tmp_popt, _ = curve_fit(
            lambda x, a, b, c: a * np.exp(-X_AXIS / b**2) + c,
            None, Y_DATA, p0=[0., 100., 3200.],
            bounds=([-3000., 10., 2900.], [3000., 1000., 3500.]))

        # Subtract the calculated base line from the data.
        tmp_data = self.Data - tmp_popt[0] * \
                np.exp(-np.arange(len(self.Data)) / tmp_popt[1]**2) - \
                    tmp_popt[2]

        # Get the decay time of the pulse, by fitting an exponential
        # function to the last part of the pulse.
        dec_popt, _ = curve_fit(
            lambda x, a, b: a * np.exp(-np.arange(1000) / b**2),
            None, tmp_data[7000:8000], p0=[0., tmp_popt[1]],
            bounds=([-3000., 10.], [3000., 1000.]))

        # The slope of the baseline should be equal to the decay time
        # of the pulse. Recaclulate the the slope of the baseline using
        # the determined values from above as start and bound parameters.
        popt, _ = curve_fit(
            lambda x, a, b, c: a * np.exp(-X_AXIS / b**2) + c,
            None, Y_DATA, p0=[tmp_popt[0], dec_popt[1], tmp_popt[2]],
            bounds=([-3000., 0.9 * dec_popt[1], 2900.],
                    [3000., 1.1 * dec_popt[1], 3500.]))

        # Subtract the base line from the pulse.
        self.Data = self.Data  - popt[2] - popt[0] * \
                np.exp(-np.arange(len(self.Data)) / popt[1]**2)

        # Determien the decay time of the pulse.
        popt, _ = curve_fit(
            lambda x, a, b: a * np.exp(-np.arange(1000) / b**2),
            None, self.Data[7000:8000], p0=[0., dec_popt[1]],
            bounds=([-3000., 10.], [3000., 1000.]))
        self.Decay = popt[1]

        # Calculate the noise
        self.Sigma = np.nanstd(self.Data[:self._entry])

        # Get the timestamps and the pixel number
        self.Time = header.split(b'Timestamp: ')[1]
        self.Time = np.uint64(self.Time.split(b'\nH Timestamp')[0].split()[0])

        self.Pixel = header.split(b'Pixel no.: ')[1]
        self.Pixel = np.uint8(self.Pixel.split(b'\nH Signal no')[0].split()[0])
        _file.close()


    def plot(self):
        """ Plot the time trace of the signal. """

        xlabel = tex_to_uni("Time in \mus")
        ylabel = "Voltage in mV"
        #             Sampling rate
        #                 |
        # 128 ns = 1 / (125 MHz / 16)
        #                         |
        #                    Oversampling
        curve = ptool.Curve(
            np.arange(len(self.Data)) * 128. / 1000., self.Data,
            title=self.Title, xlabel=xlabel, ylabel=ylabel)
        curve.plot()


    def __del__(self):
        if self.Data is not None:
            # self.Data is an array and needs to be deleted.
            del self.Data

class RawDataCorrected(RawData):
    """ Load the pulse by using pre-known decay time to determine the baseline.

        Attributes:
            Data (numpy.array): Array of int32 consiting the time trace in mV.
            Decay (float): The decay time of the pulse.
            Pixel (uint8): The pixel number.
            Sigma (float): The noise of the baseline.
            Time (uint64): The trigger time in multiples of 4 ns.
            Title (string): A title for plots.

        Methods:
            plot(): Plot the time trace.
    """

    def __init__(self, path, decay_time=None, **kwargs):
        """
           Args:
            path (string): Path to pulse file.
            decay_time (number, optional): The decay time of the pulse. Defaults to None.
        """

        # self.Data = None
        super(RawDataCorrected, self).__init__(path, init=False, **kwargs)
        self.__Decay = decay_time

        self.__get_data()
        self.Title = "Pixel " + str(self.Pixel)


    def __get_data(self):
        """ Load data from self._Path"""

        if self.__Decay is None:
            # If self._Decay is None, there is nothing to correct.
            return super(RawDataCorrected, self)._get_data()
        try:
            # Load the pulse
            _file = open(self._Path, 'rb')
            # Time trace begins after line starting with 'S'. Above is the header.
            data = _file.read().split(b'\nS ')
            header = data[0]
            # Ignore last entry of trace.
            temp = data[1][:-1]

            try:
                # Covert the uint16 binary data into an array of int32.
                # A value of 1 corresponds to 10 mV.
                # Last 4 bytes can not be used (one bit is already removed above).
                self.Data = np.fromstring(
                    temp, dtype=np.uint16).astype(np.int32)[:-35] / 10.
            except ValueError:
                try:
                    # It can occur, that one bit is missing.
                    self.Data = np.fromstring(
                        data[1], dtype=np.uint16).astype(np.int32)[:-35] / 10.
                except ValueError:
                    _file.close()
                    raise PulseReadError(
                        self._Path, "File size does not match.")
        except FileNotFoundError:
            raise PulseNotFoundError(self._Path)
        except IndexError:
            _file.close()
            raise PulseReadError(
                self._Path, "Can not separate header and raw data.")

        # Calculate the alignment of the base line
        X_AXIS = np.arange(self._entry)
        Y_DATA = self.Data[:self._entry]
        # Fit an exponential function to the baseline using the
        # pre-defined decay time.
        popt, _ = curve_fit(
            lambda x, a, c: a * np.exp(-X_AXIS / self.__Decay**2) + c,
            None, Y_DATA, p0=[0., 3200.],
            bounds=([-3000., 2900.], [3000., 3500.]))

        # Subtract the determined base line from the pulse
        self.Data = self.Data - popt[0] * \
                np.exp(-np.arange(len(self.Data)) / self.__Decay**2) - popt[1]

        # Determine the decay time of the pulse
        # It can vary from the pre-defined one, since the signal can be caused
        # by different processes.
        popt, _ = curve_fit(
            lambda x, a, b: a * np.exp(-np.arange(1000) / b**2),
            None, self.Data[7000:8000], p0=[0., self.__Decay],
            bounds=([-3000., 10.], [3000., 1000.]))
        self.Decay = popt[1]

        # Calculate the noise
        self.Sigma = np.nanstd(self.Data[:self._entry])

        # Get the timestamps and the pixel number
        self.Time = header.split(b'Timestamp: ')[1]
        self.Time = np.uint64(self.Time.split(b'\nH Timestamp')[0].split()[0])

        # Get the pixel and signal number
        self.Pixel = header.split(b'Pixel no.: ')[1]
        self.Pixel = np.uint8(self.Pixel.split(b'\nH Signal no')[0].split()[0])
        _file.close()


class PulseFromPath(RawDataCorrected):
    """ Load the pulse located in a given path.

        Attributes:
            Channel (uint8): The ADC channel.
            Data (numpy.array): Array of int32 consiting the time trace in mV.
            Decay (float): The decay time of the pulse.
            Pixel (uint8): The pixel number.
            Polarity (string): The polarity of the pulse. Either 'NEGP' or 'POSP'.
            Pulse (uint64): The pulse number.
            Sigma (float): The noise of the baseline.
            Time (uint64): The trigger time in multiples of 4 ns.
            Title (string): A title for plots.

        Methods:
            plot(): Plot the time trace.
    """
    def __init__(self, path, **kwargs):
        """
        Args:
            path (string): Path to pulse file.

        Raises:
            NonValidPathError: If path does not follow the scheme:
                 ADC5/NEGP/1_1000/S1.sraw
                 ADC16/POSP/4001_5000/S4505.sraw
        """

        # Define name of polarity directories
        NEGP_SEP = gl.PATH_SEP + gl.NEGP + gl.PATH_SEP
        POSP_SEP = gl.PATH_SEP + gl.POSP + gl.PATH_SEP
        # Check if any of the polarity directories are included in path.
        if NEGP_SEP in path:
            self.Polarity = gl.NEGP
        elif POSP_SEP in path:
            self.Polarity = gl.POSP
        else:
            raise NonValidPathError(path, "Polarity")

        # Define name of ADC channel directory and check if it is included in path
        C_SEP = gl.PATH_SEP + gl.ADC
        if C_SEP in path:
            self.Channel = path.split(C_SEP)[1]
            self.Channel = self.Channel.split(gl.PATH_SEP + self.Polarity)[0]
            self.Channel = np.uint8(self.Channel)
        else:
            raise NonValidPathError(path, "ADC-Channel")

        # Define name of pulse directory and check if it is included in path
        S_SEP = "000" + gl.PATH_SEP + gl.PS
        if S_SEP in path:
            self.Pulse = path.split(S_SEP)[1]
            # Check if the file extension is correct.
            # '.sraw' are five characters.
            if gl.SRAW == path[-5:]:
                self.Pulse = self.Pulse[:-5]
                self.Pulse = np.uint64(self.Pulse)
            else:
                raise NonValidPathError(path, gl.SRAW[1:], appendix="file")
        else:
            raise NonValidPathError(path, gl.SRAW[1:], appendix="file")

        super(PulseFromPath, self).__init__(path, **kwargs)

        # Change polarity of pulse if it is negative polarized
        if self.Polarity == gl.NEGP:
            self.Data *= -1.
        self.Title += ": Pulse " + str(self.Pulse)


class Pulse(PulseFromPath):
    """ Load a pulse.

        Attributes:
            Channel (uint8): The ADC channel.
            Data (numpy.array): Array of int32 consiting the time trace in mV.
            Decay (float): The decay time of the pulse.
            Path (string): The path to the directory containing ADC channel directories.
            Pixel (uint8): The pixel number.
            Polarity (string): The polarity of the pulse. Either 'NEGP' or 'POSP'.
            Pulse (uint64): The pulse number.
            Sigma (float): The noise of the baseline.
            Time (uint64): The trigger time in multiples of 4 ns.
            Title (string): A title for plots.

        Methods:
            get_channel(path:str, pixel:int): Get the ADC channel and polarity
                                              of a pulse included in a
                                              sub-directory of path.
            plot(): Plot the time trace.
    """

    def __init__(self, path, pulse=None, number=None, polarity=None, **kwargs):
        """
        Args:
            path (string): Either path to pulse file or to parent directory
                           of ADC channel directories.
            pulse (int, optional): Pulse number. Defaults to None.
            number (int, optional): Either the number of ADC channel or
                                    pixel number. Defaults to None.
            polarity (string, optional): Either 'NEGP' or 'POSP'. Defaults to None.

        Raises:
            PixelNotFoundError: If the pixel number does not exists in path.
            NumberNotPassedError: If pixel number or ADC channel are not passed.
        """

        self.Path = os.path.abspath(path)
        self.__tmp_path = path

        if pulse is not None:
            directory = self.__get_directory(pulse)
            if (polarity is None) and (number is not None):
                _channel, _polarity = self.get_channel(self.Path, number)

                if _channel is None:
                    raise PixelNotFoundError(str(number))

            elif (polarity is None) and (number is None):
                raise NumberNotPassedError("Pixel number")

            elif number is None:
                raise NumberNotPassedError("ADC-Channel")

            else:
                _channel = np.uint8(number)
                _polarity = polarity

            # Build path to pulse file.
            pulse_file = gl.PS + str(int(pulse)) + gl.SRAW
            adc_channel = gl.ADC + str(_channel)
            self.__tmp_path = os.path.join(
                path, adc_channel, _polarity, directory, pulse_file)

        super(Pulse, self).__init__(self.__tmp_path, **kwargs)


    def __get_directory(self, pulse):
        """ Returns the dictionary in which the pulse is located.

        Args:
            pulse (int): Pulse number.

        Returns:
            string: Parent directory of the pulse.
        """

        n = pulse // 1000
        # p = np.uint64(pulse)
        # n = 0
        # while (p >= 1000):
        #   p -= 1000
        #   n += 1
        return str(1000 * n + 1) + "_" + str(1000 * (n + 1))


    def __get_single_channel(self, path, pixel):
        """ Get the ADC number and polarity of a pixel by opening pulse files.

        Args:
            path (string): Parent directory of ADC channel directories.
            pixel (int): Pixel number.

        Returns:
            int, string: The ADC channel and polarity.
        """

        # Walk through all pulses with pulse number = 0, until a pulse of
        # the searched pixel is found.

        # Begin in directory path.
        for channels in os.scandir(path):

            # Loop over all directories in the ADC channel directories.
            if ((gl.ADC not in channels.name) or
                    (not os.listdir(channels.path))):
                # Skip this file/directory if it is a file or not a ADC directory.
                continue

            # The current ADC channel
            number = np.uint8(channels.name.split(gl.ADC)[1])

            # Loop over the polarty directories.
            for pols in os.scandir(channels.path):
                if (gl.NEGP == pols.name) or (gl.POSP == pols.name):
                    # If 'NEGP' or 'POSP' in current directory name.
                    # Note: Files with size 0 are invisible here.

                    found_signal = False
                    # Loop over all parent directories of pulses.
                    for signals in os.scandir(pols.path):

                        # Load a pulse, if not only RawDataRejection and
                        # ScopeSettings are inside this folder.
                        if gl.is_pulses_dir(signals):
                            # Loop over all pulses of current directory.
                            for pulse in os.scandir(signals.path):
                                # Check if the extension is correct.
                                if pulse.name[-5:] == gl.SRAW:
                                    # Open the file
                                    try:
                                        tmp_pulse = RawData(pulse.path)
                                        if pixel == tmp_pulse.Pixel:
                                            # Found pixel.
                                            return number, pols.name

                                        found_signal = True
                                        break

                                    except PulseReadError:
                                        # If file is corrupted, try the next one.
                                        continue

                        # If a pulse of current channel was opened before,
                        # go to next channel.
                        if found_signal:
                            break

        # Pixel is not found.
        return None, None


    def get_channel(self, path, pixel):
        """ Get the ADC number and polarity of a pixel using the list.

        Args:
            path (string): Path to parent directory of ADC channel directories.
            pixel (int): Pixel number.

        Returns:
            uint8, string: The ADC channel and polarity.
        """

        # Column names.
        names = [gl.COLUMN_ADC_CHANNEL,
                 gl.COLUMN_PIXEL_NUMBER,
                 gl.COLUMN_POLARITY]
        # Column dtypes.
        dtypes = [np.uint8, np.uint8, '<U4']

        # Path to the csv file.
        pathToCSV = os.path.join(path, gl.FILE_PIXEL_LIST)

        if os.path.exists(pathToCSV):
            # Load data
            frame = pd.read_csv(pathToCSV)
            # Get row of pixel number.
            return_frame = frame[frame[gl.COLUMN_PIXEL_NUMBER] == pixel]
            if len(return_frame) == 0:
                # Pixel is not included in the table.
                return None, None

            # ADC channel of pixel.
            number = return_frame[gl.COLUMN_ADC_CHANNEL].iloc[0]
            # Polarity of pixel
            polarity = return_frame[gl.COLUMN_POLARITY].iloc[0]
            return np.uint8(number), polarity

        # Table of pixel numbers does not exists, thus it will be created.
        # Table will include the polarity and ADC channel of each pixel.

        # Initialize the dictionary.
        data = {}
        for i in range(len(names)):
            # Maximal 72 pixels are available. Non existing will be deleted at the end.
            data[names[i]] = np.zeros(72, dtype=dtypes[i])

        # Position in the array.
        position = 0

        # Loop over all pixels.
        for p in tqdm(range(1, 72)):
            # Get the adc channel and polarity of the current pixel.
            number, polarity = self.__get_single_channel(path, p)

            # Save these information in the dictionary if pixel exists.
            if not number is None:
                data[gl.COLUMN_ADC_CHANNEL][position] = np.uint8(number)
                data[gl.COLUMN_PIXEL_NUMBER][position]  = np.uint8(p)
                data[gl.COLUMN_POLARITY][position]  = polarity
                position += 1

        # Delete all entries of non exiting pixels.
        for name in names:
            data[name] = data[name][:position]

        # Save the table in a data frame
        frame = pd.DataFrame(data=data)
        frame.to_csv(pathToCSV, index=False)

        # Get the row corresponding to the pixel.
        return_frame = frame[frame[gl.COLUMN_PIXEL_NUMBER] == pixel]

        if len(return_frame) == 0:
            # the pixel does not exits.
            return None, None

        # ADC channel of the pixel.
        number = return_frame[gl.COLUMN_ADC_CHANNEL].iloc[0]
        # Polarity of the pixel.
        polarity = return_frame[gl.COLUMN_POLARITY].iloc[0]

        return np.uint8(number), polarity


class RandomPulse(Pulse):
    """ Load the pulse with the lowest pulse number.

        Attributes:
            Channel (uint8): The ADC channel.
            Data (numpy.array): Array of int32 consiting the time trace in mV.
            Decay (float): The decay time of the pulse.
            Path (string): The path to the directory containing ADC channel directories.
            Pixel (uint8): The pixel number.
            Polarity (string): The polarity of the pulse. Either 'NEGP' or 'POSP'.
            Pulse (uint64): The pulse number.
            Sigma (float): The noise of the baseline.
            Time (uint64): The trigger time in multiples of 4 ns.
            Title (string): A title for plots.

        Methods:
            get_channel(path:str, pixel:int): Get the ADC channel and polarity
                                              of a pulse included in a
                                              sub-directory of path.
            plot(): Plot the time trace.
    """

    def __init__(self, path, number, polarity=None, **kwargs):
        """
        Args:
            path (string): Either path to pulse file or to parent directory
                           of ADC channel directories.
            number (int): Either the number of ADC channel or
                                    pixel number. Defaults to None.
            polarity (string, optional): Either 'NEGP' or 'POSP'. Defaults to None.
        """

        # Look for a pulse of the given pixel.
        in_loop = True
        # Start with pulse number 1.
        pulse = 1
        while in_loop:
            try:
                # If the pulse exists, return it.
                super(RandomPulse, self).__init__(
                    path, pulse, number, polarity=polarity, **kwargs)
                in_loop = False

            except PulseNotFoundError:
                # If pulse does not exists, try the next one.
                pulse += 1

            except PulseReadError:
                # If the file is corrupted, try the next one.
                pulse += 1
