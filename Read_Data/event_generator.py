import os
import numpy as np
import pandas as pd
import sys
from datetime import datetime

import global_parameters as gl
import pulse_generator as pg
import pixel_day_generator as pdg


class Muon_Veto:
    """ Load the timestamps of events received by the veto.

        Attributes:
            Data (numpy.array): An array of timestamps.
            Path (string): The path to the directory containing ADC channel directories.
    """

    def __init__(
        self, path, number, polarity=None, new=False, fitFile=True):
        """
        Args:
            path (string): The path to the directory containing ADC channel directories.
            number (int): Either the number of ADC channel or pixel number.
            polarity (string, optional): Either 'NEGP' or 'POSP'. Defaults to None.
            new (bool, optional): If data should be recalculated instead of
                                  being loaded. Defaults to False.
            fitFile (bool, optional): Defines if the fit file should be used or
                                      if the timestamps should be read from the
                                      raw data. Defaults to True.
        """

        # Define the path to the binary file.
        self.Path = os.path.abspath(path)
        pathToMuonVeto = os.path.join(self.Path, gl.FILE_MUON_VETO)

        # Check if the data already exists.
        if os.path.exists(pathToMuonVeto) and not new:
            self.Data = np.load(pathToMuonVeto)
            return

        # When using the fit file:
        if fitFile:
            # Load the fit file.
            fit_file = FitFile(self.Path, number)
            if fit_file.Data is None:
                self.Data = None
            else:
                # Get the timestamp axis.
                self.Data = np.array(
                    fit_file.Data['Timestamp(ADC)'][
                        fit_file.Data["/* PulsNo"] < 100 * fit_file.Data.index
                    ]).astype(gl.VETO_DTYPE)
                # Drop all empty rows.
                self.Data = self.Data[np.logical_not(np.isnan(self.Data))]
                # Save these data as binaries.
                np.save(pathToMuonVeto, self.Data)

        # When not using the fit file:
        else:
            # Get the pixel properties.
            temp_pulse = pg.RandomPulse(self.Path, number, polarity=polarity)
            temp_channel = temp_pulse.Channel
            temp_polarity = temp_pulse.Polarity
            temp_pixel = temp_pulse.Pixel
            adc_channel = "ADC" + str(temp_channel)

            # Define the path to the pulses.
            pathToDirectory = os.path.join(
                self.Path, adc_channel, temp_polarity)


            # Initialize the timestamp array.
            # TODO: This is done better in PixelDay.
            entries = int(0)
            for pulses in os.scandir(pathToDirectory):
                if gl.is_pulses_dir(pulses):
                    entries += 1
            LENGTH = 1000 * entries
            temp_data = np.zeros(LENGTH, dtype=gl.VETO_DTYPE)

            # Get the start time.
            t0 = datetime.now()

            # Iterate over all traces.
            position = 0
            for pulses in os.scandir(pathToDirectory):
                if gl.is_pulses_dir(pulses):
                    # Loop over all pulses in the folder.
                    for pulse in os.scandir(pulses.path):
                        try:
                            # Load the pulse.
                            the_pulse = pg.RawData(pulse.path)
                            # Assign the timestamp of the pulse.
                            temp_data[position] = the_pulse.Time
                            gl.show_progress(position, LENGTH, t0)
                            position += 1

                        except pg.PulseReadError:
                            pass

            # Drop the empty entries and save the binary file.
            self.Data = temp_data[:position]
            np.save(pathToMuonVeto, self.Data)

    def __del__(self):
        del self.Data


class EventManager:
    """ Generate the event manger, which checks which events are recorded coincidentally.

        Attributes:
            Data (pandas.DataFrame): The table of pixel signals and timings.
            Length (int): The number of events.
            Path (string): The path to the directory containing ADC channel directories.
    """

    def __init__(self, path, veto=False, window_length=1000, delta_muon=0,
        fitFile=True, verbose=False, veto_pixel=32, new=False):
        """
        Args:
            path (string): The path to the directory containing ADC channel directories.
            veto (bool, optional): If there is a veto pixel. Defaults to False.
            window_length (int, optional): The coincidence window. Defaults to 1000.
            delta_muon (int, optional): The time difference to the veto pixel.
                                        Defaults to 0.
            fitFile (bool, optional): Defines if the fit file should be used or
                                      if the timestamps should be read from the
                                      raw data. Defaults to True.
            verbose (bool, optional): [description]. Defaults to False.
            veto_pixel (int, optional): The pixel used as veto. Defaults to 32.
            new (bool, optional): If data should be recalculated instead of
                                  being loaded. Defaults to False.
        """

        # Assign the attribute.
        self.Path = os.path.abspath(path)

        # Load the list of pixels.
        pathToCSV = os.path.join(self.Path, gl.FILE_PIXEL_LIST)
        if not os.path.exists(pathToCSV):
            try:
                _ = pg.RandomPulse(path, 1)
                del _
            except pg.PixelNotFoundError:
                pass

        PIXEL_FRAME = pd.read_csv(pathToCSV)
        NUM_OF_PIXELS = PIXEL_FRAME.shape[0]

        # Set the names and savepaths of the columns of the data frame.
        file_names = {}
        pathes = {}
        dtypes = {}

        self.__names = list()
        # TODO: Use enumerate.
        for i in range(len(gl.EVENT_NAMES)):
            name = gl.EVENT_NAMES[i]
            file_names[name] = gl.EVENT_MANAGER + "_" + name + ".npy"
            pathes[name] = os.path.join(self.Path, file_names[name])
            dtypes[name] = gl.EVENT_DTYPES[i]
            self.__names.append(name)

        for pixel in PIXEL_FRAME[gl.COLUMN_PIXEL_NUMBER]:
            tmp_name = gl.COLUMN_PIXEL_X + str(pixel)
            file_names[tmp_name ] = gl.EVENT_MANAGER + "_" + tmp_name  + ".npy"
            pathes[tmp_name ] = os.path.join(self.Path, file_names[tmp_name ])
            dtypes[tmp_name] = gl.EVENT_PIXEL_DTYPE
            self.__names.append(tmp_name)

        # Check if data already exists.
        if os.path.exists(pathes[self.__names[0]]) and not new:
            temp_data = {}
            for name in self.__names:
                # if (not veto) and ((name == gl.COLUMN_MUON_FLAG) or
                #        (name == gl.COLUMN_DELTA_TIME_MUON)):
                #    continue
                temp_data[name] = np.load(pathes[name])

            self.Data = pd.DataFrame(data=temp_data)
            self.Length = self.Data.shape[0]
            return

        pd_collec = {}
        t_frame = None
        number_of_rows = 0
        # Load data of each pixel.
        for i in range(NUM_OF_PIXELS):
            # Get the pixel properties.
            channel_no = PIXEL_FRAME.iat[i, 0]
            pixel_no = PIXEL_FRAME.iat[i, 1]
            polarity = PIXEL_FRAME.iat[i, 2]

            # Load the pixel data.
            _ = pdg.PixelDay(self.Path, channel_no, polarity=polarity)
            # Store the data in the dictionary.
            pd_collec[pixel_no] = _.Data

            # Drop the unused columns. Only the timestamp, signal number and
            # pixel number will be used.
            pd_collec[pixel_no] = pd_collec[pixel_no].drop(columns=[
                gl.COLUMN_ADC_CHANNEL, gl.COLUMN_FILTER_AMP,
                gl.COLUMN_TEMPLATE_AMP, gl.COLUMN_DERIVATIVE_AMP,
                gl.COLUMN_FULL_INTEGRAL, gl.COLUMN_TEMPLATE_CHI,
                gl.COLUMN_DECAY_TIME])
            number_of_rows += pd_collec[pixel_no].shape[0]

            # Merge all data frames.
            if t_frame is None:
                t_frame = pd_collec[pixel_no].copy()
            else:
                t_frame = t_frame.append(
                    pd_collec[pixel_no], ignore_index=True)

        # Sort the pulses by time.
        t_frame = t_frame.sort_values(by=[gl.COLUMN_TIMESTAMP])

        # Load the timestamps of veto events.
        if veto:
            mVeto = Muon_Veto(path, veto_pixel, fitFile=fitFile)
            veto_data = mVeto.Data  # [mVeto.Data["/* PulsNo"] < \
                    # 100 * mVeto.Data.index]["Timestamp(ADC)"]
            veto_data += delta_muon

        # Initialize the pixel columns.
        temp_data = {}
        for pixel in PIXEL_FRAME[gl.COLUMN_PIXEL_NUMBER]:
            temp_data[pixel] = np.zeros(number_of_rows).astype(np.uint64)

        # Initialize the timestamp column.
        temp_data_0 = np.zeros(number_of_rows).astype(np.int64)
        # Initialize the event duration column.
        temp_data_1 = np.zeros(number_of_rows).astype(np.int64)
        # Initialize the multiplicity column.
        temp_data_2 = np.zeros(number_of_rows).astype(np.uint8)
        # Initialize the self coincidence column.
        temp_data_3 = np.zeros(number_of_rows).astype(np.uint8)
        # Initialize the veto coincidence column.
        temp_data_4 = np.zeros(number_of_rows).astype(np.uint8)
        # Initialize the time difference to the veto event column.
        temp_data_5 = np.zeros(number_of_rows).astype(np.int64)

        # Initialize the iterator.
        itr = int(0)
        NUM_OF_ITERATIONS = t_frame.shape[0] - 1
        data_pos = int(0)

        # Define the veto coincidence window.
        muon_window = 4 * window_length

        # Get the start time.
        t0 = datetime.now()

        # Iterate over all events.
        while itr < NUM_OF_ITERATIONS:
            gl.show_progress(itr, NUM_OF_ITERATIONS, t0)

            # Set the value in column pixel number to the signal number of the pixel.
            temp_data[t_frame.iat[itr, 0]][data_pos] = t_frame.iat[itr, 1]

            # Save the initial timestamp.
            temp_data_0[data_pos] = t_frame.iat[itr, 2]

            # Calculate the time difference to the next event
            delta_t = t_frame.iat[itr + 1, 2] - t_frame.iat[itr, 2]

            # If the events are coincidental go to the next event.
            while (delta_t <= window_length) and \
                    (itr < NUM_OF_ITERATIONS):
                # Add the time difference to the time window/event duration
                temp_data_1[data_pos] += delta_t

                # Increase the number of coincidences/multiplicity.
                temp_data_2[data_pos] += 1

                itr += 1
                if itr < NUM_OF_ITERATIONS:
                    gl.show_progress(itr, NUM_OF_ITERATIONS, t0)
                    # Set the value in column pixel number to signal number.
                    temp_data[t_frame.iat[itr, 0]][data_pos] = \
                            t_frame.iat[itr, 1]

                    # Calculate the time difference to the next event.
                    delta_t = t_frame.iat[itr + 1, 2] - t_frame.iat[itr, 2]

            # Check if the any pixel triggered two times.
            if temp_data_2[data_pos] > 0:
                num_of_coins = 0

                # Determine the number of coincidental pixels.
                for pixel in PIXEL_FRAME[gl.COLUMN_PIXEL_NUMBER]:
                    if temp_data[pixel][data_pos] != 0:
                        num_of_coins += 1

                if temp_data_2[data_pos] >= num_of_coins:
                    # Set the self coincidence flag.
                    temp_data_3[data_pos] = 1

            # Check if there is a coincidence to the muon veto.
            if veto:
                veto_bool = veto_data[
                    (veto_data < (t_frame.iat[itr, 2] + muon_window)) & \
                        (veto_data > (temp_data_0[data_pos] - muon_window))]

                if len(veto_bool) > 0:
                    # Set the muon flag.
                    temp_data_4[data_pos] = 1
                    # Set the veto time difference.
                    temp_data_5[data_pos] = \
                            temp_data_0[data_pos] - veto_bool[0]

            # Go to the next event.
            itr += 1
            data_pos += 1

        # Add the columns to a single dictionary.
        out_data = {}
        data_container = (
            temp_data_0, temp_data_1, temp_data_2,
            temp_data_3, temp_data_4, temp_data_5)

        # Drop all empty rows and save the data.
        # TODO: Use enumerate.
        for i in range(len(gl.EVENT_NAMES)):
            name = gl.EVENT_NAMES[i]
            out_data[name] = data_container[i][:data_pos]
            np.save(pathes[name], out_data[name])

        for pixel in PIXEL_FRAME[gl.COLUMN_PIXEL_NUMBER]:
            name = gl.COLUMN_PIXEL_X + str(pixel)
            out_data[name] = temp_data[pixel][:data_pos]
            np.save(pathes[name], out_data[name])

        # Free up memory.
        del temp_data_0
        del temp_data_1
        del temp_data_2
        del temp_data_3
        del temp_data_4
        del temp_data_5

        # Convert the dictionary to a data frame.
        self.Data = pd.DataFrame(data=out_data)
        self.Length = self.Data.shape[0]

        # Free up memory.
        for pixel in PIXEL_FRAME[gl.COLUMN_PIXEL_NUMBER]:
            del pd_collec[pixel]
        del pd_collec

        print(80 * " ", end="\r")
        sys.stdout.flush()


    def __del__(self):
        for name in self.__names:
            del self.Data[name]
        del self.Data
        del self.__names


def decode_coincidences(value):
    """ Get the pixel numbers encoded in value.

    Args:
        value (int): A coded number.

    Returns:
        list: A list of all pixels in the coded number.
    """

    # Initialize the output.
    out = []
    # Cast the input.
    val = np.uint64(value)

    # Iterate over all possible pixel numbers and check if it is included in value.
    pixels = np.arange(64)
    pixels = pixels[::-1].astype(np.uint)
    for pixel in pixels:
        tmp = np.uint64(np.power(2, pixel))
        # If the pixel is included in value, reduce value.
        if val >= tmp:
            val -= tmp
            # Add the pixel to the list.
            out.append(pixel + 1)
    return out


class FitFile:
    """ Load the data of a fit file of a pixel.

        Attributes:
            Data (pandas.DataFrame): The data stored in the fit file.
            Path (string): The path to the directory containing ADC channel directories.
    """

    def __init__(self, path, pixel):
        """
        Args:
            path (string): The path to the directory containing ADC channel directories.
            pixel (int): The pixel number.
        """

        # Assign the attribute.
        self.Path = os.path.abspath(path)

        # Get the file name containing the converted data.
        file_name = "Pixel" + str(pixel) + ".csv"

        if os.path.exists(os.path.join(self.Path, file_name)):
            # The fit file is already loaded once and converted to csv file.
            this_path = os.path.join(self.Path, file_name)
            self.Data = pd.read_csv(this_path)
        else:
            # Get the file name of the fit file.
            file_name = "Pixel" + str(pixel) + ".fit"
            if os.path.exists(os.path.join(self.Path, "Fitresults", file_name)):
                this_path = os.path.join(self.Path, "Fitresults", file_name)
                # Load the data.
                self.Data = pd.read_csv(this_path, sep='\t', header=24)
            else:
                print("Veto data can not be built. Fitfile of pixel " +
                    str(pixel) + " does not exits.")
                self.Data = None


    def __del__(self):
        del self.Data

# TODO: Combine PixelCoincidentals and PixelNoncoincidentals.
class PixelCoincidentals:
    """ Get the coincidental events of the pixel.

        Attributes:
            Channel (uint8): The ADC channel.
            Data (pandas.DataFrame): The pulse shape parameters of coincidences.
            Length (int): The number of events.
            Path (string): The path to the directory containing ADC channel directories.
            Pixel (uint8): The pixel number.
            Polarity (string): The polarity of the pulse. Either 'NEGP' or 'POSP'.
    """

    def __init__(self, path, number, polarity=None, muon='Ignore', new=False,
            verbose=False):
        """
        Args:
            path (string): The path to the directory containing ADC channel directories.
            number (int): Either a pixel number or the ADC channel.
            polarity (string, optional): The pixel polarity. Either 'NEGP' or
                                         'POSP'. Defaults to None.
            muon (str, optional): One of 'Ignore', 'Coincidental' and
                                  'Noncoincidental'. Defaults to 'Ignore'.
            new (bool, optional): If data should be recalculated instead of
                                  being loaded. Defaults to False.
            verbose (bool, optional): [description]. Defaults to False.
        """

        # Check if the input of muon is correct.
        if type(muon) != str:
            print("'Muon' argument has to be one of 'Ignore', 'Coincidental', \
                    'Noncoincidental'. It is set to 'Ignore'")
            muon = 'Ignore'
        elif muon not in ['Ignore', 'Coincidental', 'Noncoincidental']:
            print("'Muon' argument has to be one of 'Ignore', 'Coincidental', \
                    'Noncoincidental'. It is set to 'Ignore'")
            muon = 'Ignore'

        # Get the pixel properties.
        try:
            __pulse = pg.RandomPulse(path, number, polarity=polarity)
        except pg.PixelNotFoundError:
            self.Path = None
            self.Channel = None
            self.Polarity = None
            self.Pixel = None
            self.Data = None
            return

        self.Path = __pulse.Path
        self.Channel = __pulse.Channel
        self.Polarity = __pulse.Polarity
        self.Pixel = __pulse.Pixel

        del __pulse

        # Get the list of all pixels.
        pathToCSV = os.path.join(self.Path, gl.FILE_PIXEL_LIST)
        if not os.path.exists(pathToCSV):
            try:
                _ = pg.RandomPulse(path, 1)
                del _
            except pg.PixelNotFoundError:
                pass

        PIXEL_FRAME = pd.read_csv(pathToCSV)

        pathToDirectory = os.path.join(
            self.Path, gl.ADC + str(self.Channel), self.Polarity)

        # Load the names of files containing the frame columns.
        file_names = {}
        pathes = {}
        self.__names = \
                gl.PIXEL_NAMES + gl.EVENT_NAMES + [gl.COLUMN_COINCIDENCE_CODE]
        for name in self.__names:
            file_names[name] = gl.PIXEL_COINCIDENCES + "_" + \
                    gl.PIXEL_EVENTS + "_" + name + "_Muon" + muon + ".npy"
            pathes[name] = os.path.join(pathToDirectory, file_names[name])

        # Check if data already exists or if it should be loaded.
        if os.path.exists(pathes[self.__names[0]]) and not new:
            temp_data = {}
            # Load the data.
            # TODO: Check if del is necessary.
            for name in self.__names:
                temp_data[name] = np.load(pathes[name])
            self.Data = pd.DataFrame(data=temp_data)
            self.Length = self.Data.shape[0]
            return

        # Load the event manager.
        _ev = EventManager(self.Path)
        event_manager = _ev.Data

        # Get the rows of the event manger, which include the pixel.
        bool_pixel = np.ones(event_manager.shape[0], dtype=np.bool)
        bool_pixel &= event_manager[gl.COLUMN_PIXEL_X + str(self.Pixel)] > 0

        # Apply the coincidence condition.
        bool_coincidences = np.copy(bool_pixel)
        bool_coincidences &= event_manager[gl.COLUMN_NUM_OF_COINS] > 0

        # Apply the veto condition.
        bool_muon = np.copy(bool_pixel)
        if muon != 'Ignore':
            if muon == 'Coincidental':
                bool_muon &= event_manager[gl.COLUMN_MUON_FLAG] > 0
            else:
                bool_muon &= event_manager[gl.COLUMN_MUON_FLAG] == 0

        bool_pixel &= bool_muon
        bool_pixel &= bool_coincidences

        # Determine the number of coincidences.
        self.Length = bool_pixel[bool_pixel].shape[0]
        if self.Length == 0:
            print("No coincidences are found.")
            self.Data = None
            return

        # Get the column index of the event manger, which corresponds to
        # the pixel number.
        list_len = len(gl.EVENT_NAMES)
        pixel_pos = list_len
        for pixel in PIXEL_FRAME[gl.COLUMN_PIXEL_NUMBER]:
            if pixel == self.Pixel:
                break
            pixel_pos += int(1)

        # Initialize the channel column.
        temp_data_0 = np.zeros(self.Length).astype(np.uint8) + self.Channel
        # Initialize the pixel number column.
        temp_data_1 = np.zeros(self.Length).astype(np.uint8) + self.Pixel
        # Initialize the signal number column.
        temp_data_2 = np.zeros(self.Length).astype(np.uint64)
        # Initialize the timestamp column.
        temp_data_3 = np.zeros(self.Length).astype(np.uint64)
        # Initialize the filter response function amplitude column.
        temp_data_4 = np.zeros(self.Length).astype(np.float64)
        # Initialize the template fit amplitude column.
        temp_data_5 = np.zeros(self.Length).astype(np.float64)
        # Initialize the maximal gradient column.
        temp_data_6 = np.zeros(self.Length).astype(np.float64)
        # Initialize the pulse integral column.
        temp_data_7 = np.zeros(self.Length).astype(np.float64)
        # Initialize the reduced chi2 value column.
        temp_data_8 = np.zeros(self.Length).astype(np.float64)
        # Initialize the decay time column.
        temp_data_9 = np.zeros(self.Length).astype(np.float64)

        # Initialize the first timestamp column.
        temp_data_10 = np.zeros(self.Length).astype(np.int64)
        # Initialize the event duration column.
        temp_data_11 = np.zeros(self.Length).astype(np.int64)
        # Initialize the multiplicity column.
        temp_data_12 = np.zeros(self.Length).astype(np.uint8)
        # Initialize the self coincidence column.
        temp_data_13 = np.zeros(self.Length).astype(np.uint8)
        # Initialize the veto coincidence column.
        temp_data_14 = np.zeros(self.Length).astype(np.uint8)
        # Initialize the time difference to the veto event column.
        temp_data_15 = np.zeros(self.Length).astype(np.int64)

        # Initialize the coincidence code column.
        temp_data_16 = np.zeros(self.Length).astype(np.uint64)

        # Load the pixel data.
        _pday = pdg.PixelDay(
            self.Path, self.Channel, polarity=self.Polarity)
        loaded_data = _pday.Data

        # Select the rows of event manger, which correspond to the pixel.
        event_manager = event_manager[bool_pixel]

        # Get the start time.
        t0 = datetime.now()

        # Iterate over all coincidental events.
        for i in range(self.Length):
            gl.show_progress(i, self.Length, t0)

            # Assign the pulse number.
            temp_data_2[i] = \
                    event_manager.iat[i, pixel_pos]

            # Get the pulse shape paramters of the pulse.
            temp_row = loaded_data[
                loaded_data[gl.COLUMN_SIGNAL_NUMBER] == temp_data_2[i]]

            # Assign the timestamp.
            temp_data_3[i] = temp_row.iat[0, 3]
            # Assign the amplitude of the filter response function.
            temp_data_4[i] = temp_row.iat[0, 4]
            # Assign the amplitude of the template fit.
            temp_data_5[i] = temp_row.iat[0, 5]
            # Assign the maximal gradient.
            temp_data_6[i] = temp_row.iat[0, 6]
            # Assign the integral.
            temp_data_7[i] = temp_row.iat[0, 7]
            # Assign the reduced chi2.
            temp_data_8[i] = temp_row.iat[0, 8]
            # Assign the decay time.
            temp_data_9[i] = temp_row.iat[0, 9]


            # Assign the first timestamp.
            temp_data_10[i] = event_manager.iat[i, 0]
            # Assign the event duration.
            temp_data_11[i] = event_manager.iat[i, 1]
            # Assign the multiplicity.
            temp_data_12[i] = event_manager.iat[i, 2]
            # Assign the self coincidence flag.
            temp_data_13[i] = event_manager.iat[i, 3]
            # Assign the veto coincidence flag.
            temp_data_14[i] = event_manager.iat[i, 4]
            # Assign the time difference to the veto event.
            temp_data_15[i] = event_manager.iat[i, 5]

            # Assign the coincidence code.
            __p_pos = int(0)
            for pixel in PIXEL_FRAME[gl.COLUMN_PIXEL_NUMBER]:
                if event_manager.iat[i, list_len + __p_pos] > 0:
                    if (pixel != self.Pixel):
                        temp_data_16[i] += np.uint64(np.power(2, pixel - 1))
                __p_pos += int(1)

        # Convert the arrays to a pandas data frame.
        temp_data = {}
        data_container = (
            temp_data_0, temp_data_1, temp_data_2, temp_data_3, temp_data_4,
            temp_data_5, temp_data_6, temp_data_7, temp_data_8, temp_data_9,
            temp_data_10, temp_data_11, temp_data_12, temp_data_13,
            temp_data_14, temp_data_15, temp_data_16)

        for i in range(len(self.__names)):
            #print(data_container[i])
            name = self.__names[i]
            temp_data[name] = data_container[i]
            np.save(pathes[name], temp_data[name])

        # Free up memory.
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
        del temp_data_10
        del temp_data_11
        del temp_data_12
        del temp_data_13
        del temp_data_14
        del temp_data_15
        del temp_data_16

        del _ev
        del _pday
        print(80 * " ", end="\r")
        sys.stdout.flush()

        # Assign the attribute.
        self.Data = pd.DataFrame(data=temp_data)


    def __del__(self):
        if self.Data is not None:
            for name in self.__names:
                del self.Data[name]
            del self.Data
        del self.__names


class PixelNoncoincidentals:
    """ Get the non-coincidental events of the pixel.

        Attributes:
            Channel (uint8): The ADC channel.
            Data (pandas.DataFrame): The pulse shape parameters of coincidences.
            Length (int): The number of events.
            Path (string): The path to the directory containing ADC channel directories.
            Pixel (uint8): The pixel number.
            Polarity (string): The polarity of the pulse. Either 'NEGP' or 'POSP'.
    """

    def __init__(self, path, number, polarity=None, muon='Ignore', new=False,
            verbose=False):
        """
        Args:
            path (string): The path to the directory containing ADC channel directories.
            number (int): Either a pixel number or the ADC channel.
            polarity (string, optional): The pixel polarity. Either 'NEGP' or
                                         'POSP'. Defaults to None.
            muon (str, optional): One of 'Ignore', 'Coincidental' and
                                  'Noncoincidental'. Defaults to 'Ignore'.
            new (bool, optional): If data should be recalculated instead of
                                  being loaded. Defaults to False.
            verbose (bool, optional): [description]. Defaults to False.
        """

        # Check if the input of muon is correct.
        if type(muon) != str:
            print("'Muon' argument has to be one of 'Ignore', 'Coincidental', \
                    'Noncoincidental'. It is set to 'Ignore'")
            muon = 'Ignore'
        elif muon not in ['Ignore', 'Coincidental', 'Noncoincidental']:
            print("'Muon' argument has to be one of 'Ignore', 'Coincidental', \
                    'Noncoincidental'. It is set to 'Ignore'")
            muon = 'Ignore'

        # Get the pixel properties.
        try:
            __pulse = pg.RandomPulse(path, number, polarity=polarity)
        except pg.PixelNotFoundError:
            self.Path = None
            self.Channel = None
            self.Polarity = None
            self.Pixel = None
            self.Data = None
            return

        self.Path = __pulse.Path
        self.Channel = __pulse.Channel
        self.Polarity = __pulse.Polarity
        self.Pixel = __pulse.Pixel

        del __pulse

        # Get the list of all pixels.
        pathToCSV = os.path.join(self.Path, gl.FILE_PIXEL_LIST)
        if not os.path.exists(pathToCSV):
            try:
                _ = pg.RandomPulse(path, 1)
                del _
            except pg.PixelNotFoundError:
                pass

        PIXEL_FRAME = pd.read_csv(pathToCSV)

        pathToDirectory = os.path.join(
            self.Path, gl.ADC + str(self.Channel), self.Polarity)

        # Load the names of files containing the frame columns.
        file_names = {}
        pathes = {}
        self.__names = \
                gl.PIXEL_NAMES + gl.EVENT_NAMES[-2:]
        for name in self.__names:
            file_names[name] = gl.PIXEL_NONCOINCIDENCES + "_" + \
                    gl.PIXEL_EVENTS + "_" + name + "_Muon" + muon + ".npy"
            pathes[name] = os.path.join(pathToDirectory, file_names[name])

        # Check if data already exists or if it should be loaded.
        if os.path.exists(pathes[self.__names[0]]) and not new:
            temp_data = {}
            # Load the data.
            # TODO: Check if del is necessary.
            for name in self.__names:
                temp_data[name] = np.load(pathes[name])
            self.Data = pd.DataFrame(data=temp_data)
            self.Length = self.Data.shape[0]
            return

        # Load the event manager.
        _ev = EventManager(self.Path)
        event_manager = _ev.Data

        # Get the rows of the event manger, which include the pixel.
        bool_pixel = np.ones(event_manager.shape[0], dtype=np.bool)
        bool_pixel &= event_manager[gl.COLUMN_PIXEL_X + str(self.Pixel)] > 0

        # Apply the non-coincidence condition.
        bool_coincidences = event_manager[gl.COLUMN_NUM_OF_COINS] == 0

        # Apply the veto condition.
        bool_muon = np.copy(bool_pixel)
        if muon != 'Ignore':
            if muon == 'Coincidental':
                bool_muon &= event_manager[gl.COLUMN_MUON_FLAG] > 0
            else:
                bool_muon &= event_manager[gl.COLUMN_MUON_FLAG] == 0

        bool_pixel &= bool_muon
        bool_pixel &= bool_coincidences

        # Determine the number of coincidences.
        self.Length = bool_pixel[bool_pixel].shape[0]

        # Get the column index of the event manger, which corresponds to
        # the pixel number.
        pixel_pos = len(gl.EVENT_NAMES)
        for pixel in PIXEL_FRAME[gl.COLUMN_PIXEL_NUMBER]:
            if pixel == self.Pixel:
                break
            pixel_pos += int(1)

        # Initialize the channel column.
        temp_data_0 = np.zeros(self.Length).astype(np.uint8) + self.Channel
        # Initialize the pixel number column.
        temp_data_1 = np.zeros(self.Length).astype(np.uint8) + self.Pixel
        # Initialize the signal number column.
        temp_data_2 = np.zeros(self.Length).astype(np.uint64)
        # Initialize the timestamp column.
        temp_data_3 = np.zeros(self.Length).astype(np.uint64)
        # Initialize the filter response function amplitude column.
        temp_data_4 = np.zeros(self.Length).astype(np.float64)
        # Initialize the template fit amplitude column.
        temp_data_5 = np.zeros(self.Length).astype(np.float64)
        # Initialize the maximal gradient column.
        temp_data_6 = np.zeros(self.Length).astype(np.float64)
        # Initialize the pulse integral column.
        temp_data_7 = np.zeros(self.Length).astype(np.float64)
        # Initialize the reduced chi2 value column.
        temp_data_8 = np.zeros(self.Length).astype(np.float64)
        # Initialize the decay time column.
        temp_data_9 = np.zeros(self.Length).astype(np.float64)

        # Initialize the veto coincidence column.
        temp_data_10 = np.zeros(self.Length).astype(np.uint8)
        # Initialize the time difference to the veto event column.
        temp_data_11 = np.zeros(self.Length).astype(np.int64)

        # Load the pixel data.
        _pday = pdg.PixelDay(
            self.Path, self.Channel, polarity=self.Polarity)
        loaded_data = _pday.Data

        # Select the rows of event manger, which correspond to the pixel.
        event_manager = event_manager[bool_pixel]

        # Get the start time.
        t0 = datetime.now()

        # Iterate over all non-coincidental events.
        for i in range(self.Length):
            gl.show_progress(i, self.Length, t0)

            # Assign the pulse number.
            temp_data_2[i] = \
                    event_manager.iat[i, pixel_pos]

            # Get the pulse shape paramters of the pulse.
            temp_row = loaded_data[
                loaded_data[gl.COLUMN_SIGNAL_NUMBER] == temp_data_2[i]]

            # Assign the timestamp.
            temp_data_3[i] = temp_row.iat[0, 3]
            # Assign the amplitude of the filter response function.
            temp_data_4[i] = temp_row.iat[0, 4]
            # Assign the amplitude of the template fit.
            temp_data_5[i] = temp_row.iat[0, 5]
            # Assign the maximal gradient.
            temp_data_6[i] = temp_row.iat[0, 6]
            # Assign the integral.
            temp_data_7[i] = temp_row.iat[0, 7]
            # Assign the reduced chi2.
            temp_data_8[i] = temp_row.iat[0, 8]
            # Assign the decay time.
            temp_data_9[i] = temp_row.iat[0, 9]

            # Assign the veto coincidence flag.
            temp_data_10[i] = event_manager.iat[i, 4]
            # Assign the time difference to the veto event.
            temp_data_11[i] = event_manager.iat[i, 5]

        # Convert the arrays to a pandas data frame.
        temp_data = {}
        data_container = (
            temp_data_0, temp_data_1, temp_data_2, temp_data_3, temp_data_4,
            temp_data_5, temp_data_6, temp_data_7, temp_data_8, temp_data_9,
            temp_data_10, temp_data_11)

        for i in range(len(self.__names)):
            #print(data_container[i])
            name = self.__names[i]
            temp_data[name] = data_container[i]
            np.save(pathes[name], temp_data[name])

        # Free up memory.
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
        del temp_data_10
        del temp_data_11

        del _ev
        del _pday
        print(80 * " ", end="\r")
        sys.stdout.flush()

        # Assign the attribute.
        self.Data = pd.DataFrame(data=temp_data)


    def __del__(self):
        if self.Data is not None:
            for name in self.__names:
                del self.Data[name]
            del self.Data
        del self.__names

