import os
import sys
import matplotlib
import numpy as np
from datetime import datetime
###############################################################################
#   Information to the data structure:

#       path -- Path to dictionary which contained the ADC-channels
#           e.g. +- home/user/measurement01
#                   +- ADC15
#                       +- BASE
#                           +- 1_100
#                               +- S1.sraw
#                               +- S2.sraw
#                               +- ...
#                           +- ...
#                       +- NEGP
#                           +- 1_100
#                               +- S1.sraw
#                               +- S2.sraw
#                               +- ...
#                           +- ...
#                       +- POSP
#                           +- ...
#                   +- ADC16
#                       +- ...
#                   +- ...
#                   +- Fitresults
#                       +- Pixel29.fit
#                       +- Pixel30.fit
#                       +- ...
###############################################################################

# Names and extentions.
PATH_SEP = os.path.sep
NEGP = "NEGP"
POSP = "POSP"
ADC = "ADC"
PS = "S"
SRAW = ".sraw"

# Start entry of pulses.
FIRST_E = 4006

# Some pathes.
if sys.platform.startswith('linux'):
    ROOT_PATH = "/media/alexander/Code/PhytonScripts"
    ROOT_PATH_DEFAULT = "/media/alexander/Seagate/ECHoData"
    ROOT_PATH_TOSHIBA = "/media/alexander/TOSHIBA EXT/ECHoData"
else:
    ROOT_PATH =  os.path.join("G:\PhytonScripts")
    ROOT_PATH_DEFAULT = os.path.join("K:\ECHoData")
    ROOT_PATH_TOSHIBA = os.path.join("E:\ECHoData")

PATH_TO_PLOTTING = os.path.join(ROOT_PATH, "Plotting")
LANDAU_PATH = os.path.join(ROOT_PATH, "Theory", "Landau_Data")
LANDAU_E_PATH = os.path.join(ROOT_PATH, "Theory", "Landau_Electron_Data")

###############################################################################

# Calibration energies/lines.
CALIBRATION_ENERGY_HO = np.array([330., 418., 1851., 2053.])
CALIBRATION_ENERGY_FE = np.array([5890., 6490.])
CALIBRATION_ENERGY_BOTH = np.array(
    [330., 418., 1851., 2053., 5890., 6490.])

# Filenames
FILE_PIXEL_LIST = "Pixels.csv"
FILE_TEMPLATE = "Template.npy"

FILE_CALIBRATION_SIGNALS = "Calibration_Signals.npy"
FILE_CALIBRATION_HIST_Y = "Calibration_Hist_Y.npy"
FILE_CALIBRATION_HIST_X = "Calibration_Hist_X.npy"
FILE_CALIBRATION_PEAKS = "Calibration_Peaks.npy"
FILE_CALIBRATION_DETECTED_LINES = "Calibration_Detected_Lines.npy"
FILE_CALIBRATION_INDEX = "Calibration_Index.csv"
FILE_CALIBRATION_LINES = "Calibration_Lines.csv"
FILE_CALIBRATION_DECAY_TIME = "Decay_Time.npy"

FILE_MUON_VETO = "MuonVeto.npy"

FILE_PIXEL_CALIBRATION = "Pixel_Calibration.npy"

FILE_ELLIPSE_CENTERS = "Ellipse_Centers.npy"
FILE_ELLIPSE_AS = "Ellipse_As.npy"
FILE_ELLIPSE_BS = "Ellipse_Bs.npy"
FILE_ELLIPSE_PHIS = "Ellipse_Phis.npy"
FILE_ELLIPSE_SCALES = "Ellipse_Scales.npy"
FILE_ELLIPSE_RATIOS = "Ellipse_Ratios.npy"

###############################################################################

# Column names.
COLUMN_POLARITY = "Polarity"
COLUMN_MAX_VALUE = "Max_Value"
COLUMN_CAL_INDEX = "Cal_Index"
COLUMN_K = "K"
COLUMN_M = "M"


PIXEL_EVENTS = "Pixel_Events"
EVENT_MANAGER = "Event_Manager"
PIXEL_COINCIDENCES = "Pixel_Coincidences"
PIXEL_NONCOINCIDENCES = "Pixel_Noncoincidences"

# Columns of a pixel event:                 Index
COLUMN_ADC_CHANNEL = 'ADC_Channel'          # 0
COLUMN_PIXEL_NUMBER = 'Pixel_Number'        # 1
COLUMN_SIGNAL_NUMBER = 'Signal_Number'      # 2
COLUMN_TIMESTAMP = 'Timestamp'              # 3
COLUMN_LAST_TIMESTAMP = 'Last_Timestamp'    # --    (4)
COLUMN_OFFSET = 'Offset'                    # --    (5)
COLUMN_SLOPE = 'Slope'                      # --    (6)
COLUMN_FILTER_AMP = 'Filter_Amp'            # 4     (7)
COLUMN_TEMPLATE_AMP = 'Template_Amp'        # 5     (8)
COLUMN_DERIVATIVE_AMP = 'Derivative_Amp'    # 6     (9)
COLUMN_FULL_INTEGRAL = 'Integral_Amp'       # 7     (10)
COLUMN_RISE_INTEGRAL = 'Rise_Integral_Amp'  # --    (11)
COLUMN_HEIGHT = "Height"                    # --    (12)
COLUMN_TEMPLATE_CHI = 'Template_Chi'        # 8     (13)
COLUMN_DECAY_TIME = 'Decay_Time'            # 9     (14)

PIXEL_NAMES = [COLUMN_ADC_CHANNEL,          # 0
               COLUMN_PIXEL_NUMBER,         # 1
               COLUMN_SIGNAL_NUMBER,        # 2
               COLUMN_TIMESTAMP,            # 3
               # COLUMN_LAST_TIMESTAMP,     # --    (4)
               # COLUMN_OFFSET,             # --    (5)
               # COLUMN_SLOPE,              # --    (6)
               COLUMN_FILTER_AMP,           # 4     (7)
               COLUMN_TEMPLATE_AMP,         # 5     (8)
               COLUMN_DERIVATIVE_AMP,       # 6     (9)
               COLUMN_FULL_INTEGRAL,        # 7     (10)
               # COLUMN_RISE_INTEGRAL,      # --    (11)
               # COLUMN_HEIGHT,             # --    (12)
               COLUMN_TEMPLATE_CHI,         # 8     (13)
               COLUMN_DECAY_TIME]           # 9     (14)

PIXEL_DTYPES = [np.uint8,       # 0
                np.uint8,       # 1
                np.uint64,      # 2
                np.uint64,      # 3
                # np.uint64,    # --    (4)
                # np.float64,   # --    (5)
                # np.float64,   # --    (6)
                np.float64,     # 4     (7)
                np.float64,     # 5     (8)
                np.float64,     # 6     (9)
                np.float64,     # 7     (10)
                # np.float64,   # --    (11)
                # np.float64,   # --    (12)
                np.float64,     # 8     (13)
                np.float64]     # 9     (14)

# Columns of an event:
COLUMN_PIXEL_X = 'Pixel_'
COLUMN_TIME_WINDOW = 'Time_Window'              # 0
COLUMN_NUM_OF_COINS = 'Number_Of_Coincidences'  # 1
COLUMN_SELF_COIN = 'Self_Coincidental'          # 2
COLUMN_MUON_FLAG = 'Muon_Flag'                  # 3
COLUMN_FIRST_TIMESTAMP = 'First_Timestamp'      # 4
COLUMN_DELTA_TIME_MUON = 'Delta_Time_Muon'      # 5

EVENT_NAMES = [COLUMN_FIRST_TIMESTAMP,  # 0
               COLUMN_TIME_WINDOW,      # 1
               COLUMN_NUM_OF_COINS,     # 2
               COLUMN_SELF_COIN,        # 3
               COLUMN_MUON_FLAG,        # 4
               COLUMN_DELTA_TIME_MUON]  # 5
EVENT_PIXEL_DTYPE = np.uint64

EVENT_DTYPES = [np.int64,   # 0
                np.int64,   # 1
                np.uint8,   # 2
                np.uint8,   # 3
                np.uint8,   # 4
                np.int64]   # 5

VETO_DTYPE = np.uint64

# Columns for coincidental events
COLUMN_COINCIDENCE_CODE = 'Coincidence_Code'

# Columns of frame of time differences
str_time_diff = 'Time_Difference'

# Names of some files
str_event_name = "Event.csv"

# DEFAULT_PATH = os.path.join(ROOT_PATH_DEFAULT, "DefaultPixel")
# DEFAULT_PATH = os.path.join(
#    ROOT_PATH_DEFAULT, "Run24-Asymmetric",
#    "ECHo-1k_Run24_Ag_191223_10mK_0_asymmetric_channels")
DEFAULT_PATH = os.path.join(
    ROOT_PATH_TOSHIBA, "Run25", "ECHo1k_Run25_Ag_20200319_54ms_4")

# DEFAULT_PIXEL = 23
# DEFAULT_PIXEL = 3
DEFAULT_PIXEL = 12


def show_progress(iterator, max_iterator, t0, steps=100, out_str="Processing"):
    """ Prints the progress in % and left computing time.

    Args:
        iterator (int): Current iterator step.
        max_iterator (int): Maximal iterations.
        t0 (datetime): Start time.
        steps (int, optional): How much steps should be shown. Defaults to 100.
                               100 means 1% steps. 10 means 10% steps.
        out_str (str, optional): Message which is printed at each step. Defaults to "Processing".
    """
    print_value = int(max_iterator / steps)
    if print_value == 0:
        print_value = 1

    # Iterator has to be a multiple of print_value
    if iterator % print_value == 0:
        # Computing time until now.
        time_passed = datetime.now() - t0
        time_passed = time_passed.total_seconds()

        # Computing progress until now.
        percent = 100 * iterator / max_iterator
        if percent > 0:
            # Left computing time.
            time_needed = 100 * time_passed / percent
        else:
            time_needed = 0.0

        # Get passed minutes.
        minutes_passed = int(time_passed / 60)
        if minutes_passed > 0:
            minutes_passed = str(minutes_passed) + " min "
        else:
            minutes_passed = ""

        # Get needed minutes to completeness.
        minutes_needed = int(time_needed / 60)
        if minutes_needed > 0:
            minutes_needed = str(minutes_needed) + " min "
        else:
            minutes_needed = ""

        # Get passed seconds (after subtracting minutes)-
        seconds_passed = int(time_passed % 60)
        if seconds_passed > 0:
            seconds_passed = str(seconds_passed) + " s"
        else:
            seconds_passed = ""

        # Get needed seconds (after subtracting minutes) to completeness.
        seconds_needed = int(time_needed % 60)
        if seconds_needed > 0:
            seconds_needed = str(seconds_needed) + " s"
        else:
            seconds_needed = ""

        percent = str(percent).split(".")[0] + " %"
        time_passed = minutes_passed
        time_passed += seconds_passed
        if time_passed == "":
            time_passed = "0 s"
        time_needed = minutes_needed
        time_needed += seconds_needed

        # Output string
        prog = out_str + " -- " + percent + " " + \
                "[" + time_passed + " -> " + \
                time_needed + "]"

        # Output should have a length of 80 characters.
        print(prog + (80 - len(prog)) * " ", end="\r")
        sys.stdout.flush()


def is_pulses_dir(dir_itr):
    """ Check if directory is a parent directory of pulse files.

    Args:
        dir_itr (itr): Iterator of pathes.

    Returns:
        bool: Wether directory is parent directory of pulses or not.
    """
    if os.path.isdir(dir_itr.path) and ("1_" in dir_itr.name) and \
            (dir_itr.name[-3:] == "000") and \
            (len(os.listdir(dir_itr.path)) > 0):
        return True

    return False

def is_pulses_dir_(dir_path, dir_name):
    """ Check if directory is a parent directory of pulse files.

    Args:
        dir_path (string): Path to parent directory including directory dir_name.
        dir_name (string): Directory name.

    Returns:
        bool: Wether directory is parent directory of pulses or not.
    """
    if os.path.isdir(dir_path) and ("1_" in dir_name) and \
            (dir_name[-3:] == "000") and \
            (len(os.listdir(dir_path)) > 0):
        return True

    return False


def set_matplotlib_rc():
    """ Set up matplotlib properties. """
    matplotlib.rcParams['figure.facecolor'] = 'white'
    matplotlib.rcParams['font.size'] = 18 # 25
    matplotlib.rcParams['figure.figsize'] = [15.0, 10.0]
    matplotlib.rcParams['text.usetex'] = False
    matplotlib.rcParams['axes.grid'] = True
    matplotlib.rcParams['grid.alpha'] = 1.0

    matplotlib.rcParams['font.weight'] = 'normal'
