import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath("."))
import Columns as column_alias
import warnings as warn

# Define the dtypes, names of the columns and names of the files of the
# 'Primary Particle Frame'.
global_dtype_prims = \
    np.array([np.uint32, np.uint64, np.uint16, np.int32, np.uint32,
              np.int32, np.uint32, np.int8, np.uint64, np.uint8])

prims_column_event = "Event Number"
prims_column_energy = "Creation Energy in meV"
prims_column_volume = "Creation Volume"
prims_column_depE = "Deposited Energy in meV"
prims_column_particle = "Particle PDG"
prims_column_particle_id = "Particle ID"
prims_column_parent = "Parent PDG"
prims_column_parent_id = "Parent ID"
prims_column_creation = "Creation Process"
prims_column_stop = "Stopped in Absorber"

global_columns_prims = \
    np.array([prims_column_event, prims_column_energy, prims_column_volume,
              prims_column_particle, prims_column_particle_id,
              prims_column_parent, prims_column_parent_id,
              prims_column_creation, prims_column_depE, prims_column_stop])

global_names_prims = \
    np.array(['event', 'energy', 'volume', 'particle',
              'particle_id', 'parent', 'parent_id', 'process', 'depE', "stop"])

if (global_columns_prims.shape[0] != global_dtype_prims.shape[0]) or \
        (global_columns_prims.shape[0] != global_names_prims.shape[0]) or \
        (global_dtype_prims.shape[0] != global_names_prims.shape[0]):
    message = "Primaries: Missmatching shapes. Edit LoadData.py. "
    message = message + "Check: global_dtype_prims, " + \
            "global_names_prims and global_columns_prims."
    raise RuntimeError(message, RuntimeWarning)

# Define the dtypes, names of the columns and names of the files of the
# 'Deposited Energies Frame'.
global_dtype_depE = \
    np.array([np.uint32, np.uint64, np.uint8, np.uint64, np.uint64, np.int32])

depE_column_event = prims_column_event
depE_column_energy = "Deposited Energy in meV"
depE_column_volume = "Volume"
depE_column_time = "Time in ps"
depE_column_t19 = "Time + xe19 ps"
depE_column_origin = "Origin"

global_columns_depE = \
    np.array([depE_column_event, depE_column_energy, depE_column_volume,
              depE_column_time, depE_column_t19, depE_column_origin])

global_names_depE = \
    np.array(['event', 'energy', 'volume', 'first_time', 't19', 'origin'])

if (global_columns_depE.shape[0] != global_dtype_depE.shape[0]) or \
        (global_columns_depE.shape[0] != global_names_depE.shape[0]) or \
        (global_dtype_depE.shape[0] != global_names_depE.shape[0]):
    message = "Events: Missmatching shapes. Edit LoadData.py. "
    message = message + "Check: global_dtype_depE, " + \
            "global_names_depE and global_columns_depE."
    raise RuntimeError(message, RuntimeWarning)

# Define the dtypes and names of the columns of the other frames.
dtype_processes = {'Process name': np.object, 'Process number': np.int8}
dtype_volumes = {'Volume name': np.object, 'Volume number': np.uint16}
dtype_pdg = {'Particle name': np.object, 'PDG': np.int32}

# Set the path to the simulation data and to the PhotonEvaporation directory.
if sys.platform.startswith('linux'):
    path_to_radio = "/media/alexander/Code/Geant"
else:
    path_to_radio = os.path.join("G:\Geant")
path_to_radio = os.path.join(path_to_radio, "PhotonEvaporation5.3")


def read_level(Z, A, I):
    """ Get the excitation energy of an (excited) nuclide.

    Args:
        Z (int): The atomic number.
        A (int): The mass number.
        I (int): The excitation level.

    Returns:
        string: The excitation energy of the nuclide.
    """

    # Ground state.
    if int(I) == 0:
        return ""

    # Identify the corresponding file and open it.
    name = "z" + str(int(Z)) + ".a" + str(int(A))
    path = os.path.join(path_to_radio, name)
    out = ""
    f = open(path, "r")

    # Iterate over all lines.
    for l in f:
        temp = l.split()
        # Identify the line including the excitation energy.
        if len(temp) == 6:
            # Identify the excitation level.
            if int(I) == int(temp[0]):
                out = "[" + temp[2] + " keV]"
                break
    f.close()
    return out


def get_ion(number):
    """ Get the nuclide name and excitation energy.

    Args:
        number (int): A ten digit int.

    Returns:
        string: The name of the nuclide and its excitation energy.
    """
    # Define the nuclide symbols.
    e_names = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
               "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
               "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
               "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
               "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
               "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
               "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
               "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
               "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
               "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
               "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
               "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]
    # Get the atomic number.
    Z = int(str(int(number))[3:6])
    # Get the mass number.
    A = str(int(number))[6:9]
    #print(Z)
    #print(A)
    return e_names[Z - 1] + "-" + A + read_level(Z, A, int(str(int(number))[-1]))


# TODO: get_number_of_events(),
def get_number_of_events(path):
    """ Identify the number of simulated events.

    Args:
        path (string, list of strings): The path(s) to the simulation data.

    Returns:
        int: The number of simulated events.
    """

    # Identify the file, whose name include the number of simulated events.
    def get_number(path):
        for subdirs, dirs, files in os.walk(path):
            for file_ in files:
                file = str(file_)[2:]
                if ".txt" in file:
                    return int(file.split("_")[0])

    if type(path) == str:
        return get_number(path)
    else:
        num_events = 0
        for p in path:
            num_events += get_number(p)
        return num_events


def get_primary(path):
    """ Identify the primary particle.

    Args:
        path (string, list of strings): The path(s) to the simulation data.

    Returns:
        string: The primary particle.
    """

    # Identify the file, whose name include the primary particle.
    def get_prim(path):
        for subdirs, dirs, files in os.walk(path):
            for file_ in files:
                file = str(file_)[2:]
                if ".txt" not in file:
                    if "_" in file:
                        return file.split("_")[0]

    if type(path) == str:
        return get_prim(path)
    else:
        temp = int(0)
        for p in path:
            if type(temp) == int:
                temp = get_prim(p)
            else:
                if temp != get_prim(p):
                    message = "\nWarning! Found multiple primaries.\n"
                    message = message + "Last one '" + get_prim(p) + \
                            "' is used.\n"
                    warn.warn(message, UserWarning)
                    temp = get_prim(p)
        return temp


def get_prim_volume(path):
    """ Identify the primary particle.

    Args:
        path (string, list of strings): The path(s) to the simulation data.

    Returns:
        string: The primary particle.
    """

    def get_vol(path):
        for subdirs, dirs, files in os.walk(path):
            for file_ in files:
                file = str(file_)[2:]
                if ".txt" not in file:
                    if "_" in file:
                        f_ = file.split(".csv")[0]
                        temp = ""
                        for f in f_.split("_")[2:]:
                            if temp == "":
                                temp = f
                            else:
                                temp += "_" + f
                        return temp

    if type(path) == str:
        return get_vol(path)
    else:
        temp = int(0)
        for p in path:
            if type(temp) == int:
                temp = get_vol(p)
            else:
                if temp != get_vol(p):
                    message = "\nWarning! Found multiple volumes.\n"
                    message = message + "Last one '" + get_vol(p) + \
                            "' is used.\n"
                    warn.warn(message, UserWarning)
                    temp = get_vol(p)
        return temp


def get_con_flag(path):

    def get_con(path):
        for subdirs, dirs, files in os.walk(path):
            for file_ in files:
                file = str(file_)[2:]
                if ".txt" not in file:
                    if "_" in file:
                        return file.split("_")[1]

    if type(path) == str:
        return get_con(path)
    else:
        temp = int(0)
        for p in path:
            if type(temp) == int:
                temp = get_con(p)
            else:
                if temp != get_con(p):
                    message = "\nWarning! Found multiple contamination types.\n"
                    message = message + "Last one '" + get_con(p) + \
                            "' is used.\n"
                    warn.warn(message, UserWarning)
                    temp = get_con(p)
        return temp


def load_column(path, data, col, dtype, mt=False):

    def get_arr(path):
        p = os.path.join(path, data)
        ending = "_" + col + ".binary"

        if mt:
            for i in range(4):
                if i == 0:
                    arr = np.fromfile(
                        os.path.join(p, str(int(i)) + ending), dtype=dtype)
                else:
                    arr = np.append(
                        arr, np.fromfile(
                            os.path.join(p, str(int(i)) + ending), dtype=dtype))
        else:
            try :
                arr = np.fromfile(
                    os.path.join(p, str(int(-2)) + ending), dtype=dtype)
            except FileNotFoundError:
                arr = np.fromfile(
                    os.path.join(p, str(int(-1)) + ending), dtype=dtype)
        return arr

    if type(path) == str:
        return get_arr(path)
    else:
        temp = None
        for p in path:
            if temp is None:
                temp = get_arr(p)
            else:
                if col == prims_column_event:
                    temp = np.append(temp, get_arr(p) + temp.shape[0])
                else:
                    temp = np.append(temp, get_arr(p))
        return temp


def load_column_prims(path, col, dtype, mt=False):
    return load_column(path, "Prims", col, dtype, mt=mt)


def load_column_events(path, col, dtype, mt=False):
    return load_column(path, "Event", col, dtype, mt=mt)


def load_prims(path, mt=False, **kwargs):
    #print("---------------------------------------")
    #print("---------- Loading primaries ----------")
    #print("---------------------------------------")
    event_arg, energy_arg, start_energy_arg, particle_arg, particle_id_arg, \
            parent_arg, parent_id_arg, creation_arg, volume_arg, stop_arg = \
            set_kwargs_primaries(False, **kwargs)

    use_list = [event_arg, start_energy_arg, volume_arg, particle_arg,
                particle_id_arg, parent_arg, parent_id_arg, creation_arg,
                energy_arg, stop_arg]

    if len(use_list) != global_names_prims.shape[0]:
        message = "\nWarning: Primaries -- missmatching shapes. " + \
                "Edit LoadData.py\n"
        message = message + "Check: global_names_prims, " + \
                "and set_kwargs_primaries.\n"
        raise RuntimeError(message, RuntimeWarning)

    arrays = {}
    data = {}

    for i in range(global_names_prims.shape[0]):
        if use_list[i]:
            arrays[i] = load_column_prims(
                path, global_names_prims[i], global_dtype_prims[i], mt=mt)
            #print("- Completed loading " + global_names_prims[i] + ".binary")
            data[global_columns_prims[i]] = arrays[i]

    frame = pd.DataFrame(data=data)
    arrays.clear()
    data.clear()
    #print("---------------------------------------")
    #print("----- Completed loading primaries -----")
    #print("---------------------------------------")
    #print(" ")
    return frame


def load_events(path, mt=False, **kwargs):
    #print("---------------------------------------")
    #print("----------- Loading events ------------")
    #print("---------------------------------------")
    event_arg, energy_arg, volume_arg, time_arg, t19_arg, origin_arg= \
            set_kwargs_events(False, **kwargs)

    use_list = [event_arg, energy_arg, volume_arg, time_arg, t19_arg, origin_arg]

    if len(use_list) != global_names_depE.shape[0]:
        message = "\nWarning: Events -- missmatching shapes. " + \
                "Edit LoadData.py\n"
        message = message + "Check: global_names_depE, " + \
                "and set_kwargs_events.\n"
        raise RuntimeError(message, RuntimeWarning)

    arrays = {}
    data = {}

    for i in range(global_names_depE.shape[0]):
        if use_list[i]:
            arrays[i] = load_column_events(
                path, global_names_depE[i], global_dtype_depE[i], mt=mt)
            #print("- Completed loading " + global_names_depE[i] + ".binary")
            data[global_columns_depE[i]] = arrays[i]

    frame = pd.DataFrame(data=data)
    arrays.clear()
    data.clear()
    #print("---------------------------------------")
    #print("------ Completed loading events -------")
    #print("---------------------------------------")
    #print(" ")
    return frame


def print_message(parameter, p_type, default):
    out_string = "\nType of parameter '" + parameter + "' has to be " + \
            p_type.__name__ + ".\n"
    out_string = out_string + "Parameter is set to default value " + \
            str(default) + ".\n"
    warn.warn(out_string, UserWarning)

def set_variable(parameter, p_type, default, **kwargs):
    if parameter in kwargs:
        if type(kwargs[parameter]) == p_type:
            return kwargs[parameter]
        else:
            print_message(parameter, p_type, default)
            return default

def set_bool(parameter, default, **kwargs):
    return set_variable(parameter, bool, default, **kwargs)


def set_kwargs_primaries(unknown, **kwargs):

    def this_bool(parameter, default):
        return set_bool(parameter, default, **kwargs)

    event_def = True
    energy_def = True
    start_energy_def = False
    particle_def = True
    particle_id_def = False
    parent_def = False
    parent_id_def = False
    creation_def = False
    volume_def = True
    stop_def = True

    event_arg = event_def
    energy_arg = energy_def
    start_energy_arg = start_energy_def
    particle_arg = particle_def
    particle_id_arg = particle_id_def
    parent_arg = parent_def
    parent_id_arg = parent_id_def
    creation_arg = creation_def
    volume_arg = volume_def
    stop_arg = stop_def

    for arg in kwargs:
        if arg in column_alias.event_list:
            event_arg = this_bool(arg, event_def)
        elif arg in column_alias.energy_list:
            energy_arg = this_bool(arg, energy_def)
        elif arg in column_alias.start_energy_list:
            start_energy_arg = this_bool(arg, start_energy_def)
        elif arg in column_alias.particle_list:
            particle_arg = this_bool(arg, particle_def)
        elif arg in column_alias.particle_id_list:
            particle_id_arg = this_bool(arg, particle_id_def)
        elif arg in column_alias.parent_list:
            parent_arg = this_bool(arg, parent_def)
        elif arg in column_alias.parent_id_list:
            parent_id_arg = this_bool(arg, parent_id_def)
        elif arg in column_alias.creation_list:
            creation_arg = this_bool(arg, creation_def)
        elif arg in column_alias.prim_volume_list:
            volume_arg = this_bool(arg, volume_def)
        elif arg in column_alias.prim_stop_list:
            stop_arg = this_bool(arg, stop_def)
        else:
            if unknown:
                message = "\nUnknown parameter '" + arg + \
                        "'. It will be ignored.\n"
                warn.warn(message, UserWarning)

    return event_arg, energy_arg, start_energy_arg, particle_arg, \
        particle_id_arg, parent_arg, parent_id_arg, creation_arg, volume_arg, \
        stop_arg


def set_kwargs_events(unknown, **kwargs):

    def this_bool(parameter, default):
        return set_bool(parameter, default, **kwargs)

    event_def = True
    energy_def = True
    volume_def = True
    time_def = False
    t19_def = False
    origin_def = True

    event_arg = event_def
    energy_arg = energy_def
    volume_arg = volume_def
    time_arg = time_def
    t19_arg = t19_def
    origin_arg = origin_def

    for arg in kwargs:
        if arg in column_alias.event_list:
            event_arg = this_bool(arg, event_def)
        elif arg in column_alias.energy_list:
            energy_arg = this_bool(arg, energy_def)
        elif arg in column_alias.event_volume_list:
            volume_arg = this_bool(arg, volume_def)
        elif arg in column_alias.time_list:
            time_arg = this_bool(arg, time_def)
        elif arg in ["T19", "t19"]:
            t19_arg = this_bool(arg, t19_def)
        elif arg in ["Origin", "origin"]:
            origin_arg = this_bool(arg, origin_def)
        else:
            if unknown:
                message = "\nUnknown parameter " + "'" + arg + \
                        "'. It will be ignored.\n"
                warn.warn(message, UserWarning)

    return event_arg, energy_arg, volume_arg, time_arg, t19_arg, origin_arg
