import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import holoviews as hv
import bokeh
import os
import sys
import importlib
import warnings as warn
from tqdm import tqdm
sys.path.append(os.path.abspath("."))

import Columns as column_alias
import Interpreter
import LoadData
importlib.reload(column_alias)
importlib.reload(Interpreter)
importlib.reload(LoadData)

def global_parse_volume(the_frame, volume):
    message = "Cannot find volume"
    default = int(-1)
    return Interpreter.return_value(
        the_frame, "Volume number", "Volume name",
        volume, message, default)


def global_parse_particle(the_frame, particle):
    message = "Cannot find particle"
    default = int(-1)
    if particle == "primary":
        return int(0)
    return Interpreter.return_particle(
        the_frame, "PDG", "Particle name",
        particle, message, default)


def global_parse_process(the_frame, process):
    message = "Cannot find process"
    default = int(-1)
    if process == "primary":
        return int(-1)
    return Interpreter.return_value(
        the_frame, "Process number", "Process name",
        process, message, default)


def global_get_volume(the_frame, number):
    if number > 10000:
        number = int(str(number)[1:])
    message = "Cannot find volume numberd " + str(int(number))
    default = "Error"
    return Interpreter.return_value(
        the_frame, "Volume name", "Volume number",
        number, message, default)


def global_get_particle(the_frame, number):
    message = "Cannot find particle numberd " + str(int(number))
    default = "Error"
    if number == 0:
        return "primary"
    if number > 1000000000:
        return LoadData.get_ion(number)
    return Interpreter.return_value(
        the_frame, "Particle name", "PDG",
        number, message, default)


def global_get_process(the_frame, number):
    message = "Cannot find process numberd " + str(int(number))
    default = "Error"
    if number == -1:
        return "primary"
    return Interpreter.return_value(
        the_frame, "Process name", "Process number",
        number, message, default)


def get_parameters_event(value):
    return int(value), int(value)


def get_parameters_energy(value):
    return Interpreter.convert_energy(value), Interpreter.convert_energy(value)


def get_parameters_volume(parse_function, value):
    if Interpreter.is_int(value):
        cc, cc1 = get_parameters_event(value)
        return cc, cc1, False

    if (value == "absorber") or (value == "Absorber"):
        cc = parse_function("0_Absorber_A0") - 1
        cc1 = parse_function("0_Absorber_F8") + 1
    elif (value == "sensor") or (value == "Sensor"):
        cc = parse_function("3_Sensor_A0") - 1
        cc1 = parse_function("3_Sensor_F8") + 1
    elif (value == "holmium") or (value == "Holmium") or\
            (value == "Ho") or (value == "ho"):
        cc = parse_function("2_Holmium_A0") - 1
        cc1 = parse_function("2_Holmium_F8") + 1
    elif (value == "mmc") or (value == "MMC") or\
            (value == "pixel") or (value == "Pixel"):
        cc = parse_function("4_MMC_A0") - 1
        cc1 = parse_function("4_MMC_F8") + 1
    elif (value == "pcb") or (value == "PCB"):
        cc = parse_function("m_CB_0") - 1
        cc1 = parse_function("m_CB_4") + 1
    elif (value == "panel") or (value == "Panel"):
        cc = parse_function("p_MuonPanel_00") - 1
        cc1 = parse_function("p_MuonPanel_53") + 1
    elif (value == "sbath") or (value == "sBath"):
        cc = parse_function("8_sBath_0") - 1
        cc1 = parse_function("8_sBath_17") + 1
    elif (value == "plug") or (value == "Plug"):
        cc = parse_function("l_Plug_0") - 1
        cc1 = parse_function("l_Plug_8") + 1
    elif (value == "shielding") or (value == "Shielding"):
        cc = parse_function("h_Shielding") - 1
        cc1 = parse_function("k_Shielding5") + 1
    else:
        cc = parse_function(value)
        cc1 = cc

    return cc, cc1, True


def get_parameters_particle(parse_function, value):
    if Interpreter.is_int(value):
        return get_parameters_event(value)
    cc = parse_function(value)
    cc1 = cc
    if cc > 1e9:
        cc -= 1
        cc1 += 10
    return cc, cc1


def get_parameters_time(value):
    return float(value), float(value)


def get_parameters_error(name):
    raise ValueError("Cannot find column " + name)


def apply_operation(the_frame, convert_function, conditions):
    if len(conditions) == 3:
        column, value, value1, v_flag = \
            convert_function(conditions[0], conditions[2])
        if value == value1:
            return Interpreter.apply_condition(
                the_frame[column], conditions[1], value, v_flag=v_flag)

        return Interpreter.apply_condition(
            value, "<", the_frame[column], v_flag=v_flag) &\
            Interpreter.apply_condition(
                the_frame[column], "<", value1, v_flag=v_flag)

    column, value0, vv, v_flag = convert_function(conditions[2], conditions[0])
    column, value1, vv, v_flag = convert_function(conditions[2], conditions[4])
    return Interpreter.apply_condition(
        value0, conditions[1], the_frame[column], v_flag=v_flag) &\
        Interpreter.apply_condition(
            the_frame[column], conditions[3], value1, v_flag=v_flag)


def global_cut(the_frame, the_length, condition, convert_function):
    if ((condition == "") or
        (condition == "all") or
        (condition == "All") or
        (condition == "get all") or
        (condition == "Get All") or
            (condition == "Get all")):
        return np.ones(the_length, dtype=np.bool)

    def apply_op(con):
        return apply_operation(the_frame, convert_function, con)

    out, first_flag, last_flag = Interpreter.get_operation(
        Interpreter.get_ordered_list(
            Interpreter.string_to_list(condition)), apply_op)
    return out


def define_path(path):

    temp_path = 0
    pdg = 0
    volumes = 0
    processes = 0

    if type(path) == str:
        temp_path = os.path.abspath(path)

        volumes = pd.read_csv(
            os.path.join(temp_path, "Volumes.csv"),
                         dtype=LoadData.dtype_volumes,
                         header=0)
        pdg = pd.read_csv(
            os.path.join(temp_path, "PDG.csv"),
                         dtype=LoadData.dtype_pdg,
                         header=0)
        processes = pd.read_csv(
            os.path.join(temp_path, "Processes.csv"),
                         dtype=LoadData.dtype_processes,
                         header=0)
    else:
        temp = int(0)
        for p in path:
            if type(temp) == int:
                temp = np.array([os.path.abspath(p)])
            else:
                temp = np.append(temp, os.path.abspath(p))
        temp_path = temp
        volumes = pd.read_csv(
            os.path.join(temp_path[0], "Volumes.csv"),
                         dtype=LoadData.dtype_volumes,
                         header=0)
        pdg = pd.read_csv(
            os.path.join(temp_path[0], "PDG.csv"),
                         dtype=LoadData.dtype_pdg,
                         header=0)
        processes = pd.read_csv(
            os.path.join(temp_path[0], "Processes.csv"),
                         dtype=LoadData.dtype_processes,
                         header=0)

    return temp_path, pdg, volumes, processes


class Primaries:
    def __init__(self, path, frame=int(0), mt=False, **kwargs):

        self.__event_arg, self.__energy_arg, self.__start_energy_arg, \
            self.__particle_arg, self.__particle_id_arg, self.__parent_arg, \
            self.__parent_id_arg, self.__creation_arg, self.__volume_arg, \
            self.__stop_arg = \
            LoadData.set_kwargs_primaries(True, **kwargs)
        self.__kwargs = kwargs


        self.__Path, self.PDG, self.VolumeNames, self.Processes = \
            define_path(path)
        self.__MultiThread = mt

        if type(frame) != int:
            self.Table = frame
        else:
            self.Table = LoadData.load_prims(self.__Path, mt=mt, **kwargs)
            sort_axes = [LoadData.prims_column_event,
                         LoadData.prims_column_parent_id,
                         LoadData.prims_column_particle_id]
            if not self.__event_arg:
                sort_axes.remove(LoadData.prims_column_event)
            if not self.__parent_id_arg:
                sort_axes.remove(LoadData.prims_column_parent_id)
            if not self.__particle_id_arg:
                sort_axes.remove(LoadData.prims_column_particle_id)
            if len(sort_axes) > 0:
                self.Table = self.Table.sort_values(by=sort_axes)

        if self.__event_arg:
            self.Events = np.array(self.Table[LoadData.prims_column_event])
        if self.__parent_id_arg:
            self.ParentID = np.array(
                self.Table[LoadData.prims_column_parent_id])
        if self.__particle_id_arg:
            self.ParticleID = np.array(
                self.Table[LoadData.prims_column_particle_id])
        if self.__start_energy_arg:
            self.Energies = np.array(self.Table[LoadData.prims_column_energy])
        if self.__energy_arg:
            self.DepE = np.array(self.Table[LoadData.prims_column_depE])
        if self.__volume_arg:
            self.Volumes = np.array(self.Table[LoadData.prims_column_volume])
        if self.__particle_arg:
            self.Particle = np.array(
                self.Table[LoadData.prims_column_particle])
        if self.__parent_arg:
            self.Parent = np.array(self.Table[LoadData.prims_column_parent])
        if self.__creation_arg:
            self.Creation_Process = np.array(
                self.Table[LoadData.prims_column_creation])
        if self.__stop_arg:
            self.Stop = np.array(
                self.Table[LoadData.prims_column_stop])

        self.Length = self.Table.shape[0]


    def __parse_volume(self, volume):
        return global_parse_volume(self.VolumeNames, volume)

    def __parse_particle(self, particle):
        return global_parse_particle(self.PDG, particle)

    def __parse_process(self, process):
        return global_parse_process(self.Processes, process)

    def __get_volume(self, number):
        return global_get_volume(self.VolumeNames, number)

    def __get_particle(self, number):
        return global_get_particle(self.PDG, number)

    def __get_process(self, number):
        return global_get_process(self.Processes, number)

    def __convert_column(self, name, value):
        if column_alias.in_primaries(name):
            pass
        elif column_alias.in_primaries(value):
            tmp = name
            name = value
            value = tmp
        else:
            get_parameters_error(name)

        if column_alias.column_event(name) and self.__event_arg:
            column = LoadData.prims_column_event
            cc, cc1 = get_parameters_event(value)

        elif column_alias.column_energy(name) and self.__energy_arg:
            column = LoadData.prims_column_depE
            cc, cc1 = get_parameters_energy(value)

        elif column_alias.column_volume(name) and self.__volume_arg:
            column = LoadData.prims_column_volume
            cc, cc1, _ = get_parameters_volume(self.__parse_volume, value)

        elif column_alias.column_creation_energy(name) and \
                self.__start_energy_arg:
            column = LoadData.prims_column_energy
            cc, cc1 = get_parameters_energy(value)

        elif column_alias.column_particle(name) and self.__particle_arg:
            column = LoadData.prims_column_particle
            cc, cc1 = get_parameters_particle(self.__parse_particle, value)

        elif column_alias.column_particle_id(name) and self.__particle_id_arg:
            column = LoadData.prims_column_particle_id
            cc, cc1 = get_parameters_event(value)

        elif column_alias.column_parent(name) and self.__parent_arg:
            column = LoadData.prims_column_parent
            cc, cc1 = get_parameters_particle(self.__parse_particle, value)

        elif column_alias.column_parent_id(name) and self.__parent_id_arg:
            column = LoadData.prims_column_parent_id
            cc, cc1 = get_parameters_event(value)

        elif column_alias.column_process(name) and self.__creation_arg:
            column = LoadData.prims_column_creation
            cc, cc1 = get_parameters_particle(self.__parse_process, value)

        else:
            get_parameters_error(name)
        return column, cc, cc1, False

    def __cut(self, condition):
        return global_cut(
            self.Table, self.Length, condition, self.__convert_column)

    def __get_events(self, events):
        if self.__event_arg:
            out = np.zeros(self.Length, dtype=np.bool)
            for event in np.unique(events):
                out |= self.Table[LoadData.prims_column_event] == event
        else:
            out = np.ones(self.Length, dtype=np.bool)
        return out

    def cut(self, condition="", events=int(0)):
        bool_arr = self.__cut(condition)
        if type(events) != int:
            if not self.__event_arg:
                print("Warning: No column 'Event' or similar is found.")
                print("The 'event' input will be ignored.")
            else:
                bool_arr &= self.__get_events(events)
        return Primaries(
            self.__Path, frame=self.Table[bool_arr], mt=self.__MultiThread,
            **self.__kwargs)

    #def plot_energies(self, bins=10000, rang=(0, 1e5), yaxis_log=True):
    #    my_color = bokeh.palettes.Category10_10[0]
    #    counts, edges = np.histogram(
    #        self.Energies, bins=bins, range=rang)

    #    y_label = \
    #        "Counts per " + str(int((rang[-1] - rang[0]) / bins))[:5] + " eV"
    #    if yaxis_log:
    #        y_label = "log(" + y_label + ")"
    #        counts = Interpreter.get_log(counts)
    #    return hv.Histogram((counts, edges)).options(
    #        line_color=my_color, fill_color=my_color).redim.label(
    #            x=LoadData.prims_column_energy, Frequency=y_label)

    def __get_mother(self, event, particleID):
        if not self.__event_arg or not self.__particle_id_arg:
            print("Can not found particle.")
            print("Need 'Event' and 'Particle ID' informations.")
            return

        if not self.__parent_id_arg:
            print("Can not found mother particle.")
            print("Need 'Parente ID' information.")
            return

        condition = "Event = " + str(int(event)) + \
                    " and Particle ID = " + str(int(particleID))
        the_Particle = self.cut(condition=condition)

        condition = "Event = " + str(int(event)) + \
                    " and Particle ID = " + str(int(the_Particle.ParentID[0]))
        #print(condition)
        mother_Particle = self.cut(condition=condition)

        out = []
        out.append(mother_Particle.ParticleID[0])
        if self.__start_energy_arg:
            out.append(mother_Particle.Energies[0])
        if self.__energy_arg:
            out.append( mother_Particle.DepE[0])
        if self.__particle_arg:
            out.append(self.__get_particle(mother_Particle.Particle[0]))
        if self.__creation_arg:
            out.append(self.__get_process(mother_Particle.Creation_Process[0]))
        if self.__volume_arg:
            out.append(self.__get_volume(mother_Particle.Volumes[0]))
        if self.__parent_arg:
            out.append(self.__get_particle(mother_Particle.Parent[0]))
        out.append(mother_Particle.ParentID[0])

        return tuple(out)


    def get_ancestors(self, event, particleID):
        if not self.__event_arg or not self.__particle_id_arg:
            print("Can not found particle.")
            print("Need 'Event' and 'Particle ID' informations.")
            return

        if not self.__parent_id_arg:
            print("Can not found mother particle.")
            print("Need 'Parente ID' information.")
            return

        condition = "Event = " + str(int(event)) + \
                    " and Particle ID = " + str(int(particleID))
        #print(condition)
        the_Particle = self.cut(condition=condition)

        current_values = []
        names = []
        current_values.append(the_Particle.ParticleID[0])
        names.append(LoadData.prims_column_particle_id)
        if self.__start_energy_arg:
            current_values.append(the_Particle.Energies[0])
            names.append(LoadData.prims_column_energy)
        if self.__energy_arg:
            current_values.append( the_Particle.DepE[0])
            names.append(LoadData.prims_column_depE)
        if self.__particle_arg:
            current_values.append(
                self.__get_particle(the_Particle.Particle[0]))
            names.append("Particle Name")
        if self.__creation_arg:
            current_values.append(
                self.__get_process(the_Particle.Creation_Process[0]))
            names.append(LoadData.prims_column_creation)
        if self.__volume_arg:
            current_values.append(self.__get_volume(the_Particle.Volumes[0]))
            names.append(LoadData.prims_column_volume)
        if self.__parent_arg:
            current_values.append(self.__get_particle(the_Particle.Parent[0]))
            names.append("Parent Name")
        current_values.append(the_Particle.ParentID[0])
        names.append(LoadData.prims_column_parent_id)

        array_list = []
        for i in current_values:
            array_list.append(np.array([i]))

        while (current_values[-1] != 0):
            current_values = self.__get_mother(event, current_values[0])
            # print("Particle ID: " + str(particle_id))
            # print("Parent ID: " + str(parent_id))
            for i in range(len(current_values)):
                array_list[i] = np.append(array_list[i], current_values[i])

        data = {}
        for i in range(len(names)):
            data[names[i]] = array_list[i]

        frame = pd.DataFrame(data=data)
        for i in range(len(array_list)):
            array_list[i] = 0
        data.clear()
        return frame


    def get_total_depE(self):
        if not self.__event_arg:
            print("Warning: No column 'Event' or similar is found.")
            print("Can not generate sum of deposited energies.")
            return self.DepE
        else:
            events = np.unique(self.cut(condition="Energy != 0").Events)
            out = np.zeros(events.shape[0])

            for i in tqdm(range(events.shape[0])):
                bool_arr = self.__get_events(np.array([events[i]]))
                out[i] = self.DepE[bool_arr].sum()
            return out


class Events:
    def __init__(self, path, frame=int(0), mt=False, **kwargs):

        self.__event_arg, self.__energy_arg, self.__volume_arg, \
            self.__time_arg, self.__t19_arg, self.__origin_arg = \
                LoadData.set_kwargs_events(True, **kwargs)
        self.__kwargs = kwargs

        self.__Path, self.PDG, self.VolumeNames, self.Processes = \
            define_path(path)
        self.__MultiThread = mt

        if type(frame) != int:
            self.Table = frame
        else:
            self.Table = LoadData.load_events(self.__Path, mt=mt, **kwargs)
            sort_axes = [LoadData.depE_column_event,
                         LoadData.depE_column_volume,
                         LoadData.depE_column_time]
            if not self.__event_arg:
                sort_axes.remove(LoadData.depE_column_event)
            if not self.__volume_arg:
                sort_axes.remove(LoadData.depE_column_volume)
            if not self.__time_arg:
                sort_axes.remove(LoadData.depE_column_time)
            #if not self.__origin_arg:
            #    sort_axes.remove(LoadData.depE_column_origin)
            if len(sort_axes) > 0:
                self.Table = self.Table.sort_values(by=sort_axes)

        if self.__event_arg:
            self.Events = np.array(self.Table[LoadData.depE_column_event])
        if self.__energy_arg:
            self.Energies = np.array(self.Table[LoadData.depE_column_energy])
        if self.__volume_arg:
            self.Volumes = np.array(self.Table[LoadData.depE_column_volume])
        if self.__time_arg:
            self.Times = np.array(self.Table[LoadData.depE_column_time])
        if self.__t19_arg:
            self.T19 = np.array(self.Table[LoadData.depE_column_t19])
        if self.__origin_arg:
            self.Origins = np.array(self.Table[LoadData.depE_column_origin])

        self.Length = self.Table.shape[0]


    def __parse_volume(self, volume):
        return global_parse_volume(self.VolumeNames, volume)

    def __parse_particle(self, particle):
        return global_parse_particle(self.PDG, particle)

    def __parse_process(self, process):
        return global_parse_process(self.Processes, process)

    def __convert_column(self, n, v):
        volume = False
        if column_alias.in_events(n):
            name = n
            value = v
        elif column_alias.in_events(v):
            name = v
            value = n
        else:
            get_parameters_error(name)

        if column_alias.column_event(name) and self.__event_arg:
            column = LoadData.depE_column_event
            cc0, cc1 = get_parameters_event(value)

        elif column_alias.column_energy(name) and self.__energy_arg:
            column = LoadData.depE_column_energy
            cc0, cc1 = get_parameters_energy(value)

        elif column_alias.column_volume(name) and self.__volume_arg:
            column = LoadData.depE_column_volume
            cc0, cc1, volume = \
                get_parameters_volume(self.__parse_volume, value)

        elif column_alias.column_first_time(name) and self.__time_arg:
            column = LoadData.depE_column_time
            cc0, cc1 = get_parameters_time(value)

        elif column_alias.column_origin(name) and self.__origin_arg:
            column = LoadData.depE_column_origin
            cc0, cc1 = get_parameters_particle(self.__parse_particle, value)
        else:
            get_parameters_error(name)

        return column, cc0, cc1, volume

    def __cut(self, condition):
        return global_cut(
            self.Table, self.Length, condition, self.__convert_column)

    def __get_events(self, events):
        if self.__event_arg:
            out = np.zeros(self.Length, dtype=np.bool)
            for event in np.unique(events):
                out |= self.Table[LoadData.depE_column_event] == event
        else:
            out = np.ones(self.Length, dtype=np.bool)
        return out

    def cut(self, condition="", events=int(0)):
        bool_arr = self.__cut(condition)
        if type(events) != int:
            if not self.__event_arg:
                print("Warning: No column 'Event' or similar is found.")
                print("The 'event' input will be ignored.")
            else:
                bool_arr &= self.__get_events(events)

        return Events(self.__Path, frame=self.Table[bool_arr],
                      mt=self.__MultiThread, **self.__kwargs)


class Simulation:
    def __init__(
        self, path, mt=False, primaries=True, events=True,
            prim_kwargs={}, event_kwargs={}):
        self.__Path, self.PDG, self.VolumeNames, self.Processes = \
            define_path(path)

        self.Num_of_Events = LoadData.get_number_of_events(self.__Path)
        self.Contamination_Type = LoadData.get_con_flag(self.__Path)
        self.Primary_Particle = LoadData.get_primary(self.__Path)
        self.Primary_Volume = LoadData.get_prim_volume(self.__Path)

        def check_events(path, check):
            out = os.path.exists(os.path.join(path, "Event"))
            if out:
                out = check
            elif check:
                message = "\nWarning: 'Events' data can not be found in " + \
                        path + ".\n" + "'Events' option will be set to False.\n"
                warn.warn(message, UserWarning)

            return out

        def check_primaries(path, check):
            out = os.path.exists(os.path.join(path, "Prims"))
            if out:
                out = check
            elif check:
                message = "\nWarning: 'Primaries' data can not be found in " + \
                        path + ".\n" + "'Primaries' option will be set to False.\n"
                warn.warn(message, UserWarning)

            return out

        if type(self.__Path) == str:
            self.__uEv = check_events(self.__Path, events)
            self.__uPr = check_primaries(self.__Path, primaries)
        else:
            i = 0
            break_uEv = False
            break_uPr = False
            while i < len(self.__Path):
                self.__uEv = check_events(self.__Path[i], events)
                self.__uPr = check_primaries(self.__Path[i], primaries)
                if not self.__uEv:
                    break_uEv = True
                if not self.__uPr:
                    break_uPr = True
                i += 1
            if break_uPr:
                self.__uPr = False
            if break_uEv:
                self.__uEv = False

        if self.__uEv:
            self.Events = Events(self.__Path, mt=mt, **event_kwargs)
        if self.__uPr:
            self.Primaries = Primaries(self.__Path, mt=mt, **prim_kwargs)


    def parse_volume(self, volume):
        return global_parse_volume(self.VolumeNames, volume)

    def parse_particle(self, particle):
        return global_parse_particle(self.PDG, particle)

    def parse_process(self, process):
        return global_parse_process(self.Processes, process)

    def get_volume(self, number):
        return global_get_volume(self.VolumeNames, number)

    def get_particle(self, number):
        return global_get_particle(self.PDG, number)

    def get_process(self, number):
        return global_get_process(self.Processes, number)
