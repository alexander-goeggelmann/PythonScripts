import numpy as np

# Note: The following functions will be used for the parser in Interpreter.py.

# TODO: The three following functions, create_list, create_list_append and
# create_list_separate can be merged.
def create_list(list_a, list_b, m=0):
    """ Generates an array of combinations of entries of two lists.
        Entries are separated with "", " " and "_".

    Args:
        list_a (list like): List of strings.
        list_b (list like): List of strings.
        m (int, optional): 0, if entries of list_a should be used first.
                           1 for list_b. Defaults to 0.

    Returns:
        numpy.array, numpy.array: Lists of strings.
    """

    temp_list = int(0)
    temp_list_0 = int(0)
    for a in list_a:

        if m == 0:
            if type(temp_list) == int:
                temp_list = np.array([a])
            else:
                temp_list = np.append(temp_list, a)

        for b in list_b:
            if m == 1:
                if type(temp_list) == int:
                    temp_list = np.array([b])
                else:
                    temp_list = np.append(temp_list, b)

            if type(temp_list) == int:
                temp_list = np.array([a + b])
            else:
                temp_list = np.append(temp_list, a + b)
            temp_list = np.append(temp_list, a + "_" + b)

            if type(temp_list_0) == int:
                temp_list_0 = np.array([a + " " + b])
            else:
                temp_list_0 = np.append(temp_list_0, a + " " + b)

        if m == 1:
            m = 2

    return temp_list, temp_list_0


def create_list_append(list_a, list_b, m=0):
    """ Generates an array of combinations of entries of two lists.
        Entries are separated with "" and "_".

    Args:
        list_a (list like): List of strings.
        list_b (list like): List of strings.
        m (int, optional): 0, if entries of list_a should be used first.
                           1 for list_b. Defaults to 0.

    Returns:
        numpy.array: List of strings.
    """

    temp_list = int(0)
    for a in list_a:
        if m == 0:
            if type(temp_list) == int:
                temp_list = np.array([a])
            else:
                temp_list = np.append(temp_list, a)

        for b in list_b:
            if m == 1:
                if type(temp_list) == int:
                    temp_list = np.array([b])
                else:
                    temp_list = np.append(temp_list, b)

            if type(temp_list) == int:
                temp_list = np.array([a + b])
            else:
                temp_list = np.append(temp_list, a + b)
            temp_list = np.append(temp_list, a + "_" + b)

        if m == 1:
            m = 2

    return temp_list

# TODO: Typing error:
def create_list_seperate(list_a, list_b, basis, m=0):
    """ Generates an array of combinations of entries of two lists.
        Entries are separated with " ".

    Args:
        list_a (list like): List of strings.
        list_b (list like): List of strings.
        m (int, optional): 0, if entries of list_a should be used first.
                           1 for list_b. Defaults to 0.

    Returns:
        numpy.array: List of strings.
    """
    temp_list = int(0)
    for a in list_a:
        if m == 0:
            if type(temp_list) == int:
                temp_list = np.array([a])
            else:
                temp_list = np.append(temp_list, a)

        for b in list_b:
            if m == 1:
                if type(temp_list) == int:
                    temp_list = np.array([b])
                else:
                    temp_list = np.append(temp_list, b)

            if type(temp_list) == int:
                temp_list = np.array([a + " " + b])
            else:
                temp_list = np.append(temp_list, a + " " + b)

        if m == 1:
            m = 2

    for a in basis:
        for b in list_b:
            temp_list = np.append(temp_list, a + " " + b)

    return temp_list

event_part_a = np.array(["Event", "event"])
event_part_b = np.array(["Number", "number"])
event_list, event_list_0 = create_list(event_part_a, event_part_b)

energy_part_a = np.array(["Deposited", "deposited"])
energy_part_b = np.array(["Energy", "energy"])
energy_list, energy_list_0 = create_list(energy_part_a, energy_part_b, m=1)
energy_part_c = np.array(["DepE", "Depe", "depe", "depE"])
energy_list = np.append(energy_list, energy_part_c)

start_energy_part_a = np.array(["Start", "start", "Creation", "creation"])
start_energy_list , start_energy_list_0 = create_list(
    start_energy_part_a, energy_part_b, m=2)

prim_volume_part_b = np.array(["Volume", "volume"])
temp_prim_volume_list, temp_prim_volume_list_0 = create_list(
    start_energy_part_a, prim_volume_part_b, m=1)
prim_volume_part_c = np.array(["Number", "number", "Name", "name"])
prim_volume_list = create_list_append(
    temp_prim_volume_list, prim_volume_part_c)
prim_volume_list_0 = create_list_seperate(
    temp_prim_volume_list_0, prim_volume_part_c, prim_volume_part_b)

event_volume_list, event_volume_list_0 = create_list(
    prim_volume_part_b, prim_volume_part_c)

time_part_a = np.array(["Creation", "creation", "First", "first", "Start",
                        "start"])
time_part_b = np.array(["Time", "time"])
time_list, time_list_0 = create_list(time_part_a, time_part_b, m=1)

particle_part_a = np.array(["Particle", "particle"])
particle_part_b = np.array(["PDG", "pdg", "name", "Name", "Number", "number"])
particle_list, particle_list_0 = create_list(
    particle_part_a, particle_part_b)
particle_list = np.append(particle_list, np.array(["PDG", "pdg"]))

parent_part_a = np.array(["Parent", "parent"])
temp_parent_list, temp_parent_list_0 = create_list(
    parent_part_a, particle_part_a)
parent_list = create_list_append(
    temp_parent_list, particle_part_b)
parent_list_0 = create_list_seperate(
    temp_parent_list_0, particle_part_b, parent_part_a)

id_part_a = np.array(["Id", "ID", "id"])
particle_id_list, particle_id_list_0 = create_list(
    particle_part_a, id_part_a, m=2)

parent_id_list = create_list_append(
    temp_parent_list, id_part_a, m=2)
parent_id_list_0 = create_list_seperate(
    temp_parent_list_0, id_part_a, parent_part_a, m=2)

creation_part_a = np.array(["Creation", "creation"])
creation_part_b = np.array(["Process", "process"])
temp_creation_list, temp_creation_list_0 = create_list(
    creation_part_a, creation_part_b, m=1)
creation_list = create_list_append(
    temp_creation_list, prim_volume_part_c)
creation_list_0 = create_list_seperate(
    temp_creation_list_0, prim_volume_part_c, creation_part_b)

origin_part_a = np.array([
    "Origin", "origin", "Source", "source", "Particle", "particle"])
origin_list, origin_list_0 = create_list(
    origin_part_a, particle_part_b)
origin_list = np.append(origin_list, np.array(["PDG", "pdg"]))

prim_stop_list = np.array(["Stop", "Stopping flag", "Killed"])


def column_name(column, list1, list2):
    """ Checks, if (the string) column is in list1 or list2.

    Args:
        column (string): A string.
        list1 (list like): List of strings.
        list2 (list like): List of strings.

    Returns:
        boolean: True, if column is in the lists.
    """

    if column in list1:
        return True
    if column in list2:
        return True
    return False


def column_name_append(column, list1, list2, appendix):
    """ Checks, if (the string) column, or column + appendix, is in list1 or list2.

    Args:
        column (string): A string.
        list1 (list like): List of strings.
        list2 (list like): List of strings.
        appendix (string): A string.

    Returns:
        boolean: True, if column is in the lists.
    """

    for c in list1:
        if column == (c + appendix):
            return True
    for c in list2:
        if column == (c + appendix):
            return True
    return column_name(column, list1, list2)


def column_event(column):
    return column_name(column, event_list, event_list_0)


def column_energy(column):
    return column_name_append(column, energy_list, energy_list_0, " in eV")


def column_creation_energy(column):
    return column_name_append(
        column, start_energy_list, start_energy_list_0, " in eV")


def column_volume(column):
    return column_name(column, prim_volume_list, prim_volume_list_0) or \
        column_name(column, event_volume_list, event_volume_list_0)


def column_first_time(column):
    return column_name_append(column, time_list, time_list_0, " in ps")


def column_particle(column):
    return column_name(column, particle_list, particle_list_0)


def column_origin(column):
    return column_name(column, origin_list, origin_list_0)


def column_parent(column):
    return column_name(column, parent_list, parent_list_0)


def column_particle_id(column):
    return column_name(column, particle_id_list, particle_id_list_0)


def column_parent_id(column):
    return column_name(column, parent_id_list, parent_id_list_0)


def column_process(column):
    return column_name(column, creation_list, creation_list_0)


def in_primaries(column):
    out = column_event(column)
    out = out or column_energy(column)
    out = out or column_creation_energy(column)
    out = out or column_volume(column)
    out = out or column_particle(column)
    out = out or column_particle_id(column)
    out = out or column_parent(column)
    out = out or column_parent_id(column)
    return out or column_process(column)


def in_events(column):
    out = column_event(column)
    out = out or column_energy(column)
    out = out or column_volume(column)
    out = out or column_first_time(column)
    return out or column_origin(column)
