import numpy as np


def convert_energy(con):
    """ Converts an energy, given as float or string, to a float.

    Args:
        con (float or string): The energy.

    Raises:
        ValueError: If, the energy can not be interpreted.

    Returns:
        float: The energy in eV.
    """

    list_of_units = ["m", "k", "M", "G", "T"]
    list_of_values = [1e-3, 1e3, 1e6, 1e9, 1e12]

    if is_float(con):
        return float(con)

    temp = con.split("eV")[0]
    if is_float(temp):
        return float(temp)

    value = temp.split()[0]
    unit = temp.split()[-1]

    # The value and unit are seperated by spaces: e.g. 10 keV.
    if value != unit:
        value = float(value)
        for i, u in enumerate(list_of_units):
            if unit == u:
                return value * list_of_values[i]
        raise ValueError("Can not interpret energy: " + con)

    # There was no space character in temp: e.g. 10keV.
    for i, u in enumerate(list_of_units):
        if u in value:
            return float(value.split(u)[0]) * list_of_values[i]
    raise ValueError("Can not interpret energy: " + con)


def get_log(data):
    """ Determines the log10 for each entry in data, by setting a lower limit.
        Because, log10(0) -> infinity.

    Args:
        data (numpy.array): Array of numbers.

    Returns:
        numpy.array: Array of numbers.
    """

    out = np.zeros(data.shape[0]) + (np.log(data[data > 0].min()) / 10.)
    for i, d in enumerate(data):
        if d > 0:
            out[i] = np.log10(d)
    return out


def return_value(frame, col0, col1, value, message, default):
    if frame[col0][frame[col1] == value].shape[0] > 0:
        return frame[col0][frame[col1] == value].iloc[0]
    else:
        print(message)
        return default


def get_pdg(name):
    """ Get the Particle Data Group number of a nuclide.

    Args:
        name (string): The name of a nuclide. E.g. H-3.

    Returns:
        int: The PDG number.
    """

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

    if "-" in name:
        first = name.split("-")[0]
        second = name.split("-")[1]
    else:
        f = 0
        l = 0
        number = False

        for i in range(len(name)):
            try:
                is_value = int(name[i])
                if number:
                    l = i
                else:
                    f = i
                    number = True
            except ValueError:
                pass
        if f == 0:
            first = name[:l+1]
            second = name[l+1:]
        else:
            first = name[:f]
            second = name[f:]

    #print(first)
    #print(second)

    try:
        A = int(first)
        n_name = second
    except ValueError:
        A = int(second)
        n_name = first

    Z = 0
    for i in range(len(e_names)):
        if n_name == e_names[i]:
            Z = i + 1

    if Z == 0:
        return 0

    Z *= 1e4
    A *= 10

    return int(1e9 + Z + A)


def return_particle(frame, col0, col1, value, message, default):
    """ Returns the PDG number of a nuclide. Or, if existing,  returns data
        connected to this nuclide.

    Args:
        frame (pandas.DataFrame): The data frame.
        col0 (string): Data column name.
        col1 (string): PDG column name.
        value (string, int): Either the nuclide name or PDG number.
        message (string): An error message.
        default (int): Error output.

    Returns:
        int: The PDG number.
    """

    # If value is a PDG number, get associated data.
    if frame[col0][frame[col1] == value].shape[0] > 0:
        return frame[col0][frame[col1] == value].iloc[0]
    else:
        # Get the PDG number of the nuclide.
        out = get_pdg(value)
        if out == 0:
            print(message)
            return default
        else:
            return out


# TODO: is_int() and is_float() can be combined.
def is_int(con):
    """ Check if a string can be interpreted as int.

    Args:
        con (string): A string

    Returns:
        boolean: If the string can be interpreted as int.
    """

    try:
        _ = int(con)
        return True
    except ValueError:
        return False


def is_float(con):
    """ Check if a string can be interpreted as float.

    Args:
        con (string): A string

    Returns:
        boolean: If the string can be interpreted as float.
    """

    try:
        _ = float(con)
        return True
    except ValueError:
        return False


# TODO: For append_list(), add_letter(), is_empty() and delete_empty():
# Why not use the_list[-depth]? I think there should be a reason.
# I would not use recursions, if there are more simpler ways.
def append_list(the_list, depth):
    """ Appends the -depth-th entry of a list of lists by an empty list.

    Args:
        the_list (list): A list of lists.
        depth (int): The index.
    """

    # Extend the list of lists.
    if depth == 0:
        the_list.append([])
    # Extend a list of the list.
    else:
        append_list(the_list[-1], depth - 1)


def add_letter(the_list, depth, letter):
    """ Adds a character/number to the -depth-th entry of a list of lists.

    Args:
        the_list (list): A list of lists.
        depth (int): The index.
        letter (char like): A character.
    """

    if depth == 0:
        try:
            if len(the_list[-1]) != 0:
                the_list[-1][-1] += letter
            else:
                the_list[-1] = letter
        # Catch the_list[-1]: Empty list.
        except IndexError:
            the_list.append(letter)
        # Cath the_list[-1][-1] += letter: E.g. filled with a number.
        except TypeError:
            the_list[-1] += letter
    else:
        add_letter(the_list[-1], depth - 1, letter)


def is_empty(the_list, depth):
    """ Checks if the -depth-th entry of a list of lists is empty.

    Args:
        the_list (list): A list of lists.
        depth (int): The index.
    """

    if depth == 0:
        try:
            return len(the_list[-1]) == 0
        except IndexError:
            return False
    else:
        return is_empty(the_list[-1], depth - 1)


def delete_empty(the_list, depth):
    """ Delete the -depth-th entry of a list of lists.

    Args:
        the_list (list): A list of lists.
        depth (int): The index.
    """

    if depth == 0:
        del the_list[-1]
    else:
        delete_empty(the_list[-1], depth - 1)


def string_to_list(condition):
    """ Translate a string to interpretable conditions.

    Args:
        condition (string): A string including conditions.

    Returns:
        list: List of conditions.
    """

    list_of_conditions = []
    deepness = 0

    # Iterate over all characters.
    for letter in condition:
        # Start new condition.
        if (letter == "("):
            append_list(list_of_conditions, deepness)
            deepness += 1
        # End condition.
        elif (letter == ")"):
            if is_empty(list_of_conditions, deepness):
                delete_empty(list_of_conditions, deepness)
            deepness -= 1
            # There are sub-conditions.
            if deepness > 0:
                append_list(list_of_conditions, deepness)
        # Add the character to the current condition.
        else:
            add_letter(list_of_conditions, deepness, letter)
    return list_of_conditions


def insert_values(out_list, in_string):
    """ Translate logical operators to binary operators.

    Args:
        out_list (list): Translated list.
        in_string (string): String containing operators.
    """

    out_list.append([])
    and_ops = ["and", "&", "&&"]
    or_ops = ["or", "|", "||"]

    was_op = True

    # Loop over all operators/sub-strings.
    for cc in in_string.split():
        # Found and/or operators.
        if (cc in and_ops) or (cc in or_ops):
            if cc in and_ops:
                t_cc = "and"
            else:
                t_cc = "or"
            if len(out_list[-1]) == 0:
                out_list[-1] = [t_cc]
            else:
                out_list.append([t_cc])
            out_list.append([])
            was_op = True
        else:
            # Found relation operators.
            if is_operator(cc):
                out_list[-1].append(cc)
                was_op = True
            # Found an ordinary string.
            else:
                if was_op:
                    out_list[-1].append(cc)
                else:
                    out_list[-1][-1] += " " + cc
                was_op = False
    if len(out_list[-1]) == 0:
        del out_list[-1]


def get_entries(out_list, conditions):
    """ Replace all logical operators by binary ones.

    Args:
        out_list (list): Output list of conditions.
        conditions (list): Input list of conditions.
    """

    if type(conditions) == list:
        for con in conditions:
            out_list.append([])
            get_entries(out_list[-1], con)
    else:
        insert_values(out_list, conditions)


def get_ordered_list(unordered_list):
    """ Replace all logical operators by binary ones.

    Args:
        unordered_list (list): List of conditions.

    Returns:
        list: List of conditions.
    """

    out = []
    get_entries(out, unordered_list)
    # TODO: temp_out is not used.
    #temp_out = []
    #for i in out:
    #    temp_out.append(i)
    return out

# TODO: Should also working on arrays of strings, which can be interpreted as
#       numbers.
def thousands(values):
    """ Removes all digits except of the last 4 of any number of an array.

    Args:
        values (numpy.array): An array of numbers.

    Returns:
        numpy.array: An array of numbers < 10000.
    """

    if (values[values > 10000].shape[0] == 0):
        return values
    else:
        copy = np.array(values.copy())
        for i in range(copy.shape[0]):
            while copy[i] >= 10000:
                copy[i] -= 10000
        return copy


def apply_condition(con, op, concon, v_flag=False):
    """ Compare two values.

    Args:
        con (number): A number.
        op (string): An operator.
        concon (number): A number.
        v_flag (bool, optional): Defines if one of the two comparative values
                                 represents a volume, which is weighted with
                                 the time. Defaults to False. E.g. 5, 10005, are
                                 the same volumes, but with multiple hits.

    Returns:
        boolean: Comparision of the two numbers.
    """

    if v_flag:
        if (type(con) != int) and (type(con) != float):
            con1 = thousands(con)
            con2 = concon
        else:
            con1 = con
            con2 = thousands(concon)
    else:
        con1 = con
        con2 = concon
    if (op == "=") or (op == "=="):
        return con1 == con2
    if op == "!=":
        return con1 != con2
    if op == ">":
        return con1 > con2
    if op == ">=":
        return con1 >= con2
    if op == "<":
        return con1 < con2
    if op == "<=":
        return con1 <= con2
    print("Warning: Cannot apply operation " + op + ".")
    return np.ones(con1.shape[0], dtype=np.bool)


def is_operator(op):
    """ Checks if the given string is an operator.

    Args:
        op (string): A string.

    Returns:
        boolean: If the input string is an operator or not.
    """

    if (op == "=") or (op == "=="):
        return True
    if op == "!=":
        return True
    if op == ">":
        return True
    if op == ">=":
        return True
    if op == "<":
        return True
    if op == "<=":
        return True
    return False


def get_string_operation(condition, apply_operation):
    """ Translates the statements in condition to booleans.

    Args:
        condition (list): List of (lists of) strings.
        apply_operation (func): Function defining how to interpret the statements.

    Returns:
        numpy.array, int, int: Array of executed comparisions, type of start
                               condition, type of last condition.
    """

    out = np.zeros(1)
    first_flag = 0
    last_flag = 0

    # Counter for the amount of statements in condition.
    counter = 0
    first = 0
    last = len(condition) - 1
    is_first = True

    temp_or = False
    temp_and = False
    for con in condition:
        # Conditions have to be seperated by operators. If the first or last
        # statement in condition is an operator, these will be stored for
        # further calculations.
        if ((counter == first) or (counter == last)) and (len(con) == 1):
            if con[0] == "or":
                flag = 1
            elif con[0] == "and":
                flag = 2

            if counter == first:
                first_flag = flag
            if counter == last:
                last_flag = flag
        # Remember the current operator for the next calculation.
        elif (len(con) == 1):
            if con[0] == "or":
                temp_or = True
            elif con[0] == "and":
                temp_and = True
        # Save the first statement (if it is not an operator).
        elif is_first:
            out = apply_operation(con)
            is_first = False
        # Apply the last remembered operator.
        else:
            if temp_or:
                out |= apply_operation(con)
                temp_or = False
            elif temp_and:
                out &= apply_operation(con)
                temp_and = False
        # Increase the counter of statements.
        counter += 1
    return out, first_flag, last_flag


def add_operation(in_list, out_list, flag):
    """ Combines two arrays of booleans.
    Args:
        in_list (numpy.array): An array of booleans.
        out_list (numpy.array): An array of booleans.
        flag (int): 1 for a |= b, 2 for a &= b.
    """

    if flag == 1:
        out_list |= in_list
    elif flag == 2:
        out_list &= in_list
    else:
        print("Nothing happend")


def get_operation(conditions, apply_operation):
    """ Executes the given interpreted statements.

    Args:
        conditions (list): List of statements.
        apply_operation (func): Interpretation of the output.

    Returns:
        numpy.array, int, int: Cut array, first and last condition flags.
    """

    out = np.zeros(10)
    first_flag = 0
    last_flag = 0

    first = 0
    counter = 0
    # Nothing to iterate through.
    if not list_in_list(conditions):
        return get_string_operation(conditions, apply_operation)

    for condition in conditions:
        # Initialize the output.
        if (counter == first):
            out, first_flag, last_flag = \
                get_operation(condition, apply_operation)
        else:
            # Calculate the part solution.
            temp_out, temp_first_flag, temp_last_flag = \
                get_operation(condition, apply_operation)

            # Note: The 'consditions' are connected by 'and' and 'or'.
            # One of last_flag and temp_first_flag is zero.
            # The other is either 1 for 'or', or 2 for 'and'.
            if last_flag > temp_first_flag:
                flag = last_flag
            else:
                flag = temp_first_flag

            # It could be the case, that there is nothing to execute
            # For example if an unnecessary bracket is included in 'condition'.
            # E.g. 'a and (b)' results in two 'conditions': [a and], [b].
            if len(temp_out) > 1:
                add_operation(temp_out, out, flag)
            first_flag = temp_first_flag
            last_flag = temp_last_flag

        counter += 1

    return out, first_flag, last_flag


def list_in_list(the_list):
    """ Checks if a list consits of list of lists.

    Args:
        the_list (list): A list.

    Returns:
        boolean: If the list consits of lists of lists.
    """

    for l in the_list:
        for k in l:
            if type(k) == list:
                return True
    return False
