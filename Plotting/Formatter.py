# TODO: Combine formatter and log_formatter.

def formatter(v):
    """ Convert a string of a number to scientific formatted string.

    Args:
        v (string): A string containing a number.

    Returns:
        string: A string containing a scientific formatted number.
    """

    # Format the string using exponential expressions.
    value = "{:e}".format(v)

    # Define the superscript numbers.
    def sup(s):
        switcher = {"1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶",
                    "7": "⁷", "8": "⁸", "9": "⁹", "0": "⁰", "+": "⁺", "-": "⁻",
                    "(": "⁽", ")": "⁾"}

        if s in switcher:
            return switcher[s]
        return s

    # Zero is zero.
    if v == 0:
        return str(0)
    # Only format numbers which absolute values are large.
    elif ((v >= 0.1) and (v < 1000)) or ((v <= -0.1) and (v > -1000)):
        # Format the string using floats.
        out = "{:f}".format(v)
        # Reduce all non leading zeros.
        while out[-1] == "0":
            out = out[:-1]
        if out[-1] == ".":
            out = out[:-1]
        return out
    else:
        # Get the multiplicity.
        out = str(value).split("e")[0]
        # Reduce all non leading zeros from the multiplicity.
        while out[-1] == "0":
            out = out[:-1]
        if out[-1] == ".":
            out = out[:-1]

        if (out != "1"):
            # The multiplicity differs from 1, thus it has to be multiplied
            # with 10**x.
            out += u"\u2009×\u200910"
        else:
            out = "10"

        # Get the magnitude.
        for i in str(value).split("e")[1]:
            if i != "+":
                out += sup(i)
        return out


def log_formatter(v):
    """ Convert a string of a number, which is a multiple of two,
        to scientific formatted string.

    Args:
        v (string): A string containing a number.

    Returns:
        string: A string containing a scientific formatted number.
    """

    # Format the string using exponential expressions.
    value = "{:e}".format(v)

    # Define the superscript numbers.
    def sup(s):
        switcher = {"1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶",
                    "7": "⁷", "8": "⁸", "9": "⁹", "0": "⁰", "+": "⁺", "-": "⁻",
                    "(": "⁽", ")": "⁾"}

        if s in switcher:
            return switcher[s]
        return s

    # Zero is zero.
    if v == 0:
        out = str(0)
    # Only format numbers which absolute values are large.
    elif ((v >= 0.1) and (v < 1000)) or ((v <= -0.1) and (v > -1000)):

        # Identify the length of the string.
        # TODO: Use numpy.abs and numpy.log10.
        entry = 0
        if v > 0:
            if v < 1:
                entry = 4
            elif v < 10:
                entry = 3
            elif v < 100:
                entry = 2
            else:
                entry = 3
        else:
            if v > -1:
                entry = 5
            elif v > -10:
                entry = 4
            elif v > -100:
                entry = 3
            else:
                entry = 4
        out = "{:f}".format(v)[:entry]

    else:
        # Show 3 characters, meaning one decimal (and the minus sign if existing).
        if v < 0:
            entry = 4
        else:
            entry = 3
        out = str(value).split("e")[0][:entry]

        if out == "1.0":
            out = "10"
        elif out[-1] == "0":
            out = out[0]
        if out != "10":
            # The multiplicity differs from 1, thus it has to be multiplied
            # with 10**x.
            out += u"\u2009×\u200910"

        # Get the magnitude.
        for i in str(value).split("e")[1]:
            if i != "+":
                out += sup(i)

    # Only print numbers, which are multiplicities of two.
    if out.find("×") > -1:
        if v < 0:
            if int(value[1]) % 2 == 1:
                out = ""
        elif v > 0:
            if int(value[0]) % 2 == 1:
                out = ""
    return out
