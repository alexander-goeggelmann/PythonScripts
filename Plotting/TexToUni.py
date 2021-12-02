def tex_to_uni(text):
    """ Converts latex commands inside a string to unicode characters.

    Args:
        text (string): A text containing latex commands.

    Returns:
        string: A text containing unicode characters.
    """

    # TODO: Add the case of brackets in brackets.

    # Due to the backslashes, the string has to be decoded.
    string = text.encode('unicode-escape').decode()
    # Initialize the output.
    out = ""

    # Define the latex commands.
    operators = [r"\pm", r"\,", r"\cdot", r"\times", r"\leq",
                 r"\geq", r"\sum", r"\simeq", r"\sim", r"\approx",
                 r"\int", r"\propto", r"\ll", r"\gg", r"\lefarrow",
                 r"\rightarrow", r"\Leftarrow", r"\Rightarrow", r"\neq",
                 r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon",
                 r"\zeta", r"\eta", r"\theta", r"\iota", r"\kappa", r"\lambda",
                 r"\mu", r"\nu", r"\xi", r"\omicron", r"\pi", r"\rho",
                 r"\sigma", r"\tau", r"\upsilon", r"\phi", r"\chi", r"\psi",
                 r"\omega", r"\Gamma", r"\Delta", r"\Theta", r"\Lambda",
                 r"\Xi", r"\Pi", r"\Sigma", r"\Phi", r"\Psi", r"\Omega"]
    # Define the command results.
    operators_u = ["±", u"\u2009", "⋅", "×", "≤",
                   "≥", "∑", "≃", "∾", "≈",
                   "∫", "∝", "≪", "≫", "←",
                   "→", "⇐", "⇒", "≠",
                   "α", "β", "γ", "δ", "ε",
                   "ζ", "η", "θ", "ι", "κ", "λ",
                   "μ", "ν", "ξ", "ο", "π", "ρ",
                   "σ", "τ", "υ", "ϕ", "χ", "ψ",
                   "ω", "Γ", "Δ", "Θ", "Λ"
                   "Ξ", "Π", "Σ", "Φ", "Ψ", "Ω"]

    # Flag for the identification of backslashs.
    operate = False
    # Storage of the identified command.
    temp = ""
    # Flag for sub- and superscripts.
    cast = 0
    # Flag for brackets.
    multi = False

    # Iterate over all characters.
    for s in string:
        # Backslash is identified. Start the latex command.
        if s == "\\":
            operate = True
            temp = s
        # Add the character to the command.
        elif operate:
            # End of the command.
            if s == " ":
                operate = False
                # Add the decoded command to the output.
                out += temp + " "
                # Reset temp.
                temp = ""
            else:
                temp += s
                # A command is identified.
                if temp in operators:
                    operate = False
                    # Translate the command.
                    i = 0
                    while i < len(operators):
                        # Command found.
                        if temp == operators[i]:
                            # Superscript flag is set.
                            if cast == 2:
                                out += supers(operators_u[i])
                                # Check if the bracket flag is set.
                                if not multi:
                                    cast = 0
                            # Subscript flag is set.
                            elif cast == 1:
                                out += subs(operators_u[i])
                                # Check if the bracket flag is set.
                                if not multi:
                                    cast = 0
                            # Add the decoded command to the output.
                            else:
                                out += operators_u[i]
                            # Reset temp.
                            temp = ""
                            break
                        i += 1
        else:
            # Set the subscript flag.
            if s == "_":
                cast = 1
            # Set the superscript flag.
            elif s == "^":
                cast = 2
            else:
                # Set the bracket flag.
                if s == "{":
                    multi = True
                # Reset the bracket flag.
                elif s == "}":
                    multi = False
                    cast = 0
                else:
                    # Superscript flag is set.
                    if cast == 2:
                        out += supers(s)
                        if not multi:
                            cast = 0
                    # Subscript flag is set.
                    elif cast == 1:
                        out += subs(s)
                        if not multi:
                            cast = 0
                    # Nothing to do. Simply add the character to the output.
                    else:
                        out += s
    return out


def supers(s):
    """ If existing, return the superscript character.

    Args:
        s (char): A character.

    Returns:
        char: A superscript character.
    """

    switcher = {"a": "ᵃ", "b": "ᵇ", "c": "ᶜ", "d": "ᵈ", "e": "ᵉ", "f": "ᶠ",
                "g": "ᵍ", "h": "ʰ", "i": "ⁱ", "j": "ʲ", "k": "ᵏ", "l": "ˡ",
                "m": "ᵐ", "n": "ⁿ", "o": "ᵒ", "p": "ᵖ", "r": "ʳ", "s": "ˢ",
                "t": "ᵗ", "u": "ᵘ", "v": "ᵛ", "w": "ʷ", "x": "ˣ", "y": "ʸ",
                "z": "ᶻ", "A": "ᴬ", "B": "ᴮ", "D": "ᴰ", "E": "ᴱ", "G": "ᴳ",
                "H": "ᴴ", "I": "ᴵ", "J": "ᴶ", "K": "ᴷ", "L": "ᴸ", "M": "ᴹ",
                "N": "ᴺ", "O": "ᴼ", "P": "ᴾ", "R": "ᴿ", "T": "ᵀ", "U": "ᵁ",
                "V": "ⱽ", "W": "ᵂ", "1": "¹", "2": "²", "3": "³", "4": "⁴",
                "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", "0": "⁰",
                "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾", "α": "ᵅ",
                "β": "ᵝ", "γ": "ᵞ", "δ": "ᵟ", "ε": "ᵋ",
                "ι": "ᶥ", "θ": "ᶿ", "υ": "ᶹ", "ϕ": "ᵠ", "χ": "ᵡ"}

    if s in switcher:
        return switcher[s]
    return s


def subs(s):
    """ If existing, return the subscript character.

    Args:
        s (char): A character.

    Returns:
        char: A subscript character.
    """

    switcher = {"a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ", "k": "ₖ",
                "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ", "p": "ₚ", "r": "ᵣ",
                "s": "ₛ", "t": "ₜ", "u": "ᵤ", "v": "ᵥ", "x": "ₓ", "1": "₁",
                "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆", "7": "₇",
                "8": "₈", "9": "₉", "0": "₀", "β": "ᵦ", "γ": "ᵧ", "ρ": "ᵨ",
                "ϕ": "ᵩ", "χ": "ᵪ", "+": "₊", "-": "₋", "=": "₌"}

    if s in switcher:
        return switcher[s]
    return s
