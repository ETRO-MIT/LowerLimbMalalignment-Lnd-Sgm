# Compute the JLO
def compute_jlo(mldfa_value: float, mmpta_value: float):

    return mmpta_value + mldfa_value


# Compute the aHKA
def compute_ahka(mldfa_value: float, mmpta_value: float):

    return mmpta_value - mldfa_value

# Compute the CPAK value according to the aHKA and the JLO
def compute_cpak(ahka_value: float, jlo_value: float):
    if ahka_value < -2:
        if jlo_value < 177:
            cpak = 1
            return cpak
        elif 177 < jlo_value < 183:
            cpak = 4
            return cpak
        elif jlo_value > 183:
            cpak = 7
            return cpak
    elif -2 < ahka_value < 2:
        if jlo_value < 177:
            cpak = 2
            return cpak
        elif 177 < jlo_value < 183:
            cpak = 5
            return cpak
        elif jlo_value > 183:
            cpak = 8
            return cpak
    elif ahka_value > 2:
        if jlo_value < 177:
            cpak = 3
            return cpak
        elif 177 < jlo_value < 183:
            cpak = 6
            return cpak
        elif jlo_value > 183:
            cpak = 9
            return cpak
