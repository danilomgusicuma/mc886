def cutNumber(cut):
    if cut == "Fair":
        return 0
    elif cut == "Good":
        return 2.5
    elif cut == "Very Good":
        return 5
    elif cut == "Premium":
        return 7.5
    elif cut == "Ideal":
        return 10

def colorNumber(color):
    if color == "J":
        return 0
    elif color == "I":
        return 5/3
    elif color == "H":
        return 10/3
    elif color == "G":
        return 5
    elif color == "F":
        return 20/3
    elif color == "E":
        return 25/3
    elif color == "D":
        return 10

def clarityNumber(clarity):
    if clarity == "I1":
        return 0
    elif clarity == "SI2":
        return 10/7
    elif clarity == "SI1":
        return 20/7
    elif clarity == "VS2":
        return 30/7
    elif clarity == "VS1":
        return 40/7
    elif clarity == "VVS2":
        return 50/7
    elif clarity == "VVS1":
        return 60/7
    elif clarity == "IF":
        return 10
