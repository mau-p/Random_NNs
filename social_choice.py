import numpy as np

def plurality(profile):
    count = [0] * 4

    for preference in profile:
        count[preference[0]] += 1

    return np.argmax(count)
