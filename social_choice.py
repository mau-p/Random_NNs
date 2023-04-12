import numpy as np
import random
from collections import Counter
from operator import itemgetter

def plurality(profile):
    count = np.zeros(5)

    for preference in profile:
        count[preference[0]] += 1
    
    return np.argmax(count)


def dictatorship(profile):
    pref = random.choice(profile)
    return pref[0]


def STV(profile, removed=[]):
    if len(profile[0]) == 1:
        return profile[0][0]
    
    primary_choice = [pref[0] for pref in profile]
    count = Counter(primary_choice)
    to_remove = min(count.items(), key=itemgetter(1))[0]
    profile = [[alt for alt in pref if alt != to_remove] for pref in profile]

    return STV(profile, removed)