import numpy as np
import random
from collections import Counter
from operator import itemgetter


def copeland(profile):
    net_pref = np.zeros((len(profile[0]), len(profile[0])))

    for preference in profile:
        for i, alt in enumerate(preference):
            for j, alt2 in enumerate(preference):
                if i < j:
                    net_pref[alt, alt2] += 1

    count = np.zeros(len(profile[0]))
    for i in range(len(count)):
        for j in range(len(count)):
            count[i] = net_pref[i, j] - net_pref[j, i]
    return np.argmax(count)


def condorcet(profile):
    count = np.zeros((len(profile[0]), len(profile[0])))

    for preference in profile:
        for i, alt in enumerate(preference):
            for j, alt2 in enumerate(preference):
                if i < j:
                    count[alt, alt2] += 1

    for i in range(len(count)):
        for j in range(len(count)):
            if count[i, j] > count[j, i]:
                count[j, i] = 0
            elif count[i, j] < count[j, i]:
                count[i, j] = 0

    count = np.sum(count, axis=1)
    return np.argmax(count)


def borda(profile):
    count = np.zeros(len(profile[0]))

    for preference in profile:
        for i, alt in enumerate(preference):
            count[alt] += len(preference) - (1 + i)

    return np.argmax(count)


def plurality(profile):
    count = np.zeros(len(profile[0]))

    for preference in profile:
        count[preference[0]] += 1

    return np.argmax(count)


def dictatorship(profile):
    pref = random.choice(profile)
    return pref[0]


def STV(profile, alternatives=[], removed=[]):
    if len(alternatives) == 1:
        return alternatives[0]

    if not alternatives:
        alternatives = np.unique(profile)

    count = Counter({alt: 0 for alt in alternatives})
    primary_choice = [pref[0] for pref in profile]
    count.update(primary_choice)

    lowest_plur_score = min(count.items(), key=itemgetter(1))[0]
    to_remove = [alt for alt in count if count[alt] == count[lowest_plur_score]]

    profile = [[alt for alt in pref if alt not in to_remove] for pref in profile]
    removed = [alt for alt in alternatives if alt in to_remove or alt in removed]
    alternatives = [alt for alt in alternatives if alt not in to_remove]

    return STV(profile, alternatives, removed)