import numpy as np

def plurality(profile):
    print(f"Profile: {profile}")
    count = np.zeros(2)

    for preference in profile:
        count[preference[0]] += 1
    
    return np.argmax(count)


def STV(profile):
    print(f"Profile: {profile}")

    if len(profile[0]) == 1:
        return profile[0][0]

    count = np.zeros(2)
    for preference in profile:
        count[preference[0]] += 1
    
    to_remove = np.argmin(count)
    profile = [[alt for alt in pref if alt != to_remove] for pref in profile]

    return STV(profile)