import numpy as np

def plurality(profile):
    count = np.zeros(4)

    for preference in profile:
        count[preference[0]] += 1
    
    return np.argmax(count)


def STV(profile):
    if not isinstance(profile, np.ndarray):
        profile = np.array(profile)

    if profile.shape[1] == 1:
        return profile[0][0]

    counts = np.unique(profile[:, 0], return_counts=True)
    to_remove = counts[0][np.argmin(counts[1])]

    new_profile = []
    for pref in profile:
        new_pref = np.delete(pref, np.where(pref == to_remove))
        new_profile.append(new_pref)
    
    return STV(new_profile)