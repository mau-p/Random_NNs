
import os
import pandas as pd
from collections import Counter

def count_multiplicity(profile):
    counts = Counter()
    for preference in profile:
        counts.update({tuple(preference): 1})
    return counts


def profiles_to_csv(profiles):
  if not os.path.isdir('profiles/'):
      os.mkdir('profiles/')

  for index, row in profiles.iterrows():
      ground_truth = row['label']
      with open(f'profiles/profile_{index}.csv', 'w') as f:
          f.write(f'ground_truth: {ground_truth}\n')
      pref_multi = {p:m for p, m in sorted(row['multiplicity'].items(), key=lambda item: item[1], reverse=True)}
      df = pd.DataFrame.from_dict(pref_multi.items())
      df.columns = ['preference', 'multiplicity']
      df.to_csv(f'profiles/profile_{index}.csv', index=False, mode='a')


profiles = pd.read_hdf('profiles.h5', key='df')
results = pd.read_csv('results.csv', index_col=False)
profiles['multiplicity'] = profiles.apply(lambda x: count_multiplicity(x['profile']), axis=1)
profiles_to_csv(profiles)

print(results.info())
print(results.head())
print(results.describe())

print(profiles.info())
print(profiles.head())
print(profiles.describe())

prof_wrong = profiles.iloc[0]
prof_right = profiles.iloc[2]

wrong = {k: v for k, v in sorted(prof_wrong['multiplicity'].items(), key=lambda item: item[1], reverse=True)}
right = {k: v for k, v in sorted(prof_right['multiplicity'].items(), key=lambda item: item[1], reverse=True)}

dictatorship_winners = results.loc[results['dictatorship'] == results['label']]
plurality_winners = results.loc[results['plurality'] == results['label']]
borda_winners = results.loc[results['borda'] == results['label']]
STV_winners = results.loc[results['STV'] == results['label']]
condorcet_winners = results.loc[results['condorcet'] == results['label']]
copeland_winners = results.loc[results['copeland'] == results['label']]

print('Dictatorship winners: ', len(dictatorship_winners), ' accuracy: ', len(dictatorship_winners)/len(results))
print('Plurality winners: ', len(plurality_winners), ' accuracy: ', len(plurality_winners)/len(results))
print('Borda winners: ', len(borda_winners), ' accuracy: ', len(borda_winners)/len(results))
print('STV winners: ', len(STV_winners), ' accuracy: ', len(STV_winners)/len(results))
print('Condorcet winners: ', len(condorcet_winners), ' accuracy: ', len(condorcet_winners)/len(results))
print('Copeland winners: ', len(copeland_winners), ' accuracy: ', len(copeland_winners)/len(results))
