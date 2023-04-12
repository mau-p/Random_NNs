import dataset
import ensemble
import social_choice
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '' # Disable GPU, CPU appears to be faster with this model

def main():
    data = dataset.data_pipeline()

    num_of_networks = input('How many networks do you want to train? (Default: 49) \n')
    
    # Pass data and number of models to ensemble
    model_set = ensemble.Ensemble(data, int(num_of_networks))
    voting_rules = [social_choice.plurality, social_choice.dictatorship, social_choice.STV]
    accuracies = model_set.get_accuracy(data, voting_rules)

    for name, accuracy in accuracies.items():
        print(f'testing_accuracy {name}: {accuracy}')

if __name__ == '__main__':
    main()
