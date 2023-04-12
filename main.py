import dataset
import ensemble
import social_choice
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # disable GPU = '', enable GPU = '0'; from experimenting CPU appears to be faster with this model, thus recommend disabling GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '1'


def query_user(query, default='y', binary=True):
    skip = False
    ans = input(f"{query} {'[y/n]' if binary else ''} (Default: {default}) \n")
    if not ans:
        ans = default
    if binary:
        while ans.lower() not in ['y', 'n']:
            if skip:
                print(f"Please enter 'y' or 'n'")
            ans = input(f"{query} [y/n] (Default: {default}) \n")
            skip = True
    return ans


def main():
    BATCH_SIZE = 512
    stored_profiles = False
    if os.path.isfile('profiles.h5'):
        stored_profiles = True
    model_dir = 'models/'
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    pre_trained = [os.path.join(model_dir, model) for model in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, model))]
    pre_trained = [model for model in pre_trained if model.split('/')[-1].startswith('model_')]

    ans = 'n'
    if pre_trained:
        ans = query_user(f"{len(pre_trained)} models found, use these?")
    if 'y' in ans.lower():
        print(f"Using pre-trained models")
        model_set = ensemble.Ensemble(models_to_load=pre_trained, batch_size=BATCH_SIZE)
        data = None
    else:
        print(f"Not using pre-trained models")
        num_of_networks = query_user('How many networks do you want to train?', default=49, binary=False)
        data = dataset.data_pipeline()
        # Pass data and number of models to ensemble
        model_set = ensemble.Ensemble(data, int(num_of_networks), batch_size=BATCH_SIZE)

    voting_rules = [social_choice.dictatorship,
                    social_choice.plurality,
                    social_choice.STV,
                    social_choice.borda,
                    social_choice.condorcet,
                    social_choice.copeland]

    if stored_profiles and not data:
        ans = query_user('Profiles already exist, use these?')
        if 'y' in ans.lower():
            print(f"Using stored profiles")
            accuracies = model_set.get_accuracy(voting_rules)
        else:
            print(f"Not using stored profiles")
            stored_profiles = False
            data = dataset.data_pipeline()
    if data:
        accuracies = model_set.get_accuracy(voting_rules, data)
    elif not stored_profiles:
        print(f"No data or stored profiles found")

    for name, accuracy in accuracies.items():
        print(f'testing_accuracy {name}: {accuracy}')


if __name__ == '__main__':
    main()
