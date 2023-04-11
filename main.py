import dataset
import ensemble
import social_choice
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Disable GPU, CPU appears to be faster with this model

def main():
    data = dataset.data_pipeline()

    num_of_networks = input('How many networks do you want to train? (Default: 49) \n')
    
    # Pass data and number of models to ensemble
    model_set = ensemble.Ensemble(data, int(num_of_networks))
    testing_accuracy_plurality = model_set.get_accuracy(data, social_choice.plurality)
    testing_accuracy_stv = model_set.get_accuracy(data, social_choice.STV)

    print(f'testing_accuracy plurality: {testing_accuracy_plurality}')
    print(f'testing_accuracy STV: {testing_accuracy_stv}')

if __name__ == '__main__':
    main()
