import dataset
import ensemble
import social_choice

def main():
    data = dataset.data_pipeline()

    # Pass data and number of models to ensemble
    model_set = ensemble.Ensemble(data, 15)
    testing_accuracy = model_set.get_accuracy(data, social_choice.plurality)
    print(f'testing_accuracy: {testing_accuracy}')

if __name__ == '__main__':
    main()
