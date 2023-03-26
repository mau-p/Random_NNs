import dataset
import ensemble
import social_choice

def main():
    data = dataset.data_pipeline()
    model_set = ensemble.Ensemble(data, 13)
    testing_accuracy = model_set.get_accuracy(data, social_choice.plurality)
    print(f'testing_accuracy: {testing_accuracy}')

if __name__ == '__main__':
    main()
