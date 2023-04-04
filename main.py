import dataset
import ensemble
import social_choice

def main():
    data = dataset.data_pipeline()

    # Pass data and number of models to ensemble
    model_set = ensemble.Ensemble(data, 5)
    testing_accuracy_plurality = model_set.get_accuracy(data, social_choice.plurality)
    testing_accuracy_stv = model_set.get_accuracy(data, social_choice.STV)

    print(f'testing_accuracy plurality: {testing_accuracy_plurality}')
    print(f'testing_accuracy STV: {testing_accuracy_stv}')

if __name__ == '__main__':
    main()
