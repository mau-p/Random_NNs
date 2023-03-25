import dataset
import ensemble

def main():
    data = dataset.data_pipeline()
    model_set = ensemble.Ensemble(data, 13)

if __name__ == '__main__':
    main()
