# Random_NNs

Using neural networks with a random amount of layers in combination with social choice to perform classifications.

The implemented voting rules are: 
* Plurality
* STV
* Condorcet
* Borda
* Copeland
* Dictatorship

## 1. Install requirements
```
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Run model
```
python main.py
```
This will prompt the user to give an amount of networks for the ensemble. The model will then begin to train each network and run the election afterwards.
