# Random_NNs
Using randomly generated neural networks and voting rules to classify wine quality

Dataset can be found at: http://www3.dsi.uminho.pt/pcortez/wine/

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