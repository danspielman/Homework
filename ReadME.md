# How to run the language detector code

### loadData.py
This file is the language classification code for LSTM model. 

The main entry of this code is main() function
```python
for i in range(5):
    # filename = ['data/dut.txt', 'data/itn.txt', 'data/ltn1.txt', 'data/yps.txt', 'data/ger.txt', 'data/por.txt', 'data/eng.txt', 'data/frn.txt']
    filename = ['data/eng.txt', 'data/dut.txt', 'data/itn.txt', 'data/ltn1.txt']
    main(filename, False)
```

You can change the filename for the language you want to detect. 

Notice: the second parameter is whether to draw a roc curve 
for result or not. And roc curve only works for binary classification, so don't try to draw a curve within multi-class classification.


### n_gram.py
This file is the language classification code using n_gram algorithm.


Then main entry of this code is main() function.
```python
main('data/eng.txt', 'data/frn.txt')
```

This main function accepted two parameters, both are file path for input data. You can change the filename for whatever you want 
to classify. This code only fitted for two languages.
