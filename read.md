# How to run project 1 jupyter notebook

This code only has one main entry: main(iters=10000, print_steps=1000). First parameter is iterations time, the second parameter 
is every print_steps time plot the generated images and loss of generator and discriminator.

The data is import from:

```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
```

So there is no data inside data folder. The code will download the mnist dataset online for the first running.

You can adjust two parameters of main function to see improvement after various iterations.
