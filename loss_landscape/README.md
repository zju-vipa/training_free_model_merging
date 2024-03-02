# Loss Landscape Visualization

## Run

Click "Run All" in [plot.ipynb](plot.ipynb)

- The performance provided in paper are the average of 5 pairs of the prime and odd models. You can get them by the [train_mnist.py](train_mnist.py). We merged the i-th prime model with ((i+1)%5)-th odd model (because we found the i-th prime models i-th odd models are similar on weights due to the same random seed and training data).