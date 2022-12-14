## Configs
We modify the bert and roberta to allow us change the bitwith of operator more freely, and make the Module can automatically detect the operators that requires hijacking.
- to be more specific, we replace matmul with QMatmul, add with QAdd and some f.softmax() with nn.softmax()