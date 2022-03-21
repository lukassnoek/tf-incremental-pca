# tf-incremental-pca
A Tensorflow (2.0) implementation of incremental PCA. Based on the [implementation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html) of *scikit-learn*, but about 15-20 times faster when dealing with many features (i.e., *K* > 10,000). 

## Example usage:
An example using random data:

```python
import tensorflow as tf
from tqdm import tqdm
from tensorflow.data import Dataset
from tfpca import TFIncrementalPCA

N = 2048   # total nr of samples
K = 50000  # number of features
batch_size = 512
n_comp = 500 # number of components to keep

dset = Dataset.range(N)
dset = dset.map(lambda i: tf.random.normal((1, K)))
dset = dset.batch(batch_size)

pca_tf = TFIncrementalPCA(n_components=n_comp)

for i, X in enumerate(tqdm(dset)):
    # If your data is not a 2D N (obs) x K (features) array,
    # make sure to reshape it: `X = tf.reshape(X, (batch_size, -1))`
    X = tf.squeeze(X)
    pca_tf.partial_fit(X)    

# Show proportion of explained variance using `n_comp` components 
print(pca_tf.explained_variance_ratio_.numpy().sum())
```

## Limitations
This implementation is largely a line-by-line reimplementation of *scikit-learn*'s implementation, apart from the following elements:

* It does not check for or deals with `NaN` values in your data;
* It has no option to whiten your data;
* Doesn't do any data validation like *scikit-learn*;
* The components are stored as a tensor of shape `n_features` x `n_components` (instead of the other way around, like *scikit-learn*).

Also, the amount of data that can be fit with this implementation depends on the amount of VRAM of your GPU. I got it to work on a Quadro RTX 8000 with a batch size of 512 and about 800,000 features.

Given the results of my (admittedly limited) tests, this implementation is identical to the implementation by *scikit-learn*, up to a sign flip of the components (which is [not a problem](https://stackoverflow.com/questions/21115669/scikit-learn-pca-matrix-transformation-produces-pc-estimates-with-flipped-signs)).

## Attribution
If you end up using this implementation in your research, please cite the *scikit-learn* package accordingly:

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. the *Journal of machine Learning research, 12*, 2825-2830.
