import tensorflow as tf


class TFIncrementalPCA:
    
    def __init__(self, n_components=None):
        """ Initializes a Tensorflow-based incremental PCA
        object. Basically copy-pasted from the scikit-learn
        implementation (da real mvp).
        
        Parameters
        ----------
        n_components : int
            Number of components to keep
        """
        self.n_components = n_components
    
    def fit(self, X):
        """ Only partial_fit is implemented. """
        raise NotImplementedError
    
    def partial_fit(self, X):
        """ Iterative (batch-wise) PCA fit.
        
        Parameters
        ----------
        X : tf.Tensor
            Tensor of shape `n_samples` by `n_features`
        """

        first_pass = not hasattr(self, "components_")
        if first_pass:
            self.components_ = None
            self.n_samples_seen_ = 0
            self.mean_ = tf.Variable(0.0)
            self.var_ = tf.Variable(0.0)

        n_samples, n_features = X.shape
         
        if self.n_components is None:
            if self.components_ is None:
                self.n_components_ = min(n_samples, n_features)
            else:
                self.n_components_ = self.components_.shape[0]
        else:
            self.n_components_ = self.n_components

        n_samples = tf.cast(n_samples, tf.float32)
    
        col_mean, col_var, n_total_samples = _incremental_mean_and_var(
            X, self.mean_, self.var_,
            tf.repeat(self.n_samples_seen_, tf.shape(X)[1]),
            n_samples
        )
        n_total_samples = n_total_samples[0]
        
        if self.n_samples_seen_ == 0:
            X = X - col_mean
        else:
            col_batch_mean = tf.reduce_mean(X, axis=0)
            X = X - col_batch_mean

            mean_correction = tf.sqrt(
                (self.n_samples_seen_ / n_total_samples) * tf.cast(n_samples, tf.float32)
            ) * (self.mean_ - col_batch_mean)
            mean_correction = tf.expand_dims(mean_correction, 0)
            to_add = tf.math.multiply(tf.reshape(self.singular_values, (1, -1)), self.components_)
            X = tf.concat(
                (   tf.transpose(to_add),
                    X,
                    mean_correction,
                ), axis=0
            )
        
        s, u, v = tf.linalg.svd(X, full_matrices=False)

        max_abs_cols = tf.argmax(tf.abs(u), axis=0)
        signs = tf.sign(tf.linalg.diag_part(tf.gather(u, max_abs_cols, axis=0)))
        u = u * signs
        v = v * tf.expand_dims(signs, 0)
        
        explained_variance = s ** 2 / (n_total_samples - 1)
        explained_variance_ratio = s ** 2 / tf.reduce_sum(col_var * n_total_samples)
        
        self.explained_variance_ = explained_variance[:self.n_components_]
        self.explained_variance_ratio_ = explained_variance_ratio[:self.n_components_]

        self.n_samples_seen_ = n_total_samples
        self.components_ = v[:, :self.n_components_]
        self.singular_values = s[:self.n_components_]
        self.mean_ = col_mean
        self.var_ = col_var
        
        if self.n_components_ < n_features:
            self.noise_variance_ = tf.reduce_mean(explained_variance[self.n_components_ :])
        else:
            self.noise_variance_ = 0.0
        
    def transform(self, X):
        """ Applies the PCA transform.
        
        Parameters
        ----------
        X : tf.Tensor
            Tensor of shape `n_samples` by `n_features`
        
        Returns
        -------
        X_transformed : tf.Tensor
            Tensor of shape `n_samples` by `n_comp`        
        """
        X_transformed = tf.matmul(X - self.mean_, self.components_)
        return X_transformed
     

def _incremental_mean_and_var(X, last_mean, last_variance, last_sample_count, new_sample_count):
    """Helper function, reproduced line-by-line from the scikit-learn
    implementation, except for the stuff that takes care of NaNs. """ 
    last_sample_count = tf.cast(last_sample_count, tf.float32)
    last_sum = last_mean * last_sample_count
    new_sum = tf.reduce_sum(X, axis=0)
    
    updated_sample_count = last_sample_count + new_sample_count
    updated_mean = (last_sum + new_sum) / updated_sample_count
    
    T = new_sum / new_sample_count
    temp = X - T
    correction = tf.reduce_sum(temp, axis=0)
    temp = temp ** 2
    new_unnormalized_variance = tf.reduce_sum(temp, axis=0)
    
    new_unnormalized_variance = new_unnormalized_variance - correction ** 2 / new_sample_count
    last_unnormalized_variance = last_variance * last_sample_count
    last_over_new_count = last_sample_count / new_sample_count  # div by 0

    updated_unnormalized_variance = (
                last_unnormalized_variance
                + new_unnormalized_variance
                + last_over_new_count
                / updated_sample_count
                * (last_sum / last_over_new_count - new_sum) ** 2
            )
    zeros = last_sample_count == 0
    updated_unnormalized_variance = tf.tensor_scatter_nd_update(
        updated_unnormalized_variance, tf.where(zeros), 
        tf.boolean_mask(new_unnormalized_variance, zeros)
    )
    updated_variance = updated_unnormalized_variance / updated_sample_count

    return updated_mean, updated_variance, updated_sample_count
