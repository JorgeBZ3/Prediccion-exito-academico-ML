import numpy as np


class MultinomialLogisticRegressor:
    """
    Multinomial Logistic Regression classifier for multiclass problems.

    Adapted from the binary LogisticRegressor class to support K > 2 classes
    using the softmax activation and categorical cross-entropy loss.

    Key differences with respect to the binary version:
    - self.weights shape: (n, K)  instead of (n,)
    - self.bias shape:   (K,)    instead of scalar
    - predict_proba returns (m, K) probability matrix via softmax (not sigmoid)
    - log_likelihood uses categorical cross-entropy over one-hot encoded labels
    - predict returns the argmax class index (mapped back to original labels)
    - Regularization methods operate on the (n, K) weight matrix — same formulas,
      same code structure as before, the math generalises naturally.
    """

    def __init__(self):
        """
        Initializes the Multinomial Logistic Regressor.

        Attributes:
        - weights (np.ndarray): Weight matrix of shape (n_features, n_classes).
                                Initialized during fit().
        - bias (np.ndarray):    Bias vector of shape (n_classes,).
                                Initialized during fit().
        - classes_ (np.ndarray): Sorted array of unique class labels seen during fit().
                                 Used to map integer indices back to original labels.
        """
        self.weights = None
        self.bias = None
        self.classes_ = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X,
        y,
        learning_rate=0.01,
        num_iterations=1000,
        penalty=None,
        l1_ratio=0.5,
        C=1.0,
        verbose=False,
        print_every=100,
    ):
        """
        Fits the multinomial logistic regression model using gradient descent.

        The forward pass computes softmax probabilities over K classes.
        The backward pass derives gradients from the categorical cross-entropy loss.
        Regularization terms (L1, L2, ElasticNet) are applied to dw exactly as in
        the binary version — the formulas are identical; only the shape of the
        weight matrix changes from (n,) to (n, K).

        Parameters:
        - X (np.ndarray):        Input features, shape (m, n).
        - y (np.ndarray):        True class labels, shape (m,). Can be any hashable type
                                 (strings, ints, etc.).
        - learning_rate (float): Gradient descent step size. Default 0.01.
        - num_iterations (int):  Number of gradient descent steps. Default 1000.
        - penalty (str):         Regularization type: None, 'lasso', 'ridge', 'elasticnet'.
        - l1_ratio (float):      Mix ratio for ElasticNet (0 = pure Ridge, 1 = pure Lasso).
        - C (float):             Inverse regularization strength (smaller → stronger reg.).
        - verbose (bool):        Whether to print the loss periodically.
        - print_every (int):     Frequency (in iterations) of loss logging.

        Updates:
        - self.weights:  Trained weight matrix (n, K).
        - self.bias:     Trained bias vector (K,).
        - self.classes_: Unique class labels in sorted order.
        """
        m, n = X.shape

        # Store unique classes and encode y as integer indices 0..K-1
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        y_idx = self._encode_labels(y)          # shape (m,)  integer indices
        Y_onehot = self._onehot(y_idx, K)       # shape (m, K)

        # Initialise parameters — same logic as binary, extended to K classes
        self.weights = np.zeros((n, K))          # (n, K)  — one weight vector per class
        self.bias = np.zeros(K)                  # (K,)

        # Gradient descent loop
        for i in range(num_iterations):

            # Forward pass: probabilities via softmax, shape (m, K)
            y_hat = self.predict_proba(X)

            # Loss: categorical cross-entropy
            loss = self.log_likelihood(Y_onehot, y_hat)

            if verbose and i % print_every == 0:
                print(f"Iteration {i}: Loss {loss:.6f}")

            # Gradients — same structure as binary case, now matrices
            # dw shape (n, K),  db shape (K,)
            dw = (1 / m) * np.dot(X.T, (y_hat - Y_onehot))
            db = (1 / m) * np.sum(y_hat - Y_onehot, axis=0)

            # Apply regularization to dw (db is not regularized, as is standard)
            if penalty == "lasso":
                dw = self.lasso_regularization(dw, m, C)
            elif penalty == "ridge":
                dw = self.ridge_regularization(dw, m, C)
            elif penalty == "elasticnet":
                dw = self.elasticnet_regularization(dw, m, C, l1_ratio)

            # Parameter update — identical structure to binary version
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

    def predict_proba(self, X):
        """
        Returns the probability of each class for every sample.

        Replaces sigmoid with softmax, which generalises sigmoid to K classes.
        For K=2 both are mathematically equivalent.

        Parameters:
        - X (np.ndarray): Input features, shape (m, n).

        Returns:
        - np.ndarray: Probability matrix of shape (m, K).
                      Each row sums to 1.
        """
        # Logits: shape (m, K)
        z = np.dot(X, self.weights) + self.bias
        return self.softmax(z)

    def predict(self, X):
        """
        Predicts the most probable class label for each sample.

        Replaces the threshold-based binary decision with argmax over K classes.

        Parameters:
        - X (np.ndarray): Input features, shape (m, n).

        Returns:
        - np.ndarray: Predicted class labels, shape (m,), in the original label space
                      (e.g. strings like 'abandono', 'graduado', 'matriculado').
        """
        probs = self.predict_proba(X)                 # (m, K)
        class_indices = np.argmax(probs, axis=1)      # (m,)
        return self.classes_[class_indices]           # map back to original labels

    # ------------------------------------------------------------------
    # Regularization methods
    # (identical formulas to binary version; work on matrices naturally)
    # ------------------------------------------------------------------

    def lasso_regularization(self, dw, m, C):
        """
        Applies L1 (Lasso) regularization to the weight gradient.

        Formula (same as binary):
            dw += (C / m) * sign(W)

        Parameters:
        - dw (np.ndarray): Gradient of the loss w.r.t. weights, shape (n, K).
        - m (int):         Number of training samples.
        - C (float):       Inverse regularization strength.

        Returns:
        - np.ndarray: Regularized gradient, shape (n, K).
        """
        lasso_gradient = (C / m) * np.sign(self.weights)
        return dw + lasso_gradient

    def ridge_regularization(self, dw, m, C):
        """
        Applies L2 (Ridge) regularization to the weight gradient.

        Formula (same as binary):
            dw += (C / m) * W

        Parameters:
        - dw (np.ndarray): Gradient of the loss w.r.t. weights, shape (n, K).
        - m (int):         Number of training samples.
        - C (float):       Inverse regularization strength.

        Returns:
        - np.ndarray: Regularized gradient, shape (n, K).
        """
        ridge_gradient = (C / m) * self.weights
        return dw + ridge_gradient

    def elasticnet_regularization(self, dw, m, C, l1_ratio):
        """
        Applies ElasticNet regularization (combination of L1 and L2).

        Formula (same as binary):
            dw += l1_ratio * (C/m)*sign(W) + (1 - l1_ratio) * (C/m)*W

        Parameters:
        - dw (np.ndarray): Gradient of the loss w.r.t. weights, shape (n, K).
        - m (int):         Number of training samples.
        - C (float):       Inverse regularization strength.
        - l1_ratio (float): Mix between L1 (1.0) and L2 (0.0).

        Returns:
        - np.ndarray: Regularized gradient, shape (n, K).
        """
        lasso_part = (C / m) * np.sign(self.weights)
        ridge_part = (C / m) * self.weights
        elasticnet_gradient = l1_ratio * lasso_part + (1 - l1_ratio) * ridge_part
        return dw + elasticnet_gradient

    # ------------------------------------------------------------------
    # Loss and activation functions
    # ------------------------------------------------------------------

    @staticmethod
    def log_likelihood(Y_onehot, y_hat):
        """
        Categorical cross-entropy loss — the multiclass generalisation of the
        binary log-likelihood used in the original class.

        Binary formula:
            L = -(1/m) * sum(y*log(y_hat) + (1-y)*log(1-y_hat))

        Multiclass generalisation:
            L = -(1/m) * sum_i sum_k [ Y_onehot[i,k] * log(y_hat[i,k]) ]

        When K=2 both formulas are mathematically identical.

        Parameters:
        - Y_onehot (np.ndarray): One-hot encoded true labels, shape (m, K).
        - y_hat (np.ndarray):    Predicted probabilities from softmax, shape (m, K).

        Returns:
        - float: Scalar loss value.
        """
        m = Y_onehot.shape[0]
        # Clip probabilities to avoid log(0)
        y_hat_clipped = np.clip(y_hat, 1e-15, 1 - 1e-15)
        loss = -(1 / m) * np.sum(Y_onehot * np.log(y_hat_clipped))
        return loss

    @staticmethod
    def softmax(z):
        """
        Softmax activation — generalisation of sigmoid to K classes.

        For numerical stability, the maximum logit is subtracted before
        exponentiation (the result is mathematically identical).

        When K=2, softmax and sigmoid produce equivalent predictions.

        Parameters:
        - z (np.ndarray): Logit matrix, shape (m, K).

        Returns:
        - np.ndarray: Probability matrix, shape (m, K). Each row sums to 1.
        """
        # Subtract row-wise max for numerical stability
        z_stable = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _encode_labels(self, y):
        """
        Converts original class labels to integer indices 0..K-1.

        Parameters:
        - y (np.ndarray): Original labels, shape (m,).

        Returns:
        - np.ndarray: Integer indices, shape (m,).
        """
        return np.searchsorted(self.classes_, y)

    @staticmethod
    def _onehot(y_idx, K):
        """
        Converts integer class indices to one-hot encoded matrix.

        Parameters:
        - y_idx (np.ndarray): Integer class indices, shape (m,).
        - K (int):            Number of classes.

        Returns:
        - np.ndarray: One-hot matrix, shape (m, K).
        """
        m = len(y_idx)
        Y = np.zeros((m, K))
        Y[np.arange(m), y_idx] = 1
        return Y
