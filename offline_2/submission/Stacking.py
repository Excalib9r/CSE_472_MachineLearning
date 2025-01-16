import Bagging as bagging
from LR import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

class Stacking:
    def __init__ (self, base_estimator = LogisticRegression, meta_estimator= LogisticRegression):
        self.base_estimators = base_estimator
        self.meta_estimator = meta_estimator
        self.meta_model = None
        self.base_models = None

    def train(self, X, y, X_meta, y_meta):
        self.base_models = bagging.Bagging(self.base_estimators, n_estimators=9)
        self.base_models.train(X, y)
        meta_samples = self.base_models.predict(X_meta, False).T #shape = (n_samples, n_estimators)
        meta_samples = np.hstack([ X_meta , meta_samples])
        print(meta_samples.shape)
        self.meta_model = self.meta_estimator()
        self.meta_model.train(meta_samples, y_meta)

    def predict(self, X):
        meta_samples = self.base_models.predict(X, False).T
        print(meta_samples.shape)
        meta_samples = np.hstack([X, meta_samples])
        print(meta_samples.shape)
        prediction = self.meta_model.predict(meta_samples)
        return prediction