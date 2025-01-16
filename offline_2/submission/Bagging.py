import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from LR import LogisticRegression

class Bagging:
    def __init__ (self, base_estimator = LogisticRegression, n_estimators=10, random_state=42, sample_size=0.9):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.sample_size = sample_size
        self.estimators_ = []
    
    def train(self, X, y):
        self.estimators_ = []
        np.random.seed(self.random_state)  # Setting the random seed ensures that the random processes in your code produce the same results every time you run it. This is crucial for debugging and verifying results.
        for _ in range(self.n_estimators):
            estimator = self.base_estimator()
            X_resampled, y_resampled = resample(X, y, n_samples=int(self.sample_size * X.shape[0]))
            estimator.train(X_resampled, y_resampled)
            self.estimators_.append(estimator)
    
    def predict(self, X, final=True):
        # predictions = np.array([estimator.predict(X, final) for estimator in self.estimators_]) #shape = (n_estimators, n_samples)
        predictions = []
        for estimator in self.estimators_:
            predictions.append(estimator.predict(X, final))
        predictions = np.array(predictions)
        return predictions
    
    def predict_majority(self, X, final=True):
        predictions = self.predict(X, final)
        majority_votes = []
        for i in range(predictions.shape[1]):
            # Extract the predictions for the i-th feature (or instance)
            feature_predictions = predictions[:, i].astype(int)
            majority_vote = np.argmax(np.bincount(feature_predictions))
            majority_votes.append(majority_vote)
        return np.array(majority_votes)
        # return np.array([np.argmax(np.bincount(predictions[:, i])) for i in range(predictions.shape[1])])
    