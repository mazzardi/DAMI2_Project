import numpy as np
from sklearn.tree import DecisionTreeClassifier
import random
from scipy.stats import mode
#
from numpy.random import RandomState
#

def unbalanced_bagger(x, y, M=1, sampling_techniques=None, base_estimator=None, random_seed=None):
    if sampling_techniques is None:
        sampling_techniques = []
    if base_estimator is None:
        if random_seed is None:
            base_estimator = DecisionTreeClassifier()
        else:
            base_estimator = DecisionTreeClassifier(random_state=random_seed)

    #måste vara i if else eftersom random_seed kan vara NONE
    rand = RandomState(random_seed)
    #
    base_learners = []
    for i in range(M):
        #rnd = np.random.choice(x.shape[0], size=x.shape[0])
        rnd = rand.choice(x.shape[0], size=x.shape[0])
        x_bootstrap = x.iloc[rnd, :]
        y_bootstrap = y.iloc[rnd]

        #technique = random.randint(0, len(sampling_techniques) - 1)
        technique = rand.randint(0, len(sampling_techniques) - 1)
        x_sample, y_sample = sampling_techniques[technique].fit_sample(x_bootstrap, y_bootstrap)

        base_estimator.fit(x_sample, y_sample)
        base_learners.append(base_estimator)

    def predict(x):
        predictions = np.zeros([x.shape[0], len(base_learners)])
        for i in range(len(base_learners)):
            predictions[:, i] = base_learners[i].predict(x)
        return mode(predictions, axis=1)[0]

    def predict_proba(x):
        n_classes = len(base_learners[0].classes_)
        proba = np.zeros([x.shape[0], n_classes])
        for i, base_learner in enumerate(base_learners):
            proba += base_learner.predict_proba(x)
        return proba / len(base_learners)

    return predict, predict_proba
