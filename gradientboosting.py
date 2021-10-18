# For this basic implementation, we only need these modules
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# Load the well-known Breast Cancer dataset
# Split into train and test sets
x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=23)

# Gradient Boosting initialization
# The base learner is a decision tree as default
# The number of estimators is 5
# The depth for each deciion tree is 2
# The learning rate for each estimator in the sequence is 1
gradientBoosting = GradientBoostingClassifier(n_estimators=5, learning_rate=1, max_depth=2, random_state=23)

# Train!
gradientBoosting.fit(x_train, y_train)

# Evaluation
print(f"Train score: {gradientBoosting.score(x_train, y_train)}")
print(f"Test score: {gradientBoosting.score(x_test, y_test)}")
