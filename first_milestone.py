import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load the dataset
with open('train_data.pkl', 'rb') as f:
    data = pickle.load(f)


with open('test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)
    
    
# Access images and labels
images = data['images']
labels = data['labels']

test_images = test_data['images']

# A simple Gaussian RBF kernel
def rbf_kernel(x1, x2, gamma=0.1):
    diff = x1 - x2
    return np.exp(-gamma * np.dot(diff, diff.T))

# Kernelized SVM using dual form (simplified)
class KernelizedSVM:
    def __init__(self, C=1.0, kernel=rbf_kernel, gamma=0.1):
        self.C = C
        self.kernel = lambda x1, x2: kernel(x1, x2, gamma)

    def fit(self, X, y, epochs=100, lr=0.001):
        self.X = X
        self.y = np.array(y) * 2 - 1  # Convert labels from {0, 1} to {-1, 1}
        self.alpha = np.zeros(X.shape[0])

        # Compute kernel matrix
        self.K = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                self.K[i, j] = self.kernel(X[i], X[j])

        # Optimize alpha using gradient descent
        for epoch in range(epochs):
            gradient = 1 - (self.alpha * self.y) @ self.K
            self.alpha += lr * gradient
            self.alpha = np.clip(self.alpha, 0, self.C)

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            pred = 0
            for i in range(self.X.shape[0]):
                pred += self.alpha[i] * self.y[i] * self.kernel(self.X[i], x)
            y_pred.append(np.sign(pred))
        return np.array(y_pred)


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        unique_classes = np.unique(y)

        # If all labels are the same or max depth is reached, return a leaf node
        if len(unique_classes) == 1 or (self.max_depth and depth >= self.max_depth):
            return unique_classes[0]

        best_split = self._best_split(X, y)
        left_X, right_X, left_y, right_y = self._split(X, y, best_split)

        left_node = self._build_tree(left_X, left_y, depth + 1)
        right_node = self._build_tree(right_X, right_y, depth + 1)

        return {
            'feature': best_split[0],
            'value': best_split[1],
            'left': left_node,
            'right': right_node
        }

    def _best_split(self, X, y):
        best_gini = float('inf')
        best_split = None

        for feature_index in range(X.shape[1]):
            feature_values = np.unique(X[:, feature_index])

            for value in feature_values:
                left_X, right_X, left_y, right_y = self._split(X, y, (feature_index, value))

                gini = self._gini_index(left_y, right_y)
                if gini < best_gini:
                    best_gini = gini
                    best_split = (feature_index, value)

        return best_split

    def _split(self, X, y, split):
        feature_index, value = split
        left_mask = X[:, feature_index] <= value
        right_mask = ~left_mask

        left_X = X[left_mask]
        right_X = X[right_mask]
        left_y = y[left_mask]
        right_y = y[right_mask]

        return left_X, right_X, left_y, right_y

    def _gini_index(self, left_y, right_y):
        total = len(left_y) + len(right_y)
        left_size = len(left_y) / total
        right_size = len(right_y) / total

        def gini(y):
            unique_classes = np.unique(y)
            gini = 1
            for c in unique_classes:
                prob = np.sum(y == c) / len(y)
                gini -= prob ** 2
            return gini

        return left_size * gini(left_y) + right_size * gini(right_y)

    def predict(self, X):
        return np.array([self._predict_sample(sample, self.tree) for sample in X])

    def _predict_sample(self, sample, node):
        if isinstance(node, dict):
            if sample[node['feature']] <= node['value']:
                return self._predict_sample(sample, node['left'])
            else:
                return self._predict_sample(sample, node['right'])
        else:
            return node
        

class OptimizedDecisionTreeClassifier:
    def __init__(
        self,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree_ = None

    def _gini(self, y):
        """Compute Gini Impurity."""
        m = len(y)
        if m == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / m
        return 1 - np.sum(probabilities ** 2)

    def _entropy(self, y):
        """Compute Entropy."""
        m = len(y)
        if m == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / m
        return -np.sum(probabilities * np.log2(probabilities + 1e-9))

    def _criterion_function(self, y):
        """Select the splitting criterion."""
        if self.criterion == "gini":
            return self._gini(y)
        elif self.criterion == "entropy":
            return self._entropy(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

    def _best_split(self, X, y):
        """Find the best split for a node."""
        best_gain = -float("inf")
        best_feature = None
        best_threshold = None
        current_criterion = self._criterion_function(y)

        n_samples, n_features = X.shape
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_idx = X[:, feature_idx] <= threshold
                right_idx = ~left_idx

                left_y, right_y = y[left_idx], y[right_idx]
                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue

                # Weighted average of the criterion for child nodes
                left_weight = len(left_y) / len(y)
                right_weight = len(right_y) / len(y)
                gain = current_criterion - (
                    left_weight * self._criterion_function(left_y)
                    + right_weight * self._criterion_function(right_y)
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth):
        """Recursively build the decision tree."""
        n_samples, n_features = X.shape
        num_classes = len(np.unique(y))

        # Stopping conditions
        if depth == self.max_depth or num_classes == 1 or n_samples < self.min_samples_split:
            leaf_value = self._majority_class(y)
            return {"type": "leaf", "value": leaf_value}

        # Find the best split
        feature_idx, threshold = self._best_split(X, y)
        if feature_idx is None:
            leaf_value = self._majority_class(y)
            return {"type": "leaf", "value": leaf_value}

        left_idx = X[:, feature_idx] <= threshold
        right_idx = ~left_idx

        left_subtree = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_subtree = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return {
            "type": "node",
            "feature": feature_idx,
            "threshold": threshold,
            "left": left_subtree,
            "right": right_subtree,
        }

    def _majority_class(self, y):
        """Return the majority class in a leaf node."""
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def fit(self, X, y):
        """Fit the decision tree."""
        X, y = np.array(X), np.array(y)
        self.tree_ = self._build_tree(X, y, depth=0)

    def _predict_sample(self, node, sample):
        """Traverse the tree for a single sample."""
        if node["type"] == "leaf":
            return node["value"]
        feature = node["feature"]
        threshold = node["threshold"]
        if sample[feature] <= threshold:
            return self._predict_sample(node["left"], sample)
        else:
            return self._predict_sample(node["right"], sample)

    def predict(self, X):
        """Predict for a batch of samples."""
        check_tree = self.tree_ is not None
        if not check_tree:
            raise RuntimeError("The tree has not been fitted yet.")
        X = np.array(X)
        return np.array([self._predict_sample(self.tree_, sample) for sample in X])
            

class OptimizedRandomForest:
    def __init__(self, n_trees=10, max_depth=None, min_samples_split=2, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples, n_features = X.shape
        max_features = self.max_features or n_features

        for _ in range(self.n_trees):
            # Bootstrap sample
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_X = X[bootstrap_indices]
            bootstrap_y = y[bootstrap_indices]

            # Random feature selection
            feature_indices = np.random.choice(n_features, max_features, replace=False)
            tree = OptimizedDecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(bootstrap_X[:, feature_indices], bootstrap_y)
            self.trees.append((tree, feature_indices))

    def predict(self, X):
        tree_predictions = np.array([
            tree.predict(X[:, feature_indices]) for tree, feature_indices in self.trees
        ])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=tree_predictions)


# Split the data
train_images, val_images, train_labels, val_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

# Ensure the data is in numpy array format
train_images = np.array(train_images).reshape(len(train_images), -1)  # Flatten images
val_images = np.array(val_images).reshape(len(val_images), -1)
train_labels = np.array(train_labels)
val_labels = np.array(val_labels)

subset_size = 10000
train_images = train_images[:subset_size]
train_labels = train_labels[:subset_size]

# Train and evaluate the optimized Decision Tree
opt_tree = OptimizedDecisionTreeClassifier(max_depth=10, min_samples_split=5)
opt_tree.fit(train_images, train_labels)
opt_tree_preds = opt_tree.predict(val_images)
opt_tree_accuracy = np.mean(opt_tree_preds == val_labels)
print(f"Optimized Decision Tree Validation Accuracy: {opt_tree_accuracy * 100:.2f}%")

# change test data in order to predict test labels 
test_images = np.array(test_images).reshape(len(test_images), -1)
test_preds = opt_tree.predict(test_images)

data_to_csv = {
    'label': test_preds
}

pd.DataFrame(data_to_csv).to_csv('pred_labels_first_milestone.csv')

# Train and evaluate the optimized Random Forest
opt_forest = OptimizedRandomForest(n_trees=20, max_depth=10, min_samples_split=5, max_features=int(train_images.shape[1] ** 0.5))
opt_forest.fit(train_images, train_labels)
opt_forest_preds = opt_forest.predict(val_images)
opt_forest_accuracy = np.mean(opt_forest_preds == val_labels)
print(f"Optimized Random Forest Validation Accuracy: {opt_forest_accuracy * 100:.2f}%")
