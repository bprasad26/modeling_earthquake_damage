import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import learning_curve
import plotly.graph_objects as go
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
from sklearn import tree


def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])


class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k

    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self

    def transform(self, X):
        return X[:, self.feature_indices_]


# plot precision, recall vs threshold


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=precisions[:-1],
            name="Precision",
            mode="lines",
            line=dict(color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=recalls[:-1],
            name="Recall",
            mode="lines",
            line=dict(color="green"),
        )
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(range=[-50000, 50000])
    fig.update_layout(
        title="Precision and recall versus the decision threshold",
        xaxis_title="Threshold",
    )
    fig.show()


def plot_precision_vs_recall(precisions, recalls):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=recalls, y=precisions, mode="lines", line=dict(color="green"))
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(range=[0, 1])
    fig.update_layout(
        title="Precision vs Recall", xaxis_title="Recall",
    )
    fig.show()


def plot_roc_curve(fpr, trp, label=None):

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color="green"), name=label)
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="blue"),
            name="random classifier",
        )
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(range=[0, 1])
    if label == None:
        fig.update_layout(
            title="The ROC Curve",
            xaxis_title="False Positive Rate (Fall-Out)",
            yaxis_title="True Positive Rate (Recall)",
            showlegend=False,
        )
    else:
        fig.update_layout(
            title="The ROC Curve",
            xaxis_title="False Positive Rate (Fall-Out)",
            yaxis_title="True Positive Rate (Recall)",
        )

    fig.show()


def compare_roc_curve(fpr_clf1, trp_clf1, label1, fpr_clf2, tpr_clf2, label2):

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr_clf1, y=trp_clf1, mode="lines", line=dict(color="green"), name=label1
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fpr_clf2, y=tpr_clf2, mode="lines", line=dict(color="red"), name=label2
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="blue"),
            name="random classifier",
        )
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(range=[0, 1])
    fig.update_layout(
        title="The ROC Curve",
        xaxis_title="False Positive Rate (Fall-Out)",
        yaxis_title="True Positive Rate (Recall)",
    )

    fig.show()


from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]


def plot_learning_curves(estimator, X, y, cv):

    train_sizes, train_scores, test_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=cv,
        n_jobs=-1,
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=train_mean,
            name="Training accuracy",
            mode="lines",
            line=dict(color="blue"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=test_mean,
            name="Validation accuracy",
            mode="lines",
            line=dict(color="green"),
        )
    )

    fig.update_layout(
        title="Learning Curves",
        xaxis_title="Number of training examples",
        yaxis_title="Accuracy",
    )

    fig.show()


def plot_validation_curves(estimator, X, y, param_name, param_range, cv):
    train_scores, test_scores = validation_curve(
        estimator=estimator,
        X=X,
        y=y,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=param_range,
            y=train_mean,
            name="Training Accuracy",
            mode="lines",
            line=dict(color="Blue"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=param_range,
            y=test_mean,
            name="Validation Accuracy",
            mode="lines",
            line=dict(color="Green"),
        )
    )

    fig.update_layout(
        title="Validation Curves", xaxis_title=param_name, yaxis_title="Accuracy"
    )

    fig.show()


def plot_decision_tree(classifier, feature_names, class_names):
    """This function plots decision tree.
    classifier: The name of the classifier,
    feature_names: Feature names
    class_name: class names
    """
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    tree.plot_tree(
        classifier,
        feature_names=feature_names,
        class_names=class_names,
        rounded=True,
        filled=True,
    )
    fig.show()


def plot_silhouetter_scores(k_range, silhouette_scores):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=k_range,
            y=silhouette_scores,
            mode="lines+markers",
            marker=dict(color="green"),
        )
    )
    fig.update_layout(xaxis_title="K", yaxis_title="Silhouette Score")
    fig.show()


def num_to_cat_list(df, num_col_list, n_unique_val):
    """This function takes a pandas dataframe, a list of numerical columns
    and create a list of columns that needs to be converted to categorical column if
    it is less than or equal to n_unique_val."""

    # columns that needs to converted
    cols_to_convert = []
    for col in num_col_list:
        unique_val = df[col].nunique()
        print(col, unique_val)
        if unique_val <= n_unique_val:
            cols_to_convert.append(col)
    return cols_to_convert
