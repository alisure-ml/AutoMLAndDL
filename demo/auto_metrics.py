import numpy as np
import sklearn.metrics
import sklearn.datasets
import autosklearn.metrics
import sklearn.model_selection
import autosklearn.classification


def accuracy(solution, prediction):
    # custom function defining accuracy
    return np.mean(solution == prediction)


def error(solution, prediction):
    # custom function defining error
    return np.mean(solution != prediction)


def accuracy_wk(solution, prediction, dummy):
    # custom function defining accuracy and accepting an additional argument
    assert dummy is None
    return np.mean(solution == prediction)


def error_wk(solution, prediction, dummy):
    # custom function defining error and accepting an additional argument
    assert dummy is None
    return np.mean(solution != prediction)


def main():
    print("#################################################################################################")
    # Print a list of available metrics
    print("Available CLASSIFICATION metrics autosklearn.metrics.*:")
    print("\t*" + "\n\t*".join(autosklearn.metrics.CLASSIFICATION_METRICS))
    print("Available REGRESSION autosklearn.metrics.*:")
    print("\t*" + "\n\t*".join(autosklearn.metrics.REGRESSION_METRICS))

    print("#################################################################################################")
    x, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, random_state=1)

    print("#################################################################################################")
    # First example: Use predefined accuracy metric
    print("Use predefined accuracy metric")
    auto_classifier = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=60,
                                                                       per_run_time_limit=30, seed=1)
    auto_classifier.fit(x_train, y_train, metric=autosklearn.metrics.accuracy)

    predictions = auto_classifier.predict(x_test)
    print("Accuracy score {:g} using {:s}".format(sklearn.metrics.accuracy_score(y_test, predictions),
                                                  auto_classifier._automl[0]._metric.name))

    print("#################################################################################################")
    # Second example: Use own accuracy metric
    print("Use self defined accuracy metric")
    accuracy_scorer = autosklearn.metrics.make_scorer(name="accu", score_func=accuracy, optimum=1,
                                                      greater_is_better=True, needs_proba=False, needs_threshold=False)
    auto_classifier = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=60,
                                                                       per_run_time_limit=30, seed=1)
    auto_classifier.fit(x_train, y_train, metric=accuracy_scorer)

    predictions = auto_classifier.predict(x_test)
    print("Accuracy score {:g} using {:s}".format(sklearn.metrics.accuracy_score(y_test, predictions),
                                                  auto_classifier._automl[0]._metric.name))

    print("#################################################################################################")
    print("Use self defined error metric")
    error_rate = autosklearn.metrics.make_scorer(name='error', score_func=error, optimum=0,
                                                 greater_is_better=False, needs_proba=False, needs_threshold=False)
    auto_classifier = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=60,
                                                                       per_run_time_limit=30, seed=1)
    auto_classifier.fit(x_train, y_train, metric=error_rate)

    auto_classifier.predictions = auto_classifier.predict(x_test)
    print("Error rate {:g} using {:s}".format(error_rate(y_test, predictions),
                                              auto_classifier._automl[0]._metric.name))

    print("#################################################################################################")
    # Third example: Use own accuracy metric with additional argument
    print("Use self defined accuracy with additional argument")
    accuracy_scorer = autosklearn.metrics.make_scorer(
        name="accu_add", score_func=accuracy_wk, optimum=1,
        greater_is_better=True, needs_proba=False, needs_threshold=False, dummy=None)
    auto_classifier = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=60,
                                                                       per_run_time_limit=30, seed=1)
    auto_classifier.fit(x_train, y_train, metric=accuracy_scorer)

    predictions = auto_classifier.predict(x_test)
    print("Accuracy score {:g} using {:s}".format(sklearn.metrics.accuracy_score(y_test, predictions),
                                                  auto_classifier._automl[0]._metric.name))

    print("#################################################################################################")
    print("Use self defined error with additional argument")
    error_rate = autosklearn.metrics.make_scorer(
        name="error_add", score_func=error_wk, optimum=0,
        greater_is_better=True, needs_proba=False, needs_threshold=False, dummy=None)
    auto_classifier = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=60,
                                                                       per_run_time_limit=30, seed=1)
    auto_classifier.fit(x_train, y_train, metric=error_rate)

    predictions = auto_classifier.predict(x_test)
    print("Error rate {:g} using {:s}".format(error_rate(y_test, predictions),
                                              auto_classifier._automl[0]._metric.name))
    pass


if __name__ == "__main__":
    main()
