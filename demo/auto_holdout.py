import sklearn.metrics
import sklearn.datasets
import sklearn.model_selection
import autosklearn.classification


def main():
    x, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, random_state=1)

    # 采样策略：holdout（0.67）
    auto_ml = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=120,
        per_run_time_limit=30,
        tmp_folder='/tmp/autosklearn_holdout_example_tmp',
        output_folder='/tmp/autosklearn_holdout_example_out',
        disable_evaluator_output=False,
        resampling_strategy='holdout',
        resampling_strategy_arguments={'train_size': 0.67}
    )
    auto_ml.fit(x_train, y_train, dataset_name='breast_cancer')

    # Print the final ensemble constructed by auto-sklearn.
    print(auto_ml.show_models())

    predictions = auto_ml.predict(x_test)
    # Print statistics about the auto-sklearn run such as number of iterations, number of models failed with a time out.
    print(auto_ml.sprint_statistics())
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))
    pass


if __name__ == '__main__':
    main()
