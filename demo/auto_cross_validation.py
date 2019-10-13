import sklearn.metrics
import sklearn.datasets
import sklearn.model_selection
import autosklearn.classification


def main():
    x, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, random_state=1)

    # 采样策略：K-折交叉验证（5, cv）
    auto_ml = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=120,
        per_run_time_limit=30,
        tmp_folder='/tmp/autosklearn_cv_example_tmp',
        output_folder='/tmp/autosklearn_cv_example_out',
        delete_tmp_folder_after_terminate=False,
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds': 5},
    )

    # fit() changes the data in place, but refit needs the original data. We therefore copy the data.
    # In practice, one should reload the data
    auto_ml.fit(x_train.copy(), y_train.copy(), dataset_name='breast_cancer')

    # During fit(), models are fit on individual cross-validation folds. Necessary when using cross-validation.
    # To use all available data, we call refit() which trains all models in the final ensemble on the whole dataset.
    auto_ml.refit(x_train.copy(), y_train.copy())

    print(auto_ml.show_models())

    predictions = auto_ml.predict(x_test)
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))
    pass


if __name__ == '__main__':
    main()
