import sklearn.metrics
import sklearn.datasets
import autosklearn.regression
import sklearn.model_selection


def main():
    # 加载数据
    x, y = sklearn.datasets.load_boston(return_X_y=True)

    # 用于特征工程中的数据预处理
    feature_types = (['numerical'] * 3) + ['categorical'] + (['numerical'] * 9)

    # 划分数据集
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, random_state=1)

    # 创建自动化分类器示例
    auto_ml = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=120, per_run_time_limit=30,
        tmp_folder='/tmp/autosklearn_regression_example_tmp',
        output_folder='/tmp/autosklearn_regression_example_out',
    )

    # 自动化训练
    auto_ml.fit(x_train, y_train, dataset_name='boston', feat_type=feature_types)

    print(auto_ml.show_models())

    # 预测
    predictions = auto_ml.predict(x_test)

    # 计算准确率
    print("R2 score:", sklearn.metrics.r2_score(y_test, predictions))
    pass


if __name__ == '__main__':
    main()
