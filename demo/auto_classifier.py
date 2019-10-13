import sklearn.model_selection
import autosklearn.classification


def demo():
    # 加载数据
    x, y = sklearn.datasets.load_digits(return_X_y=True)

    # 划分数据集
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, random_state=1)

    # 创建自动化分类器示例
    auto_ml = autosklearn.classification.AutoSklearnClassifier()

    # 自动化训练
    auto_ml.fit(x_train, y_train)

    # 预测
    y_hat = auto_ml.predict(x_test)

    # 计算准确率
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))
    pass


if __name__ == '__main__':
    demo()
    pass
