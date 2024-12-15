from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def generate_data(num_samples):
    X = np.array(range(num_samples))
    temp_y = 3.65 * X + 10
    random_array = np.random.uniform(low=0, high=50, size=num_samples)
    y = temp_y + random_array
    return X, y


if __name__ == '__main__':
    args = Namespace(
        seed=1234,
        data_file="sample_data.csv",
        num_samples=100,
        train_size=0.75,
        test_size=0.25,
        num_epochs=100,
    )

    np.random.seed(args.seed)
    X, y = generate_data(args.num_samples)
    data = np.vstack([X, y]).T
    df = pd.DataFrame(data, columns=['X', 'y'])

    # 划分数据到训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        df["X"].values.reshape(-1, 1), df["y"], test_size=args.test_size,
        random_state=args.seed)
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    # 标准化训练集数据 (mean=0, std=1)
    X_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train.values.reshape(-1, 1))

    # 在训练集和测试集上进行标准化操作
    standardized_X_train = X_scaler.transform(X_train)
    standardized_y_train = y_scaler.transform(y_train.values.reshape(-1, 1)).ravel()
    standardized_X_test = X_scaler.transform(X_test)
    standardized_y_test = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

    # 检查
    print("mean:", np.mean(standardized_X_train, axis=0),
          np.mean(standardized_y_train, axis=0))  # mean 应该是 ~0
    print("std:", np.std(standardized_X_train, axis=0),
          np.std(standardized_y_train, axis=0))  # std 应该是 1

    # 初始化模型
    lm = SGDRegressor(loss='squared_epsilon_insensitive', penalty="l2", max_iter=args.num_epochs)

    # 训练
    lm.fit(X=standardized_X_train, y=standardized_y_train)

    # 预测 (还未标准化)
    pred_train = (lm.predict(standardized_X_train) * np.sqrt(y_scaler.var_)) + y_scaler.mean_
    pred_test = (lm.predict(standardized_X_test) * np.sqrt(y_scaler.var_)) + y_scaler.mean_

    train_mse = np.mean((y_train - pred_train) ** 2)
    test_mse = np.mean((y_test - pred_test) ** 2)

    # 图例大小
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Origin Point")
    plt.scatter(df["X"], df["y"], label="origin")
    plt.legend(loc='lower right')

    # 画出训练数据
    plt.subplot(1, 3, 2)
    plt.title("Train")
    plt.scatter(X_train, y_train, label="y_train")
    plt.plot(X_train, pred_train, color="red", linewidth=1, linestyle="-", label="lm")
    plt.legend(loc='lower right')

    # 画出测试数据
    plt.subplot(1, 3, 3)
    plt.title("Test")
    plt.scatter(X_test, y_test, label="y_test")
    plt.plot(X_test, pred_test, color="red", linewidth=1, linestyle="-", label="lm")
    plt.legend(loc='lower right')

    # 显示图例
    plt.show()