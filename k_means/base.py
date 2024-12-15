import numpy as np
import matplotlib.pyplot as plt

# 生成一些随机数据（二维数据）
np.random.seed(0)
# 300个 点，[300,2]
X = np.vstack((np.random.randn(100, 2) + [2, 2],
               np.random.randn(100, 2) + [-2, -2],
               np.random.randn(100, 2) + [2, -2]))


## 参数配置
plt_max_row = 3
plt_max_column = 4
k_means_max_iters = plt_max_row * plt_max_column - 1
fig, axs = plt.subplots(plt_max_row, plt_max_column, figsize=(15, 5))

# 实现K - means算法
def kmeans(X, K):
    m, n = X.shape
    # 随机选取K个中心点
    centroids = X[np.random.choice(m, K, replace=False)]
    for index in range(k_means_max_iters):
        print("迭代.... " + str(index))
        # 计算300个点到K个中心点到距离
        distances = np.sqrt(((X[:, None] - centroids) ** 2).sum(axis=-1))
        # 找个每个数据点距离最新的中心点
        labels = np.argmin(distances, axis=-1)
        # 计算新的中心点
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])
        if np.all(centroids == new_centroids):
            break
        # 阶段性可视化聚类结果
        axs_iter = axs[int(index / plt_max_column)][int(index % plt_max_column)]
        axs_iter.scatter(X[:, 0], X[:, 1], c = labels)
        axs_iter.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='o')
        axs_iter.set_title("iter" + str(index))
        centroids = new_centroids
    return centroids, labels

# 运行K - means算法，假设K = 3
centroids, labels = kmeans(X, 4)

# 可视化聚类结果
axs_result = axs[plt_max_row - 1][plt_max_column - 1]
axs_result.scatter(X[:, 0], X[:, 1], c=labels)
axs_result.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
axs_result.set_title("result")

plt.tight_layout()
plt.show()