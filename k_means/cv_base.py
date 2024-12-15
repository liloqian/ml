import numpy as np
import matplotlib.pyplot as plt
from skimage import io


# 读取图像
image_path = "./pic/pic3.jpeg"
image = io.imread(image_path)

# 将图像数据转换为适合K-means算法处理的格式
pixels = image.reshape((-1, 3))
print("origin image " + str(image.shape))
print("origin image reshape" + str(pixels.shape))

# 设定聚类数量K（这里假设为3，可根据实际需求调整）
K = 2

## 参数配置
plt_max_row = 3
plt_max_column = 3
k_means_max_iters = 5
# k_means_max_iters = plt_max_row * plt_max_column - 2
fig, axs = plt.subplots(plt_max_row, plt_max_column, figsize=(8, 8))

# 初始化聚类中心
def initialize_centroids(pixels, K):
    np.random.seed(0)  # 设置随机种子，以便结果可复现
    indices = np.random.choice(pixels.shape[0], K, replace=False)
    return pixels[indices]


# 计算欧几里得距离
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


# 分配像素点到聚类
def assign_clusters(pixels, centroids):
    num_pixels = pixels.shape[0]
    clusters = np.zeros(num_pixels)
    for i in range(num_pixels):
        distances = [euclidean_distance(pixels[i], centroid) for centroid in centroids]
        clusters[i] = np.argmin(distances)
    return clusters


# 更新聚类中心
def update_centroids(pixels, clusters, K):
    centroids = np.zeros((K, 3))
    for k in range(K):
        cluster_pixels = pixels[clusters == k]
        centroids[k] = np.mean(cluster_pixels, axis=0)
    return centroids


# 执行K-means算法
def kmeans(pixels, K):
    centroids = initialize_centroids(pixels, K)
    for index in range(k_means_max_iters):
        print("iter " + str(index))
        clusters = assign_clusters(pixels, centroids)
        new_centroids = update_centroids(pixels, clusters, K)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
        iter_img = centroids[clusters.astype(np.uint8)].reshape(image.shape)
        axs_iter = axs[int(index / plt_max_column)][int(index % plt_max_column)]
        axs_iter.imshow(iter_img.astype(np.uint8))
        axs_iter.set_title("iter " + str(index))
    return centroids, clusters


# 执行K-means算法进行图像分割
centroids, clusters = kmeans(pixels, K)

# 根据聚类结果重建图像
segmented_image = centroids[clusters.astype(np.uint8)].reshape(image.shape)

# 可视化结果
axs[plt_max_column-1][plt_max_column-2].imshow(image)
axs[plt_max_column-1][plt_max_column-2].set_title("Original Image")
axs[plt_max_column-1][plt_max_column-1].imshow(segmented_image.astype(np.uint8))
axs[plt_max_column-1][plt_max_column-1].set_title("Segmented Image")
plt.tight_layout()
plt.show()