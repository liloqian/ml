import numpy as np
import matplotlib.pyplot as plt

# 测试数据
# 设置线性回归的系数
beta_0 = 2
beta_1 = 3
# 生成x值，这里生成100个从0到10的等间距数据点
train_x = np.linspace(0, 100, 50)
# 根据线性关系计算y值，并添加正态分布的噪声
noise = np.random.normal(0, 15, size = train_x.shape)
train_y = beta_0 + beta_1 * train_x + noise

# 标准化x
train_x_mean = train_x.mean()
train_x_std = train_x.std()
def stand(x):
        return (x - train_x_mean) / train_x_std
train_z = stand(train_x)

# 定义预测函数
theta0 = np.random.rand()
theta1 = np.random.rand()
def f(x):
    return theta0 + theta1 * x

# 目标函数，损失函数
def Error(x, y):
    return 0.5 * np.sum((y - f(x)) ** 2)

ETA = 1e-3
diff = 1
count = 0
error = Error(train_z, train_y)
while diff > 1e-2:
    tmp_theta0 = theta0 - ETA * np.sum((f(train_z) - train_y))
    tmp_theta1 = theta1 - ETA * np.sum((f(train_z) - train_y) * train_z)
    theta0 = tmp_theta0
    theta1 = tmp_theta1
    current_error = Error(train_z, train_y)
    diff = error - current_error
    error = current_error
    count += 1
    print("迭代 %d 次 theta0= %.2f theta1= %.2f 差值= %.2f" % (count, theta0, theta1, diff))


# show ui
plt.subplot(1, 2, 1)
plt.scatter(train_z, train_y)
plt.plot(train_z, f(train_z))

plt.subplot(1, 2, 2)
un_theta1= theta1 / train_x_std
un_theta0 = theta0 - theta1 * train_x_mean / train_x_std
plt.scatter(train_x, train_y)
plt.plot(train_x, un_theta0 + un_theta1 * train_x)

plt.show()

