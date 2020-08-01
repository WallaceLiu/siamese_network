# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import time

import matplotlib.pyplot as plt
import matplotlib

from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets.samples_generator import make_circles

DATA_TYPE = 'blobs'

# 数据样本数 N
N = 200

# 如果我们选择环状，聚类数只有2个(Number of clusters,if we choose circles,only 2 will be enough)
if (DATA_TYPE == 'circle'):
    K = 2
else:
    K = 4

# 最大迭代次数设置，如果没有说明(Maximum number of iterations, if the conditions are not met)
MAX_ITERS = 1000

# 开始时间
start = time.time()

centers = [(-2, -2), (-2, 1.5), (1.5, -2), (2, 1.5)]

if (DATA_TYPE == 'circle'):
    data, features = make_circles(n_samples=N, shuffle=True, noise=0.01, factor=0.4)
    # n_samples 数据的数目
    # shuffle 数据是否打乱(True/False)
    # noise 添加到圆形数据集上的随机噪声数据
    # factor 环形数据间的比例因子
else:
    data, features = make_blobs(n_samples=N, centers=centers, n_features=2, shuffle=False, random_state=42)
    # n_samples 数据的数目
    # centers 类的中心
    # n_features 特征数据的列的数目(维度)
    # shuffle 数据是否打乱(True/False)
    # random_state 随机种子

# 聚类中心点
fig, ax = plt.subplots()
ax.scatter(np.array(centers).transpose()[0], np.array(centers).transpose()[1], marker='o', s=200)
plt.show()

# 如果DATA_TYPE是blobs型，绘图
fig, ax = plt.subplots()
if (DATA_TYPE == 'blobs'):
    ax.scatter(np.asarray(centers).transpose()[0], np.asarray(centers).transpose()[1], marker='o', s=250, c='0.1')
    ax.scatter(np.asarray(data).transpose()[0], np.asarray(data).transpose()[1], marker='o', s=80, c=features,
               cmap=plt.cm.coolwarm)
plt.show()

# points 用来存放数据集的点的坐标
points = tf.Variable(data)  # 200*2
# cluster_assigments用来存放为每个数据元素分配的类的索引
cluster_assigments = tf.Variable(tf.zeros([N], dtype=tf.int64))
# centrroids 用于存放每个组质心的坐标,随机选择相应的聚类中心
centroids = tf.Variable(tf.slice(points.initialized_value(), [0, 0], [K, 2]))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(centroids)

rep_centriods = tf.reshape(tf.tile(centroids, [N, 1]), [N, K, 2])
# tf.tile(centroids,[N,b]) 的结果是：
#     [centroids ... centroids
#         .            .
#         .            .
#         .            .
#     centroids ... centroids]的 N*b 矩阵

rep_points = tf.reshape(tf.tile(points, [1, K]), [N, K, 2])  # 复制原数据拓展列数为原数据的K倍

# sum_squares计算每一个样本到每一个质心点之间在所有维度上的距离
sum_squares = tf.reduce_sum(tf.square(rep_points - rep_centriods),
                            reduction_indices=2)  # 减少索引，第三层求和，总量变为(200,4) . reduction_indices=0(列和)|reduction_indices=1(行和)

# 对所有维度求和，得到和最小的那个索引(这个索引就是每个点所属的新的类)
best_centroids = tf.argmin(sum_squares, 1)  # 数据为200
# did_assignents_change 用来判断是否有变更
did_assignents_change = tf.reduce_any(tf.not_equal(best_centroids, cluster_assigments))  # 布尔值，结果为True


# ------------1------------tf.not_equal
# t=tf.not_equal(a,b)#只要前后是不一样的，那就符合要求，为True,否则为False
# t=tf.not_equal([1,1],[0,0])
# print(t.eval(session=sess))

# t=tf.not_equal([0,1],[0,0])
# print(t.eval(session=sess))

# t=tf.not_equal([1,1],[0,1])
# print(t.eval(session=sess))
# ------------2------------tf.constant
# x = tf.constant([[True,  True], [False, False]])
#     tf.reduce_any(x)  # True
#     tf.reduce_any(x, 0)  # [True, True]
#     tf.reduce_any(x, 1)  # [True, False]

def bucket_mean(data, bucket_ids, num_buckets):
    total = tf.unsorted_segment_sum(data, bucket_ids, num_buckets)  # 统计每个类的数据总和
    count = tf.unsorted_segment_sum(tf.ones_like(data), bucket_ids, num_buckets)  # 统计每个类的数量
    #     关于 tf.ones_like 只要含有数字都变为1
    #     tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
    #     tf.ones_like(tensor)  # [[1, 1, 1], [1, 1, 1]]
    return total / count  # 返回每个类的均值


means = bucket_mean(points, best_centroids, K)  # 样本数据的均值 4*2
with tf.control_dependencies([did_assignents_change]):
    do_updates = tf.group(centroids.assign(means), cluster_assigments.assign(best_centroids))  # 更新聚类中心和分类索引号

changed = True
iters = 0
fig, ax = plt.subplots()
if (DATA_TYPE == 'blobs'):
    colourindexes = [2, 1, 4, 3]
else:
    colourindexes = [2, 1]
while changed and iters < MAX_ITERS:
    fig, ax = plt.subplots()
    iters += 1
    [changed, _] = sess.run([did_assignents_change, do_updates])
    [centers, assignments] = sess.run([centroids, cluster_assigments])
    ax.scatter(sess.run(points).transpose()[0], sess.run(points).transpose()[1], marker='o', s=200, c=assignments,
               cmap=plt.cm.coolwarm)
    ax.scatter(centers[:, 0], centers[:, 1], marker='^', s=550, c=colourindexes, cmap=plt.cm.plasma)
    ax.set_title('Iteration' + str(iters))
    plt.savefig('kmeans' + str(iters) + '.jpg')

ax.scatter(sess.run(points).transpose()[0], sess.run(points).transpose()[1], marker='o', s=200, c=assignments,
           cmap=plt.cm.coolwarm)
plt.show()

end = time.time()
print('Found in %.2f seconds' % (end - start), iters, 'Iterations')
print('Centroids:')
print(centers)
print('Cluster assignments:', assignments)
