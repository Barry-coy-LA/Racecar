import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree  # 使用 KDTree 加速最近邻查找
from utils import wrapAngle

EPS = 0.0001
MAX_ITER = 100
show_animation = False

if show_animation:
    fig = plt.figure()

def icp_matching(edges, scan, pose):
    if len(scan) < 5 or len(edges) < len(scan):
        return None
    
    # 删除重复的扫描点
    scan = np.unique(scan, axis=0)

    # 转置 edges 和 scan 以匹配算法的实现
    edges = edges.T
    scan = scan.T

    H = np.diag([1, 1, 1])  # 初始化齐次变换矩阵

    dError = np.inf
    preError = np.inf
    count = 0

    # 使用 KD-Tree 加速最近邻查找
    tree = KDTree(edges.T)

    while dError >= EPS:
        count += 1

        # 使用 KD-Tree 进行最近邻搜索
        indexes, total_error = nearest_neighbor_association_kdtree(tree, edges, scan)
        edges_matched = edges[:, indexes]

        if show_animation:
            plot_points(edges_matched, scan, fig)

        # 执行 RANSAC
        min_error = np.float('inf')
        best_Rt = None
        best_Tt = None
        for _ in range(30):  # 增加 RANSAC 迭代次数，提高匹配可靠性
            sample = np.random.choice(scan.shape[1], 5, replace=False)
            
            Rt, Tt = svd_motion_estimation(edges_matched[:, sample], scan[:, sample])
            temp_points = (Rt @ scan) + Tt[:, np.newaxis]
            _, error = nearest_neighbor_association_kdtree(tree, edges, temp_points)
            if error < min_error:
                min_error = error
                best_Rt = Rt
                best_Tt = Tt

        # 更新当前扫描点以进行迭代优化
        scan = (best_Rt @ scan) + best_Tt[:, np.newaxis]

        dError = preError - total_error
        preError = total_error
        H = update_homogeneous_matrix(H, best_Rt, best_Tt)

        if MAX_ITER <= count:
            break

    R = np.array(H[0:-1, 0:-1])
    T = np.array(H[0:-1, -1])

    if abs(T[0]) > 5 or abs(T[1]) > 5:
        return None
    else:
        x = pose[0] + T[0]
        y = pose[1] + T[1]
        orientation = wrapAngle(pose[2] + np.arctan2(R[1][0], R[0][0]))

        return np.array((x, y, orientation))

def update_homogeneous_matrix(Hin, R, T):
    H = np.zeros((3, 3))
    H[0:2, 0:2] = R
    H[0:2, 2] = T
    H[2, 2] = 1.0
    return Hin @ H

def nearest_neighbor_association_kdtree(tree, prev_points, curr_points):
    # 使用 KD-Tree 查找最近邻
    distances, indexes = tree.query(curr_points.T, k=1)
    error = distances.ravel()
    return indexes.ravel(), np.sum(error)

def svd_motion_estimation(previous_points, current_points):
    pm = np.mean(previous_points, axis=1)
    cm = np.mean(current_points, axis=1)

    p_shift = previous_points - pm[:, np.newaxis]
    c_shift = current_points - cm[:, np.newaxis]

    W = c_shift @ p_shift.T
    u, s, vh = np.linalg.svd(W)

    R = (u @ vh).T
    t = pm - (R @ cm)

    return R, t

def plot_points(previous_points, current_points, figure):
    # 用于按 ESC 键停止模拟
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])
    if previous_points.shape[0] == 3:
        plt.clf()
        axes = figure.add_subplot(111, projection='3d')
        axes.scatter(previous_points[0, :], previous_points[1, :],
                    previous_points[2, :], c="r", marker=".")
        axes.scatter(current_points[0, :], current_points[1, :],
                    current_points[2, :], c="b", marker=".")
        axes.scatter(0.0, 0.0, 0.0, c="r", marker="x")
        figure.canvas.draw()
    else:
        plt.cla()
        plt.plot(previous_points[0, :], previous_points[1, :], ".r", markersize=1)
        plt.plot(current_points[0, :], current_points[1, :], ".b", markersize=1)
        plt.plot(0.0, 0.0, "xr")
        plt.axis("equal")

    plt.pause(0.01)
    plt.draw()
    plt.clf()
