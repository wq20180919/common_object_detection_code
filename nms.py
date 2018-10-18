# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def nms(bbox, overlap, top_k=200):
    '''
    计算nms
    :param bbox: array()
    :param overlap: float
    :param top_k: 先选取bbox中得分最高的top_k个
    :return: nms之后的bbox: array()
    '''
    if len(bbox) == 0:
        pick = []
    else:
        x1 = bbox[:, 0]
        y1 = bbox[:, 1]
        x2 = bbox[:, 2]
        y2 = bbox[:, 3]
        score = bbox[:, 4]
        area = (x2 - x1) * (y2 - y1)
        ind = np.argsort(score)
        ind = ind[-top_k:]
        pick = []
        while len(ind) > 0:
            last = len(ind) - 1
            i = ind[last]
            pick.append(i)
            suppress = [last]
            for pos in range(last):
                j = ind[pos]
                xx1 = np.maximum(x1[i], x1[j])
                yy1 = np.maximum(y1[i], y1[j])
                xx2 = np.minimum(x2[i], y2[j])
                yy2 = np.minimum(y2[i], y2[j])
                w = xx2 - xx1
                h = yy2 - yy1
                if w > 0 and h > 0:
                    ol = w * h / area[i]  # 这里有的是使用iou的计算方式，有的不是使用iou，而是：并集/最大得分的面积，这里使用的就不是iou的计算方式
                    if ol > overlap:
                        suppress.append(pos)
            ind = np.delete(ind, suppress)
    return bbox[pick, :]


def soft_nms(bbox, overlap, threshold=0.3, method='linear', top_k=200):
    '''
    计算soft-nms的方法
    :param bbox: array()
    :param overlap: float,和nms的含义一样
    :param threshold: float, 将权重处理之后，将小于此阈值的踢掉
    :param method: string, soft-nms的线性和高斯方法
    :param top_k: 先选取bbox中得分最高的top_k个
    :return: nms之后的bbox: array()
    '''
    if len(bbox) == 0:
        pick = []
    else:
        x1 = bbox[:, 0]
        y1 = bbox[:, 1]
        x2 = bbox[:, 2]
        y2 = bbox[:, 3]
        score = bbox[:, 4]
        area = (x2 - x1) * (y2 - y1)
        ind = np.argsort(score)
        ind = ind[-top_k:]
        pick = []
        while len(ind) > 0:
            last = len(ind) - 1
            i = ind[last]
            pick.append(i)
            suppress = [last]
            for pos in range(last):
                j = ind[pos]
                xx1 = np.maximum(x1[i], x1[j])
                yy1 = np.maximum(y1[i], y1[j])
                xx2 = np.minimum(x2[i], y2[j])
                yy2 = np.minimum(y2[i], y2[j])
                w = xx2 - xx1
                h = yy2 - yy1
                if w > 0 and h > 0:
                    ov = w * h / area[i]
                    if method == 'linear':
                        if ov > overlap:
                            weight = 1 - ov
                        else:
                            weight = ov
                    elif method == 'gaussian':
                        weight = np.exp(-(ov * ov) / 0.5)  # 0.5表示的是高斯函数中的sigma，默认取0.5
                    else:
                        if ov > overlap:
                            weight = 0
                        else:
                            weight = 1
                    score[j] = weight * score[j]
                if score[j] < threshold:
                    suppress.append(pos)
            ind = np.delete(ind, suppress)
    return bbox[pick, :]


if __name__ == '__main__':
    bbox = np.array([[200, 200, 400, 400, 0.99],
                     [220, 220, 420, 420, 0.9],
                     [100, 100, 150, 150, 0.82],
                     [200, 240, 400, 440, 0.5],
                     [150, 250, 300, 400, 0.88]])
    overlap = 0.7
    # pick=nms(bbox,overlap)
    pick = soft_nms(bbox, overlap)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, aspect='equal')
    bbox = bbox / 500
    pick = pick / 500
    for i in range(bbox.shape[0]):
        rect1 = patches.Rectangle(xy=(bbox[i, 0], bbox[i, 1]), width=bbox[i, 2] - bbox[i, 0],
                                  height=bbox[i, 3] - bbox[i, 1], edgecolor='r', linewidth=1, fill=False)
        plt.text(x=bbox[i, 0], y=bbox[i, 1], s='%s' % i, color='b')
        ax.add_patch(rect1)
    for i in range(pick.shape[0]):
        rect2 = patches.Rectangle(xy=(pick[i, 0], pick[i, 1]), width=pick[i, 2] - pick[i, 0],
                                  height=pick[i, 3] - pick[i, 1], edgecolor='b', linewidth=3, fill=False)
        plt.text(x=pick[i, 0], y=pick[i, 1], s='%s' % i, color='r', fontsize=20)
        ax.add_patch(rect2)
    plt.show()
