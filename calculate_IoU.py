# coding:utf-8
import numpy as np


def cal_IOU(bbox1, bbox2):
    """
    calculate the IoU of two bbox
    :param bbox1: array([x1,y1,x2,y2])
    :param bbox2: array([x1,y1,x2,y2])
    :return: iou: float
    """
    x0, y0, width0, height0 = bbox1[0], bbox1[1], bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
    x1, y1, width1, height1 = bbox2[0], bbox2[1], bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
    min_x, max_x = np.minimum(x0, x1), np.maximum(bbox1[2], bbox2[2])
    width = width0 + width1 - (max_x - min_x)
    min_y, max_y = np.minimum(y0, y1), np.maximum(bbox1[3], bbox2[3])
    height = height0 + height1 - (max_y - min_y)
    if width <= 0 or height <= 0:  # 没有交集的两个边界框设置为0
        return 0
    inter_area = width * height
    area0 = width0 * height0
    area1 = width1 * height1
    iou = inter_area / (area0 + area1 - inter_area)  # 计算公式，iou=交集/（并集-交集）
    return iou


def convert_xywh_to_x1y1x2y2(center):
    """
    convert center box to bounding box
    :param center: array([center_x,center_y,width,height])
    :return: bbox: array([x1,y1,x2,y2])
    """
    x1y1 = center[:2] - 0.5 * center[2:]
    wh = center[:2] + 0.5 * center[2:]
    return np.concatenate((x1y1, wh))


if __name__ == '__main__':
    bbox1 = np.array([100, 100, 250, 250])
    bbox2 = np.array([100, 100, 250, 250])
    iou = cal_IOU(bbox1, bbox2)
    print(iou)
    center = np.array([100, 100, 80, 80])
    bbox3 = convert_xywh_to_x1y1x2y2(center)
    print(bbox3)
