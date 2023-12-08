from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import VOC_ROOT, VOCAnnotationTransform, VOCDetection, BaseTransform
from data import VOC_CLASSES as labelmap
import torch.utils.data as data

from SMENet import build_SMENet

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='SMENet Detector Evaluation')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=False, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOC_ROOT,
                    help='Location of VOC root directory')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

# cuda
if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
# apply cpu
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# load data
annopath = os.path.join(args.voc_root, 'VOC2012', 'Annotations', '%s.xml')
imgpath = os.path.join(args.voc_root, 'VOC2012', 'JPEGImages', '%s.jpg')
imgsetpath = os.path.join(args.voc_root, 'VOC2012', 'ImageSets', 'Main', '{:s}.txt')
YEAR = '2012'
devkit_path = args.voc_root + 'VOC' + YEAR
dataset_mean = (86, 91, 82)
set_type = 'test'

# Defining timers
class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

# Analysis of VOC data set
def parse_rec(filename):  # filename：图片的xml注释文件路径
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects  # 一个列表，每个元素是一个字典类型，这个字典记录当前图片中包含的目标信息

# Return to the directory where the test results are saved
def get_output_dir(name, phase):
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir

# Get test data path
def get_voc_results_file_template(image_set, cls):
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(devkit_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path

# Write test results
def write_voc_results_file(all_boxes, dataset):
    for cls_ind, cls in enumerate(labelmap):  # 遍历每一个类别，将该类别的检测结果写入对应的文件中
        print('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template(set_type, cls)  #'/VOCNWPU/VOC2012/results/det_test_airplane.txt'
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):  # 遍历测试集的每一张图片
                dets = all_boxes[cls_ind+1][im_ind]  # 找到当前图片检测到当前类别的目标 [num, 5]
                if dets == []:
                    continue
                for k in range(dets.shape[0]):  # 遍历每一个检测结果
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index[1], dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))  # 写入的是图片名称id，预测分数，预测坐标

# Calculate AP for each category  计算每一个类别的ap
def do_python_eval(output_dir='output', use_07=True):  # output_dir：'SMENet/test'
    cachedir = os.path.join(devkit_path, 'annotations_cache')  # '/Users/wyting/Desktop/smenet_inld/VOCNWPU/VOC2012/annotations_cache'
    aps = []
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)  # output_dir：'SMENet/test'
    for i, cls in enumerate(labelmap):  # 遍历每一个类别
        filename = get_voc_results_file_template(set_type, cls)  # '/VOCNWPU/VOC2012/results/det_test_airplane.txt'
        imgsetpathss = '/Users/wyting/Desktop/smenet_inld/VOCNWPU/VOC2012/ImageSets/Main/test.txt'
        rec, prec, ap = voc_eval(filename, annopath, imgsetpathss, cls, cachedir, ovthresh=0.5,
                                 use_07_metric=use_07_metric)  # 计算当前类别的ap
        aps += [ap]  # 记录每一个类别的ap值
        print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('--------------------------------------------------------------')
    return np.mean(aps)

def voc_ap(rec, prec, use_07_metric=False):
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]

        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath, annopath, imagesetfile, classname, cachedir, ovthresh=0.5, use_07_metric=True):
    # detpath：该文件记录了测试集所有图片检测到当前类别的目标/VOCNWPU/VOC2012/results/det_test_airplane.txt'
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')  # '/VOCNWPU/VOC2012/annotations_cache/annots.pkl' 存放测试集中目标信息
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()  # 读取测试集每一张图片的id
    imagenames = [x.strip() for x in lines]  # 存放测试集每一张图片的id
    if not os.path.isfile(cachefile):
        recs = {}  # 有测试集数量个元素，字典中每一个元素是一个列表，列表长度是一张图片中包含的目标个数，列表中的每个元素是一个字典类型，这个字典记录当前图片中包含的目标的gt信息
        for i, imagename in enumerate(imagenames):  # 遍历每一张图片
            recs[imagename] = parse_rec(annopath % (imagename))  # parse_rec()：返回一个列表，每个元素是一个字典类型，这个字典记录当前图片中包含的目标的gt信息
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    class_recs = {}  # 有测试集数量个元素，每个元素是一个字典，记录了对应图片中类别为当前类别的目标gt信息
    npos = 0  # 记录当前图片当前类别的所有目标个数
    for imagename in imagenames:  # 遍历测试集中每一张图片
        R = [obj for obj in recs[imagename] if obj['name'] == classname]  # R用来记录每一张图片中类别为clssname的目标，每一个obj是一个字典类型，记录目标信息
        bbox = np.array([x['bbox'] for x in R])  # 记录当前图片当前类别的所有目标的坐标信息 [num_gt_cls, 4]
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)  # 记录当前图片当前类别的所有目标的难易程度[num_gt_cls]
        det = [False] * len(R)  # 先让每一张图片中类别为classname的目标都标记为未检测过[num_gt_cls]
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox, 'difficult': difficult, 'det': det}

    # read dets
    detfile = detpath.format(classname)  # /VOCNWPU/VOC2012/results/det_test_airplane.txt
    with open(detfile, 'r') as f:
        lines = f.readlines()  # 每一个元素是测试集中检测为当前类别的目标

    if any(lines) == 1:
        splitlines = [x.strip().split(' ') for x in lines]  # [num_all_predict_cls, 6]
        image_ids = [x[0] for x in splitlines]  # 测试集中检测为当前类别的所有目标所对应的图片id
        confidence = np.array([float(x[1]) for x in splitlines])  # 测试集中检测为当前类别的所有目标所对应的预测分数
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])  # 测试集中检测为当前类别的所有目标所对应的绝对坐标

        # sort by confidence
        sorted_ind = np.argsort(-confidence)  # 按照confidence排序后的idx
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]  # 测试集中检测为当前类别的所有目标经过预测分数排序后所对应的绝对坐标
        image_ids = [image_ids[x] for x in sorted_ind]  # 测试集中检测为当前类别的所有目标经过预测分数排序后所对应的图片id

        # go down dets and mark TPs and FPs
        nd = len(image_ids)  # 测试集中检测为当前类别的目标数
        tp = np.zeros(nd)  # 用来记录测试集中检测为当前类别的目标是tp还是fp
        fp = np.zeros(nd)
        for d in range(nd):  # 遍历检测为当前类别的每一个目标  class_recs有测试集数量个元素，每个元素是一个字典，记录了对应图片中类别为当前类别的目标gt信息
            R = class_recs[image_ids[d]]  # 记录了预测为当前类别的目标所对应图片中类别为当前类别的所有目标gt信息
            bb = BB[d, :].astype(float)  # 当前目标所对应的绝对坐标
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)  # 当前目标所对应图片中该类别的所有gt的坐标
            if BBGT.size > 0:  # 当前预测目标与对应图片中当前类别gt的IOU
                if (bb[2] - bb[0]) * (bb[3] - bb[1]) <= 32 * 32:
                    temp_bb = torch.from_numpy(bb).unsqueeze(0)
                    temp_BB = torch.from_numpy(BBGT)
                    overlaps = calc_nwd_tensor(temp_bb, temp_BB)
                else:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin, 0.)
                    ih = np.maximum(iymax - iymin, 0.)
                    inters = iw * ih
                    uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                           (BBGT[:, 2] - BBGT[:, 0]) *
                           (BBGT[:, 3] - BBGT[:, 1]) - inters)
                    overlaps = inters / uni
                ovmax = np.max(overlaps)  # 最大的iou值
                jmax = np.argmax(overlaps)  # 当前预测目标与对应图片中当前类别哪个gt的IOU最大

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)  # 查全率
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)  # 精度
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap


def calc_nwd_tensor(bboxes1, bboxes2, eps=1e-6, constant=150):

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    area1 = area1.unsqueeze(-1)  # [num_gt, 1]
    area2 = area2.unsqueeze(0)  # [1, num_anchors]

    temp1 = torch.sqrt(area1)
    temp2 = torch.sqrt(area2)
    constant = (temp1 + temp2 + eps) / 2  # [num_gt, num_anchors]
    eps = torch.tensor([eps])

    center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
    center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
    whs = center1[..., :2] - center2[..., :2]

    center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps

    w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
    h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
    w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
    h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

    wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4

    wassersteins = torch.sqrt(center_distance + wh_distance)
    normalized_wassersteins = torch.exp(-wassersteins/constant)

    return normalized_wassersteins


def test_net(net, dataset):
    num_images = len(dataset)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]  # 存放最终预测结果，对于每一个类有一个列表，列表长度为测试图片数量，记录每张图片检测到当前类的目标信息

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir('SMENet', set_type)  # SMENet/test
    det_file = os.path.join(output_dir, 'detections.pkl')  # SMENet/test/detections.pkl
    avg_det_time = 0
    for i in range(num_images):  # 检测每一张图片
        im, gt, h, w = dataset.pull_item(i)  # im:[3, 400, 400] gt:[n, 5] 5--->[xmin, ymin, xmax, ymax, cls_id] cls_id=0表示第一个类别

        x = Variable(im.unsqueeze(0))  # [1, 3, 400, 400]
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()  # 开始计算检测一张图片的时间
        ne = net(x)  # 网络最终的预测框(只要预测得分大于0.01都会保留)  [1, 11, 200, 5]  5--->[scores, xmin, ymin, xmax, ymax]
        detections = ne.data
        detect_time = _t['im_detect'].toc(average=False)
        avg_det_time += detect_time
        for j in range(1, detections.size(1)):  # 遍历每一个类别的检测结果
            dets = detections[0, j, :]   # each label: [200, 5]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()   # [200, 5]  dets[:, 0] is score
            dets = torch.masked_select(dets, mask).view(-1, 5)  # 找出最终的预测框[num_predict, 5]
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]  # [num_pre, 4]最终预测框的相对坐标转为绝对坐标
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()  # [num_pre]
            # [num_pre, 5] 5--->[xmin, ymin, xmax, ymax, scores] 绝对坐标
            cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])).astype(np.float32, copy=False)
            all_boxes[j][i] = cls_dets

        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_images, detect_time))
    print('Average Predict Time: {:.3f}s'.format(avg_det_time/num_images))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    map = evaluate_detections(all_boxes, output_dir, dataset)
    return map


def evaluate_detections(box_list, output_dir, dataset):
    write_voc_results_file(box_list, dataset)  # 将每张图片的所有检测结果分别写入到对应类别的文件夹中#'/VOCNWPU/VOC2012/results/det_test_cls_name.txt'
    map = do_python_eval(output_dir)
    return map


if __name__ == '__main__':
    # load net
    num_classes = len(labelmap) + 1                      # +1 for background
    net = build_SMENet('test', 400, num_classes)
    best_map = 0.5
    # load data
    dataset = VOCDetection(args.voc_root, [('2012', 'test')],
                           BaseTransform(400, dataset_mean),
                           VOCAnnotationTransform())
    for iter in range(1):
        weights_path = 'weights/SME4004.pth'

        net.load_state_dict(torch.load(weights_path, map_location="cpu"))
        net.eval()
        print('Finished loading model!')

        if args.cuda:
            net = net.cuda()
        mAP = test_net(net, dataset)