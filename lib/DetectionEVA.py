import os
import sys
import os.path as osp
import numpy as np
import cv2
import copy
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import BBoxXmlTool as bxt
import pickle
import cPickle

BIG_SET = True
# CLASSES_B = ['egg', 'sausage', 'sausage_other', 'tin_bottle', 'tin',  'milk_bag', 'milk_bottle', 'milk_other', 'fruit', 'milk_box']
# CSASS_MAP = {'egg':'egg', 'sausage':'sausage', 'sausage_other':'sausage', 'tin_bottle':'tin', 'tin':'tin',  'milk_bag':'milk', 'milk_bottle':'milk', 'milk_other':'milk', 'fruit':'fruit', 'milk_box':'milk'}
# CLASSES = ['egg', 'sausage', 'tin', 'milk', 'fruit']
CLASSES = ['spray', 'hammer', 'slingshot', 'gun', 'scissors', 'blunt', 'knife']

class TMPIMGBBox(object):
    def __init__(self):
        self.bboxes = []
        self.width = 0
        self.height = 0

def calc_iou(bbox1, bbox2) :
    sw = max(min(bbox1.xmax, bbox2.xmax) - max(bbox1.xmin, bbox2.xmin), 0.)
    sh = max(min(bbox1.ymax, bbox2.ymax) - max(bbox1.ymin, bbox2.ymin), 0.)

    area1 = (bbox1.xmax - bbox1.xmin) * (bbox1.ymax - bbox1.ymin)
    area2 = (bbox2.xmax - bbox2.xmin) * (bbox2.ymax - bbox2.ymin)

    return sw * sh / float(area1 + area2 - sw * sh)

def calc_param(gt_list, result_list, th = 0.5, iou_th = 0.5):
    TP = 0.
    TN = 0.
    FP = 0.
    FN = 0.
    for idx in range(len(gt_list)) :
        gt_img = copy.deepcopy(gt_list[idx])
        result_img = copy.deepcopy(result_list[idx])
        result_img.bboxes = [x for x in result_img.bboxes if x.score >= th]

        ## TN
        if len(gt_img.bboxes) == 0 and len(result_img.bboxes) == 0 :
            TN += 1.
            continue

        ## FP
        if len(gt_img.bboxes) == 0 and len(result_img.bboxes) > 0 :
            FP += 1.
            continue

        ## FN
        if len(gt_img.bboxes) > 0 and len(result_img.bboxes) == 0 :
            FN += 1.
            continue

        overlap = False
        for gt_bbox in gt_img.bboxes :
            for result_bbox in result_img.bboxes :
                if calc_iou(gt_bbox, result_bbox) >= iou_th :
                    overlap = True
                    break

            if overlap : break

        if overlap : TP += 1.
        else : FN += 1.

    return TP, TN, FP, FN

def Recall(TP, TN, FP, FN):
    if TP == 0 : return 0
    if TP + FN == 0 : return 1
    return TP / float(TP + FN)

def Precision(TP, TN, FP, FN):
    if TP == 0 : return 0
    if TP + FP == 0 : return 1
    return TP / float(TP + FP)

def TPR(TP, TN, FP, FN):
    if TP == 0 : return 0
    if TP + FN == 0 : return 1
    return TP / float(TP + FN)

def FPR(TP, TN, FP, FN):
    if FP == 0 : return 0
    if FP + TN == 0 : return 1
    return FP / float(FP + TN)

def Accuracy(TP, TN, FP, FN):
    if TP + TN + FP + FN == 0 : return 1
    return float(TP + TN) / float(TP + TN + FP + FN)

def eval(image_list, gt_dir, result_dir, save_dir = ".") :
    if not osp.isdir(gt_dir) or not osp.isdir(result_dir) :
        print "Import DIR"
        print gt_dir
        print result_dir
        return None

    with open(image_list) as fr:
        lines = fr.readlines()


    gt_evaimg_dict = {}
    result_evaimg_dict = {}

    tmp_classes = CLASSES
    for cls_name in CLASSES:
        gt_evaimg_dict[cls_name] = []
        result_evaimg_dict[cls_name] = []

    image_num = len(lines)
    pos_num = 0
    neg_num = 0

    detect_pos_num = 0
    detect_neg_num = 0
    for idx, xml_name in enumerate(lines):
        xml_name, _ = osp.splitext(xml_name)
        # if idx > 400 : break

        fname = xml_name.strip()
        print "[%d/%d] %s" % (idx, image_num, xml_name)

        gt_xml_path = osp.join(gt_dir, fname + ".xml")
        result_xml_path = osp.join(result_dir, fname + ".xml")

        gt_img = bxt.IMGBBox(xml_path= gt_xml_path)
        result_img = bxt.IMGBBox(xml_path=result_xml_path)

        for cls_name in tmp_classes:
            gt_bbox = TMPIMGBBox()
            if len(gt_img.bboxes) > 0 : pos_num += 1
            else : neg_num += 1
            for bbox in gt_img.bboxes:

                tmp_cls_name = bbox.name
                if tmp_cls_name == cls_name:
                    bbox.name = tmp_cls_name
                    gt_bbox.bboxes.append(bbox)
            gt_evaimg_dict[cls_name].append(gt_bbox)

            result_bbox = TMPIMGBBox()
            if len(result_img.bboxes) > 0 : detect_pos_num += 1
            else : detect_neg_num += 1
            for bbox in result_img.bboxes:
                tmp_cls_name = bbox.name
                if tmp_cls_name == cls_name:
                    bbox.name = tmp_cls_name
                    result_bbox.bboxes.append(bbox)
            result_evaimg_dict[cls_name].append(result_bbox)

    recall_list = []
    precision_list = []
    tpr_list = []
    fpr_list = []

    iou_th = 0.5

    for cls_name in tmp_classes:
        fw = None
        if osp.isdir(save_dir):
            fw = open(osp.join(save_dir, "ana_" + cls_name + "_iou" + str(iou_th) + ".csv"), "w")
            fw.write("threshold, TP, TN, FP, FN, Recall, Precision, Accuracy, TPR, FPR\n")

        for t in range(5, 11, 1) :
            th = 0.1 * t
            TP, TN, FP, FN = calc_param(gt_evaimg_dict[cls_name], result_evaimg_dict[cls_name], th, iou_th)

            print cls_name, "th:", th, "IOU:", iou_th
            print cls_name, "TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN

            recall = Recall(TP, TN, FP, FN)
            print cls_name, "Recall: ", recall

            precision = Precision(TP, TN, FP, FN)
            print cls_name, "Precision: ", precision

            accuracy = Accuracy(TP, TN, FP, FN)
            print cls_name, "Accuracy: ", accuracy

            tpr = TPR(TP, TN, FP, FN)
            print cls_name, "TPR: ", tpr

            fpr = FPR(TP, TN, FP, FN)
            print cls_name, "FPR: ", fpr
            print "  ---- "
            if fw != None :
                fw.write("%f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n" % (th, TP, TN, FP, FN, recall, precision, accuracy, tpr, fpr))
            recall_list.append(recall)
            precision_list.append(precision)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        if fw != None : fw.close()
        #
        #     plt.figure(1)
        #     plt.plot(np.array(recall_list), np.array(precision_list))
        #     plt.ylabel('precision')
        #     plt.xlabel('recall')
        #     plt.title("P/R")
        #     plt.savefig("ana_pr_iou" + str(iou_th) + ".png")
        #     # plt.show()
        #
        #     plt.figure(2)
        #     plt.plot(np.array(fpr_list), np.array(tpr_list))
        #     plt.ylabel('tpr')
        #     plt.xlabel('fpr')
        #     plt.title("ROC")
        #     plt.savefig("ana_roc_iou" + str(iou_th) + ".png")
        #     # plt.show()

if __name__ == '__main__':
    argv = sys.argv
    argc = len(argv)

    # # eva_pkl()
    #
    # image_list=r'D:\D\yisuo_knife_result_180227\yisuo_knife_result_180227\list.txt'
    # gt_dir=r'D:\D\yisuo_knife_result_180227\yisuo_knife_result_180227\xml'
    # result_dir=r'D:\D\yisuo_knife_result_180227\yisuo_knife_result_180227\all_xml'
    # save_dir=r'D:\D\yisuo_knife_result_180227\yisuo_knife_result_180227\1'
    #
    # eval(image_list, gt_dir, result_dir, save_dir)

    #if argc == 3 : eval(argv[1], argv[2])
    if argc == 5 : eval(argv[1], argv[2], argv[3], argv[4])
    else: print "usage: %s imagelist  gt_ann test_ann <optional resut_save_dir>" % argv[0]