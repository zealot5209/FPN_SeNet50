#coding=utf-8
import os
import sys
import os.path as osp
import numpy as np
import cv2
import copy
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import BBoxXmlTool as bxt
import random
import shutil
import cPickle


def main(img_dir, xml_dir, save_dir):

    img_list = os.listdir(img_dir)
    xml_list = os.listdir(xml_dir)

    name_list = []

    for img_name in img_list :
        fname, ext = osp.splitext(img_name)
        if ext == ".jpg" and osp.isfile(osp.join(xml_dir, fname + ".xml")) : name_list.append(fname)

    # neg_list = []
    # neg_img_list = os.listdir(neg_jpg_dir)
    # for img_name in neg_img_list :
    #     fname, ext = osp.splitext(img_name)
    #     if ext == ".jpg" and osp.isfile(osp.join(neg_xml_dir, fname + ".xml")) : neg_list.append(fname)
    #
    # random.shuffle(neg_list)
    # neg_num = len(neg_list)
    # neg_idx = 0
    #print name_list
    if not osp.isdir(save_dir + "/draw"):
        os.makedirs(save_dir + "/draw")

    img_num = len(name_list)
    for idx, name in enumerate(name_list):
        print "-- [%d/%d]%s " % (idx, img_num, fname)
        img_path = osp.join(img_dir, name + ".jpg")
        xml_path = osp.join(xml_dir, name + ".xml")
        tmp_img = bxt.IMGBBox(img_path=img_path, xml_path=xml_path)
        tmp_img.saveIMGRoi(save_dir)
        tmp_img.showIMG(save_dir=save_dir+"/draw")

if __name__ == '__main__':
    argv = sys.argv
    argc = len(argv)

    #generateXML()

    if argc == 4 : main(argv[1], argv[2], argv[3])
