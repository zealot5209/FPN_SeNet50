#!/usr/bin/env python

# --------------------------------------------------------
# This script is used to test model
# and save the result images and xmls
# as well

# Date:201803
# Author: YAN SONG
# parameters example:
# --gpu 1
# --imdb voc_2007_test
# --cfg /home/yansong/FPN-git-unsky2/experiments/cfgs/FP_Net_end2end.yml
# --net /home/yansong/FPN-git-unsky2/output/FP_Net_end2end/voc_2007_trainval/fpn_iter_50000.caffemodel
# --def /home/yansong/FPN-git-unsky2/models/pascal_voc/FPN/FP_Net_end2end/test.prototxt
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""
import _init_paths
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
from fast_rcnn.test import im_detect, test_net

from lib.fast_rcnn.nms_wrapper import nms
from lib.utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os.path as osp
import lib.BBoxXmlTool as bb
import caffe, cv2
import pprint
import time, os, sys


CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

color_dict = {}


def init_colors(CLASSES):
    for cls_ind, cls in enumerate(CLASSES[1:]):
        colors = [255*np.random.rand(), 255*np.random.rand(), 255*np.random.rand()]
        color_dict[cls] = colors


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=1, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default='/home/yansong/FPN-git-unsky2/models/pascal_voc/FPN/FP_Net_end2end/test.prototxt', type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default='/home/yansong/FPN-git-unsky2/output/FP_Net_end2end/voc_2007_trainval/fpn_iter_50000.caffemodel', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default='/home/yansong/FPN-git-unsky2/experiments/cfgs/FP_Net_end2end.yml', type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--vis', dest='vis', help='visualize detections',
                        action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)
    parser.add_argument('--img_path', dest='IMAGE_PATH',
                        help='path of test images',
                        default='VOCdevkit2007/VOC2007/JPEGImages', type=str)
    parser.add_argument('--num_test', dest='num_test',
                        help='number of images to test',
                        default=0, type=int)
    parser.add_argument('--save_img_dir', dest='save_img_dir',
                        default='/home/yansong/FPN-git-unsky2/experiments/result/ResNet50/img', type=str)
    parser.add_argument('--save_xml_dir', dest='save_xml_dir',
                        default='/home/yansong/FPN-git-unsky2/experiments/result/ResNet50/xml', type=str)
    parser.add_argument('--conf_thresh', dest='CONF_THRESH',
                        default=0.8, type=float)
    parser.add_argument('--nms_thresh', dest='NMS_THRESH',
                        default=0.3, type=float)
    parser.add_argument('--test_img_name_list', dest='name_list',
                        default='/home/yansong/FPN-git-unsky2/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def get_img_name(names, path, num):
    idx = 0
    fr = open(path)
    array = fr.readlines()
    num_of_lines = len(array)
    if num == 0:
        num = num_of_lines
    for line in array:
        if idx >= min(num, num_of_lines):
            break
        line = line.strip() + '.jpg'
        names.append(line)
        idx += 1

def vis_detections(new_bboxes, result, class_name, dets, thresh=0.5):
#def vis_detections(new_bboxes, result, class_name, dets, ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    colors = color_dict[class_name]
    r1 = colors[0]
    r2 = colors[1]
    r3 = colors[2]
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        # ax.add_patch(
        #     plt.Rectangle((bbox[0], bbox[1]),
        #                   bbox[2] - bbox[0],
        #                   bbox[3] - bbox[1], fill=False,
        #                   edgecolor='red', linewidth=1)
        #     )
        #
        # ax.text(bbox[0], bbox[1] - 2,
        #         '{:s} {:.3f}'.format(class_name, score),
        #         bbox=dict(facecolor='blue', alpha=0.5),
        #         fontsize=14, color='white')

        cv2.rectangle(result, (bbox[0], bbox[3]), (bbox[2], bbox[1]), (r1, r2, r3), 2)

        print '==========='
        print (bbox[0], bbox[1])
        print (bbox[2], bbox[3])
        print '=========='

        rec = np.array([[bbox[0], bbox[1]], [bbox[0] + 70, bbox[1]], [bbox[0] + 70, bbox[1] + 16], [bbox[0], bbox[1] + 16]])
        cv2.fillPoly(result, np.int32([rec]), (255, 0, 0))
        cv2.putText(result, str(class_name) + " " + ("%.3f" % score), (int(bbox[0]), int(bbox[1] + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        #save bboxes
        tmp_bbox = bb.BBox(class_name, score, bbox[0], bbox[1], bbox[2], bbox[3])
        new_bboxes.append(tmp_bbox)

    # ax.set_title(('{} detections with '
    #                'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                                thresh),
    #                fontsize=14)


def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, args.IMAGE_PATH, image_name)
    #im__ = None
    im__ = cv2.imread(im_file)
    #print '----------', im__
    #if im__ == None:
    #    print " --- ERROR: Cannot Load Image: ", im_file
    #    return False

    print 'image name:', image_name
    result = im__

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im__)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    im__ = im__[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots(figsize=(12, 12))
    #ax.imshow(im__, aspect='equal')

    new_bboxes = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, args.NMS_THRESH)
        dets = dets[keep, :]
        #vis_detections(new_bboxes, result, cls, dets, ax, thresh=args.CONF_THRESH)
        vis_detections(new_bboxes, result, cls, dets, thresh=args.CONF_THRESH)


    #Save result
    img_obj = bb.IMGBBox()
    img_obj.img_name = image_name
    img_obj.xml_name = image_name.strip().split('.')[0] + '.xml'
    img_obj.setIMG(result)
    img_obj.saveIMG(args.save_img_dir, args.save_img_dir)
    img_obj.bboxes = new_bboxes
    img_obj.saveXML(args.save_xml_dir, args.save_xml_dir)
    ####

    #cv2.imshow('result', result)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.imwrite(osp.join(save_dir, image_name), result)

    #plt.axis('off')
    #plt.tight_layout()
    #plt.draw()
    #return True


if __name__ == '__main__':
    args = parse_args()

    #initiate box colors
    init_colors(CLASSES)

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    print 'caffemodel',args.caffemodel
    while not os.path.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    #uncomment to show mAP
    # imdb = get_imdb(args.imdb_name)
    # imdb.competition_mode(args.comp_mode)
    # if not cfg.TEST.HAS_RPN:
    #     imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)
    #
    # test_net(net, imdb, max_per_image=args.max_per_image, vis=args.vis)

    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _ = im_detect(net, im)

    testImg_names = []
    get_img_name(testImg_names, args.name_list, args.num_test)
    print 'testImg_names', testImg_names

    for im_name in testImg_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        flag = demo(net, im_name)
        if not flag:
            continue

    #plt.show()

