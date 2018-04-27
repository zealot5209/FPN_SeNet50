#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import os.path as osp
import sys
import glob
import cPickle
from xml.dom.minidom import parse
import xml.dom.minidom
import numpy as np
import BBoxXmlTool as bxt
import xml.etree.ElementTree as ET
# =======================================


# usage:this script is used to check
#       the files between ground truth files
#       and predect results files
# date:20180227
# author:guoc

# =======================================
def check_neg(gtxmlPath,resultxmlPath):
    afiles = []
    bfiles = []
    for root, dirs, files in os.walk(gtxmlPath):
        print gtxmlPath, 'All files numbers:', len(files)
        for f in files:
            # 比较文件名不含格式后缀
            # afiles.append(root + f[0:-4])

            # 比较文件名含格式后缀
            afiles.append(root + f)
    for root, dirs, files in os.walk(resultxmlPath):
        print resultxmlPath, 'All files numbers:', len(files)
        for f in files:
            # 比较文件名不含格式后缀
            # bfiles.append(root + f[0:-4])

            # 比较文件名含格式后缀
            bfiles.append(root + f)
            # sizeB = os.path.getsize(root + "/" + f) 此处定义的size无法在commonfiles进行比较. (A,B在各自的循环里面)

    # 去掉afiles中文件名的gtxmlPath (拿A,B相同的路径\文件名,做成集合,去找交集)
    gtxmlPathlen = len(gtxmlPath)
    aafiles = []
    for f in afiles:
        aafiles.append(f[gtxmlPathlen:])

    # 去掉bfiles中文件名的resultxmlPath
    resultxmlPathlen = len(resultxmlPath)
    bbfiles = []
    for f in bfiles:
        bbfiles.append(f[resultxmlPathlen:])

    afiles = aafiles
    bfiles = bbfiles
    setA = set(afiles)
    setB = set(bfiles)

    afiles = aafiles

    bfiles = bbfiles
    setA = set(afiles)
    setB = set(bfiles)

    print("-------------done!------------")
    # 处理仅出现在results_XML目录中的文件
    onlyFiles = setA ^ setB
    afiles = aafiles
    bfiles = bbfiles
    setA = set(afiles)
    setB = set(bfiles)

    print("-------------done!------------")
    # 处理仅出现在results_XML目录中的文件
    onlyFiles = setA ^ setB
    print("-------------done!------------")
    # 处理仅出现在results_XML目录中的文件
    onlyFiles = setA ^ setB
    aonlyFiles = []
    bonlyFiles = []
    for of in onlyFiles:
        if of in afiles:
            aonlyFiles.append(of)
        elif of in bfiles:
            bonlyFiles.append(of)

    print gtxmlPath, 'only files numbers:', len(aonlyFiles)
    print resultxmlPath, 'only files numbers:', len(bonlyFiles)

    for of in sorted(bonlyFiles):
        os.remove(resultxmlPath + '/' + of)

    xmlnames = os.listdir(gtxmlPath)
    tmpdir = os.path.dirname(gtxmlPath)
    imglistPath = tmpdir + '/imglist' + '.txt'
    imglistfile = open(imglistPath, 'w+')
    if (len(xmlnames) > 0):

        for xmlname in xmlnames:

            # imglistfile = open(tmpdir + '/imglist' + '.txt')
            tmp1 = xmlname.split(".")
            tmp2 = tmp1[0]
            imglistfile.write(tmp2)
            imglistfile.write('\n')

        imglistfile.close()

    return imglistPath


# ======================================

def cleandata(xml_dir):

    xml_list = os.listdir(xml_dir)

    minw = 10000
    minh = 10000
    maxw = 0
    maxh = 0
    meanw = 0
    meanh = 0
    counter = 0

    img_counter = 0
    bbox_counter = 0
    cls_dict = {}

    for idx, fname in enumerate(xml_list):
        xml_path = osp.join(xml_dir, fname)

        tmp_img = bxt.IMGBBox(xml_path=xml_path)

        minw = min(minw, tmp_img.width)
        minh = min(minh, tmp_img.height)
        maxw = max(maxw, tmp_img.width)
        maxh = max(maxh, tmp_img.height)
        counter += 1
        meanw += tmp_img.width
        meanh += tmp_img.height


        bbox_num = len(tmp_img.bboxes)
        # if bbox_num > 0 and not tmp_img.img is None :
        if bbox_num > 0:
            # print "-- [%d/%d] %s %d" % (idx, img_num, fname, bbox_num)

            # 记录每一个类别的框的个数
            for tmpidx, item in enumerate(tmp_img.bboxes):
                if cls_dict.has_key(item.name) : cls_dict[item.name] += 1
                else: cls_dict[item.name] = 1

                if item.name == 'portable_other' : tmp_img.bboxes[tmpidx].name = 'portable_side'

            img_counter += 1
            bbox_counter += bbox_num
            # if save_img :
            #     tmp_img.saveXML(save_dir=save_xml_dir)
            #     tmp_img.saveIMG(save_dir=save_img_dir)
        else:
            print "no bbox: " + fname
    # print "bbox num :", bbox_counter
    # print "img num :", img_counter
    print "cls :", cls_dict
    # print "width: min %d mean %d max %d" % (minw, meanw / counter, maxw)
    # print "height: min %d, mean %d max %d" % (minh, meanh / counter, maxh)

    return cls_dict




# ==========================================

def savetxt_extractxml(objcls,resultxmlPath):
    # dirPath = '/home/ubuntu/workfile_guoc/pkg_mAP/results_xml'
    # outPath = '/home/ubuntu/workfile_guoc/pkg_mAP/output_txt/hammer.txt'

    print 'objcls', objcls

    outdirPath = os.path.dirname(resultxmlPath)
    # os.mknod(outdirPath + objcls +'.txt3')
    outtmpPath = outdirPath + '/tmpcompTxt'
    if not (os.path.exists(outtmpPath)):
        os.mkdir(outtmpPath)
    outPath = outtmpPath + '/' + objcls + '.txt'
    dirPath = resultxmlPath

    if (os.path.exists(dirPath)):
        # filenames = os.listdir(dirPath)
        filenames = glob.glob(dirPath + '//*.xml')
        # filenames = glob.glob(r'/home/ubuntu/workfile_guoc/pkg_mAP/results_xml1/*.xml')
        fileout = open(outPath, 'w+')
        # fileout = open(outPath)
        for filename in filenames:

            # 使用minidom解析器打开 XML 文档
            DOMTree = xml.dom.minidom.parse(filename)
            collection = DOMTree.documentElement
            if collection.hasAttribute("annotation"):
                print "Root element : %s" % collection.getAttribute("annotation")

            # 在集合中获取所有对象
            objects = collection.getElementsByTagName("object")


            # 查找每个object的详细信息
            for object in objects:
                name = object.getElementsByTagName('name')[0]
                namestr = name.childNodes[0].data
                if namestr == objcls:
                    score = object.getElementsByTagName('score')[0]
                    scorestr = score.childNodes[0].data
                    bndbox = object.getElementsByTagName("bndbox")
                    bndbox_xmin = bndbox[0].getElementsByTagName('xmin')[0]
                    bndbox_ymin = bndbox[0].getElementsByTagName('ymin')[0]
                    bndbox_xmax = bndbox[0].getElementsByTagName('xmax')[0]
                    bndbox_ymax = bndbox[0].getElementsByTagName('ymax')[0]
                    bndbox_xminstr = bndbox_xmin.childNodes[0].data
                    bndbox_yminstr = bndbox_ymin.childNodes[0].data
                    bndbox_xmaxstr = bndbox_xmax.childNodes[0].data
                    bndbox_ymaxstr = bndbox_ymax.childNodes[0].data

                    filebasename = os.path.basename(filename)
                    temp = [filebasename, scorestr, bndbox_xminstr, bndbox_yminstr, bndbox_xmaxstr, bndbox_ymaxstr]
                    # temp1 = ' '.join(temp)
                    # ' '.join(temp)

                    fileout.writelines(' '.join(temp))
                    fileout.write('\n')

        fileout.close()
    return outPath



# ====================================================


def parse_rec(filename):
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
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
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
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,     #该函数主要实现单一类别的AP计算
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    ###if not os.path.isdir(cachedir):
    ###    os.mkdir(cachedir)
    ###cachefile = os.path.join(cachedir, 'annots.pkl')


    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    ###if not os.path.isfile(cachefile):
        # load annots
    recs = {}
    for i, imagename in enumerate(imagenames):
        recs[imagename] = parse_rec(annopath.format(imagename))
        if i % 100 == 0:
            print 'Reading annotation for {:d}/{:d}'.format(
                i + 1, len(imagenames))
        # save
        ###print 'Saving cached annotations to {:s}'.format(cachefile)
        ###with open(cachefile, 'w') as f:
        ###    cPickle.dump(recs, f)  #根据cpickle模块对recs进行序列化操作
    ###else:
        # load
    ###    with open(cachefile, 'r') as f:
    ###        recs = cPickle.load(f)

    # extract gt objects for this class：从groundtruth_xml文件夹中提取单一类别的矩形框
    class_recs = {}
    npos = 0
    # 根据imgname和clasname，从xml文件中抽取出对应的矩形框objec
    for imagename in imagenames:
        print '======'
        print 'imagename', imagename
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        print 'bbox', bbox
        print '======'
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,   #class_recs是一个结构体，用来存放某张图片的某个类别的信息
                                 'difficult': difficult,
                                 'det': det}

    # read dets：读取检测结果
    detfile = detpath.format(classname)
    print 'detfile===', detfile
    with open(detfile, 'r') as f:
        lines = f.readlines()


    if len(lines) == 0:
        print 'no boudning box:', classname
        return None, None, -1

    splitlines = [x.strip().split(' ') for x in lines]  #
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    print 'splitlines====', splitlines
    print 'confidence====', confidence
    print 'BB====', BB
    # sort by confidence：对单一类别的检测目标，进行降序排序
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)

    BB = BB[sorted_ind, :]

    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs：
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[(image_ids[d].split('.'))[0]]   #imagenames没有.xml后缀名，image_ids是有.xml后缀名的，所以报错！！！
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

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
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap





# ===================================================

def do_mAP_eval(gtxmlPath, resultxmlPath):

    annopath = gtxmlPath + '/{:s}.xml'     #此处方法

    imagesetfile = check_neg(gtxmlPath, resultxmlPath)

    # cachedir = os.path.join(self._devkit_path, 'annotations_cache')
    cachedir = os.path.join(os.path.dirname(gtxmlPath), 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    # use_07_metric = True if int(self._year) < 2010 else False
    use_07_metric = False
    print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')

    # self_classes = ('__background__', 'spray', 'hammer', 'knife')
    classes_dict = cleandata(gtxmlPath)
    classes_dict = cleandata(resultxmlPath)
    self_classes = classes_dict.keys()
    self_classes.append('__background__')
    # self_classes = cleanData(gtxmlPath,resultxmlPath)  #此处需要调用cleandata脚本去求出classes矩阵,在类别矩阵需要添加“_background_”

    dict_mAP = {}

    print 'self_classes length', len(self_classes)
    #for i, cls in enumerate(self_classes):             #此处需要调用classes
    for i, cls in enumerate(self_classes):  # 此处需要调用classes
        if cls == '__background__':
            continue
        # filename = self._get_voc_results_file_template().format(cls)

        filename = savetxt_extractxml(cls, resultxmlPath)



        rec, prec, ap = voc_eval(
            filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
            use_07_metric=use_07_metric)

        if cls == 'person':
            print 'pascal_voc'
            print 'filename===', filename
            print 'annopath===', annopath
            print 'imagesetfile===', imagesetfile
            print 'cls===', cls
            print 'ap===', ap

        if ap == -1:
            continue
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        dict_mAP[cls] = ap

    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    dict_mAP['Mean AP'] = np.mean(aps)

    mAP_output = os.path.dirname(gtxmlPath) + '/mAP_output' + '.txt'
    mAP_os = open(mAP_output, 'w+')
    for key, value in dict_mAP.items():
        mAP_os.write(key + '：' + str(value))
        mAP_os.write('\n')

    mAP_os.close()

    print('--------------------------------------------------------------')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('--------------------------------------------------------------')
    print('')
    print('--------------------------------------------------------------')



# =============================================
#usage：调用do_mAP_eval(groundxmlpath,resultxmlpath)函数，即可输出结果

if __name__ == '__main__':
    if len(sys.argv) == 3:
        gtxmlPath = sys.argv[1]
        resultxmlPath = sys.argv[2]
        #gtxmlPath = '/home/yansong/FPN-git-unsky2/VOCdevkit2007/VOC2007/Annotations/'
        #resultxmlPath = '/home/yansong/FPN-git-unsky2/experiments/result/result'
        #gtxmlPath = '/home/yansong/FPN-git-unsky2/experiments/result/gt'
        #resultxmlPath = '/home/yansong/FPN-git-unsky2/experiments/result/result'
        do_mAP_eval(gtxmlPath, resultxmlPath)
    print("done!")
#==============================================



