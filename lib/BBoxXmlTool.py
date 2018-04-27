# coding: utf-8
import xml.etree.ElementTree as ET
from lxml import etree as et
import os
import os.path as osp
import numpy as np
import cv2
import math
import copy
import random
import scipy.signal as signal
import codecs
ENCODE_METHOD = 'UTF-8'

def indent(elem, level=0):
    i = "\n" + level*"\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def prettify(elem):
    """
        Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf8')
    root = et.fromstring(rough_string)
    return et.tostring(root, pretty_print=True, encoding=ENCODE_METHOD).replace("  ".encode(), "\t".encode())
    # minidom does not support UTF-8
    '''reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t", encoding=ENCODE_METHOD)'''

class BBox(object):
    def __init__(self, name = "none", score = 1., xmin = 0., ymin = 0., xmax = 0., ymax = 0.):
        self.name = name
        self.score = score
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.pose = 'Unspecified'
        self.truncated = 0
        self.difficult = 0

class IMGBBox(object):
    def __init__(self, img_path = "", xml_path = "", txt_path = ""):
        self.img_path = img_path
        self.xml_path = xml_path
        self.xml_name = ""
        self.img_name = ""
        self.bboxes = []
        self.width = 0
        self.height = 0
        self.orig_xmin = 0
        self.orig_xmax = 0
        self.orig_ymin= 0
        self.orig_ymax= 0
        self.img = None

        if osp.isfile(img_path) : self.loadIMG(img_path)
        if osp.isfile(xml_path): self.loadXML(xml_path)
        if osp.isfile(txt_path): self.loadTXT(txt_path)

    def loadIMG(self, img_path):
        self.clearIMG()
        if not osp.isfile(img_path): return False
        img = None
        try:
            img = cv2.imread(img_path)
        except:
            print " --- ERROR: Cannot Load Image: ", img_path
            return False

        if self.setIMG(img) :
            self.img_path = img_path
            self.img_name = self.getBaseName(img_path)
            return True
        return False

    def setIMG(self, img):
        self.width = 0
        self.height = 0

        if img is None : return False
        try:
            self.width = img.shape[1]
            self.height = img.shape[0]
        except:
            print " --- ERROR: Cannot Set Image: "
            return False

        self.img = img
        return True

    def clearIMG(self):
        self.clearIMGDatum()
        self.img_path = ""
        self.img_name = ""
        self.width = 0
        self.height = 0

    def clearIMGDatum(self):
        self.img = None

    def doNMS(self, iou_th = 0.01):
        if len(self.bboxes) == 0 : return False
        tmp_bboxes = []

        name_set = set([])
        for item in self.bboxes:
            name_set.add(item.name)
        name_list = list(name_set)
        bboxes_list = [] * len(name_list)


        for item in self.bboxes:
            tmp_bboxes.append([item.xmin, item.ymin, item.xmax, item.ymax, item.score])
        keep = py_cpu_nms(np.array(tmp_bboxes), iou_th)

        final_bboxes = []
        for idx in keep:
            final_bboxes.append(BBox())


    def loadXML(self, xml_path):
        self.clearXML()
        if not osp.isfile(xml_path): return False

        tree = None
        # parser = ET.XMLParser(recover=True)
        # parser = ET.XMLParser(encoding=ENCODE_METHOD)

        tree = ET.parse(xml_path).getroot()
        # try:
        #
        #     parser = ET.XMLParser(encoding=ENCODE_METHOD)
        #     tree = ET.parse(xml_path, parser=parser).getroot()
        #     # tree = ET.parse(xml_path)
        # except:
        #     print " --- ERROR: Cannot Load XML: ", xml_path
        #     return False

        sizes = tree.findall("size")
        if not sizes is None and len(sizes) > 0:
            self.width = int(sizes[0].find("width").text)
            self.height = int(sizes[0].find("height").text)

        orig_xmin = tree.find("orig_xmin")
        if not orig_xmin is None:
            self.orig_xmin = int(orig_xmin.text)

        orig_xmax = tree.findall("orig_xmax")
        if not orig_xmax is None and len(orig_xmax) > 0:
            self.orig_xmax = int(orig_xmax[0].text)

        orig_ymin = tree.findall("orig_ymin")
        if not orig_ymin is None and len(orig_ymin) > 0:
            self.orig_ymin = int(orig_ymin[0].text)

        orig_ymax = tree.findall("orig_ymax")
        if not orig_ymax is None and len(orig_ymax) > 0:
            self.orig_ymax = int(orig_ymax[0].text)

        objs = tree.findall('object')
        for obj in objs:
            tmp_bbox = BBox()

            score = obj.find('score')
            if not score is None : tmp_bbox.score = float(score.text)

            name = obj.find('name')
            if not name is None and not name.text is None : tmp_bbox.name = name.text

            pose = obj.find('pose')
            if not pose is None and not pose.text is None : tmp_bbox.pose = pose.text

            truncated = obj.find('truncated')
            if not truncated is None and not truncated.text is None: tmp_bbox.truncated = int(truncated.text)

            difficult = obj.find('difficult')
            if not difficult is None and not difficult.text is None : tmp_bbox.difficult = int(difficult.text)

            bndbox = obj.find('bndbox')

            if bndbox is None : continue

            x1 = float(bndbox.find('xmin').text)
            x2 = float(bndbox.find('xmax').text)
            y1 = float(bndbox.find('ymin').text)
            y2 = float(bndbox.find('ymax').text)

            tmp_bbox.xmin = max(min(x1, x2), 0)
            tmp_bbox.xmax = min(max(x1, x2), self.width)
            tmp_bbox.ymin = max(min(y1, y2), 0)
            tmp_bbox.ymax = min(max(y1, y2), self.height)

            if tmp_bbox.xmin >= tmp_bbox.xmax  or tmp_bbox.ymin >= tmp_bbox.ymax : continue

            self.bboxes.append(tmp_bbox)

        if len(self.bboxes) == 0 :
            self.clearXML()
            return False

        self.xml_path = xml_path
        self.xml_name = self.getBaseName(xml_path)

        width = tree.findall("image_width")
        if not width is None and len(width) > 0 : self.width = float(width[0].text)
        height = tree.findall("image_height")
        if not height is None and len(height) > 0 : self.height = float(height[0].text)

        image_name = tree.findall("image_name")
        if not image_name is None and len(image_name) > 0 : self.img_name = self.getBaseName(image_name[0].text)

        return True

    def clearXML(self):
        self.xml_path = ""
        self.xml_name = ""
        self.bboxes = []

    def getBaseName(self, path):
        return osp.basename(path)

    def saveXML(self, save_path = "", save_dir = ""):
        #if len(self.bboxes) <= 0 : return False
        # print "saveing"

        if self.xml_name == "" :
            fname, _ = osp.split(self.img_name)
            self.xml_name = fname + ".xml"

        tmp_save_path = save_path
        if osp.isdir(save_dir) : tmp_save_path = osp.join(save_dir, self.xml_name)

        root = ET.Element("annotation")
        img_name = ET.SubElement(root, "filename")
        img_name.text = self.img_name

        path = ET.SubElement(root, "path")
        path.text = self.img_path

        size = ET.SubElement(root, "size")

        width = ET.SubElement(size, "width")
        width.text = str(int(self.width))

        height = ET.SubElement(size, "height")
        height.text = str(int(self.height))

        orig_xmin = ET.SubElement(root, "orig_xmin")
        orig_xmin.text = str(self.orig_xmin)

        orig_ymin = ET.SubElement(root, "orig_ymin")
        orig_ymin.text = str(self.orig_ymin)

        orig_xmax = ET.SubElement(root, "orig_xmax")
        orig_xmax.text = str(self.orig_xmax)

        orig_ymax = ET.SubElement(root, "orig_ymax")
        orig_ymax.text = str(self.orig_ymax)

        for item in self.bboxes:

            object = ET.SubElement(root, "object")

            score = ET.SubElement(object, "score")
            score.text = str(item.score)

            flag = ET.SubElement(object, "flag")
            flag.text = "0"

            name = ET.SubElement(object, "name")
            name.text = item.name

            pose = ET.SubElement(object, "pose")
            pose.text = item.pose

            truncated = ET.SubElement(object, "truncated")
            truncated.text = str(item.truncated)

            difficult = ET.SubElement(object, "difficult")
            difficult.text = str(item.difficult)

            bndbox = ET.SubElement(object, "bndbox")

            # xmin = ET.SubElement(bndbox, "xmin")
            # xmin.text = str(int(math.floor(item.xmin)))
            #
            # ymin = ET.SubElement(bndbox, "ymin")
            # ymin.text = str(int(math.floor(item.ymin)))
            #
            # xmax = ET.SubElement(bndbox, "xmax")
            # xmax.text = str(int(math.floor(item.xmax)))
            #
            # ymax = ET.SubElement(bndbox, "ymax")
            # ymax.text = str(int(math.floor(item.ymax)))

            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text = str(item.xmin)

            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text = str(item.ymin)

            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text = str(item.xmax)

            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(int(math.floor(item.ymax)))


        out_file = codecs.open(tmp_save_path, 'w', encoding=ENCODE_METHOD)

        prettifyResult = prettify(root)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()

        # try:
        #     out_file = codecs.open(tmp_save_path, 'w', encoding=ENCODE_METHOD)
        #
        #     prettifyResult = self.prettify(root)
        #     out_file.write(prettifyResult.decode('utf8'))
        #     out_file.close()
        # except:
        #     print " --- ERROR: Cannot Save XML: ", tmp_save_path
        #     return False
        self.xml_path = save_path
        self.xml_name = self.getBaseName(save_path)

        # print "saved"
        return True



    def loadDet(self, det, default_name="knife"):
        tmp_bboxes = []

        for items in list(det):
            if len(items) == 5:
                tmp_bbox = BBox(name=default_name, score=float(items[4]), xmin=float(items[0]), ymin=float(items[1]), xmax=float(items[2]), ymax=float(items[3]))
                tmp = tmp_bbox
            elif len(items) == 6:
                tmp_bbox = BBox(name=items[5], score=float(items[4]), xmin=float(items[0]), ymin=float(items[1]),
                                xmax=float(items[2]), ymax=float(items[3]))
            else: continue
            if not tmp_bbox is None:
                tmp_bboxes.append(tmp_bbox)
        if len(tmp_bboxes) > 0 : self.bboxes = tmp_bboxes

    def loadTXT(self, txt_path, default_name="knife"):
        if not osp.isfile(txt_path) :
            print " --- ERROR: Cannot Load TXT: ", txt_path
            return False
        fname, _ = osp.splitext(self.getBaseName(txt_path))
        lines = []

        with open(txt_path) as fr:
            lines = fr.readlines()

        tmp_bboxes = []
        for line in lines:
            items = line.strip().split(" ")
            tmp_bbox = None
            if len(items) == 5:
                tmp_bbox = BBox(name=default_name, score=float(items[4]), xmin=float(items[0]), ymin=float(items[1]), xmax=float(items[2]), ymax=float(items[3]))
            elif len(items) == 6:
                tmp_bbox = BBox(name=items[5], score=float(items[4]), xmin=float(items[0]), ymin=float(items[1]),
                                xmax=float(items[2]), ymax=float(items[3]))
            else: continue
            if not tmp_bbox is None: tmp_bboxes.append(tmp_bbox)

        #if len(tmp_bboxes) == 0: return False
        self.clearXML()
        self.bboxes = tmp_bboxes
        self.xml_name = fname + ".xml"
        return True

    def saveIMG(self, save_path="", save_dir=""):
        tmp_save_path = save_path
        if osp.isdir(save_dir): tmp_save_path = osp.join(save_dir, self.img_name)
        try:
            #print '==tmp_save_path==', tmp_save_path
            cv2.imwrite(tmp_save_path, self.img)
        except:
            print " --- ERROR: Cannot Save IMG: ", tmp_save_path
            return False
        self.img_path = tmp_save_path
        self.img_name = self.getBaseName(tmp_save_path)
        return True

    def saveIMGRoi(self, save_dir):
        if not osp.isdir(save_dir) or (self.img is None) or len(self.bboxes) == 0:
            return False
        for idx, item in enumerate(self.bboxes):
            roi_img = self.img[int(item.ymin):int(item.ymax), int(item.xmin):int(item.xmax)]
            fname, _ = osp.splitext(self.img_name)

            tmp_save_dir = save_dir + "/" + item.name
            if not osp.isdir(tmp_save_dir) : os.makedirs(tmp_save_dir)
            save_path = osp.join(tmp_save_dir, fname + "_roi" + str(idx) + "_" + item.name + ".jpg")

            try:
                #cv2.imshow("roi img", roi_img)
                #cv2.waitKey()
                cv2.imwrite(save_path, roi_img)
            except:
                print " --- ERROR: Cannot Save IMG ROI ", save_path
                continue

    def showIMG(self, save_dir = "", color = None,th=0.0):
        ret_val = -1
        rect_color = color
        if rect_color is None: rect_color = (random.uniform(120, 255), 255, random.uniform(128, 255))

        if not self.img is None :
            img = copy.deepcopy(self.img)

            try:
                cv2.rectangle(img, (self.orig_xmin, self.orig_ymin), (self.orig_xmax, self.orig_ymax), (0, 255, 0), 2)

                for item in self.bboxes:
                    if item.score >= th:
                        X = int((item.xmin + item.xmax) * 0.5)
                        Y = int((item.ymin + item.ymax) * 0.5)
                        if color==(0,255,0):
                            X = int(item.xmax)
                            Y = int(item.ymin)
                        if color==(0,0,255):
                            X = int(item.xmax)
                            Y = int(item.ymax)
                        cv2.line(img, (int(item.xmin), int(item.ymin)), (int(item.xmax), int(item.ymax)), rect_color, 1)
                        cv2.line(img, (int(item.xmax), int(item.ymin)), (int(item.xmin), int(item.ymax)), rect_color, 1)
                        cv2.rectangle(img, (int(item.xmin), int(item.ymin)), (int(item.xmax), int(item.ymax)), rect_color, 1)
                        cv2.putText(img, str(item.name) + " " + ("%.2f" % item.score), (X, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 1)
                        # cv2.putText(img, str(item.name) + " " + ("%.2f" % item.score), (int(item.xmax),int(item.ymin) ), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 1)

                if save_dir != "":
                    cv2.imwrite(osp.join(save_dir, self.img_name), img)
                else:
                    cv2.imshow(self.img_name, img)
                    ret_val = cv2.waitKey()
                    cv2.destroyWindow(self.img_name)
            except:
                print " --- ERROR: Connot Show Image!"
                return "ERR"
        return ret_val

    def maskROI(self, cls_name="", save_cls_name = []):
        #_, _, mean = mean_background(self.img, 5)
        del_cls = set([])
        for item in self.bboxes:
            if item.name == cls_name or not item.name in save_cls_name:

                # print "####"
                del_cls.add(item.name)
                roi = self.img[int(item.ymin):int(item.ymax), int(item.xmin):int(item.xmax)]
                for i in range(1, roi.shape[1]):
                    roi[:,i] = roi[:,0]
                #self.img[int(item.ymin):int(item.ymax), int(item.xmin):int(item.xmax)] = mean[0:3]
        self.bboxes = [x for x in self.bboxes if x.name in del_cls]

    def makeBorder(self, new_width, new_height):
        if self.img is None and osp.isfile(self.img_path): self.loadIMG(self.img_path)
        if self.img is None or not osp.isfile(self.img_path): return False

        img = copy.deepcopy(self.img)

        tmpw = img.shape[1]
        tmph = img.shape[0]

        tmp_bboxes = self.bboxes
        ratio = min(new_width / float(tmpw), new_height / float(tmph))
        if ratio < 1. :
            if len(tmp_bboxes) > 0:
                for idx in len(tmp_bboxes):
                    tmp_bboxes[idx].xmin *= ratio
                    tmp_bboxes[idx].xmax *= ratio
                    tmp_bboxes[idx].ymin *= ratio
                    tmp_bboxes[idx].ymax *= ratio

            try: img = cv2.resize(img, math.floor(tmpw * ratio), math.floor(tmph * ratio))
            except:
                print " --- ERROR: Resize Image: "
                return False

        tmpw = img.shape[1]
        tmph = img.shape[0]

        left = (new_width - tmpw) / 2
        right = new_width - tmpw - left
        top = (new_height - tmph) / 2
        bottom = new_height - tmph - top

        new_img = None
        try:
            new_img = cv2.copyMakeBorder(img, top, bottom, left, right, 1)
        except:
            print " --- ERROR: Cannot MakeBorder: "
            return False

        new_bboxes = []
        #item.img = new_img
        for bbox  in tmp_bboxes:
            new_bboxes.append(BBox(name=bbox.name, score=bbox.score, xmin=bbox.xmin+left, xmax=bbox.xmax+left, ymin=bbox.ymin+top, ymax=bbox.ymax+top))

        if not self.setIMG(new_img): return False
        self.bboxes = new_bboxes


        self.orig_xmin = left
        self.orig_ymin = top
        self.orig_xmax = left + tmpw - 1
        self.orig_ymax = top + tmph - 1

        return True
