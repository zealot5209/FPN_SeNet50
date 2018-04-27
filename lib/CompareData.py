#coding: utf-8
import sys
import os
# import cv2
import os.path as osp
import hashlib
import time


def md5(fname) :
    '''计算文件的 MD5 码'''
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_jpg(img_dir):
    '''加载目录下的 .jpg 文件列表'''
    ret_list = []
    if not osp.isdir(img_dir) : return ret_list
    file_list = os.listdir(img_dir)
    for line in file_list:
        fname, fext = osp.splitext(line)
        if ".jpg" == fext : ret_list.append(line)
    return ret_list

def main(img_dir1, img_dir2) :
    '''找出两个目录下重复的 .jpg 文件'''
    if not osp.isdir(img_dir1) or not osp.isdir(img_dir2) :
        print "-- Unvalid image directory (%s) or xml directory (%s)!" % (img_dir1, img_dir2)
        return False

    list1 = load_jpg(img_dir1)
    list2 = load_jpg(img_dir2)

    same_counter = 0
    notsame_counter = 0

    for line in list1:
        image_path1 = osp.join(img_dir1, line)
        image_path2 = osp.join(img_dir2, line)
        if line in list2:
            fname, fext = osp.splitext(line)
            # if md5(image_path1) == md5(image_path2):
            # img1 = cv2.imread(image_path1)
            # img2 = cv2.imread(image_path2)

            same_counter += 1
            print fname

                # cv2.imshow("img1", img1)
                # cv2.imshow("img2", img2)
                #
                # cv2.waitKey(1)
            # else:
            #
            #     new_path = osp.join(img_dir1, "repir_" + fname + time.strftime("%Y%m%d", time.localtime()) + ".jpg")
            #     os.rename(image_path1, image_path2)
            #     notsame_counter += 1

    print "same counter: ", same_counter
    print "notsame counter: ", notsame_counter

    return True

if __name__ == '__main__' :
    argv = sys.argv
    argc = len(argv)

    if argc == 3 : main(argv[1], argv[2])
    else : print "usage: %s <image dir1> <image dir2>" % argv[0]
