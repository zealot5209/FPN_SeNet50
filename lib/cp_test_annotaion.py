import os
import shutil

#!/usr/bin/env python

# --------------------------------------------------------
# This script is used to create test annotations from gt
# Date:201803
# Author: YAN SONG

def del_file(dir):
    for i in os.listdir(dir):
        path_file = os.path.join(dir, i)
        #print 'path_file', path_file
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)


def cp_annots(old_dir, new_dir, name_list):
    if os.path.exists(new_dir):
        del_file(new_dir)
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    fr = None
    fr = open(name_list)
    array = fr.readlines()
    for line in array:
        line = line.strip() + '.xml'
        #line = line.strip() + '.jpg'
        oldname = old_dir + line
        newname = new_dir + line
        shutil.copyfile(oldname, newname)


if __name__ == "__main__":
    old_dir = '/home/yansong/FPN-git-unsky3/data/VOCdevkit2007/VOC2007/Annotations/'
    new_dir = '/home/yansong/FPN-git-unsky3/experiments/result/ResNet50/Test_annotations/'
    name_list = '/home/yansong/FPN-git-unsky3/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt'
    cp_annots(old_dir, new_dir, name_list)
