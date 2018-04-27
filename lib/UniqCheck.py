#coding: utf-8
import sys
import os
# import cv2
import os.path as osp
import hashlib
import time
import shutil


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

def main(root_dir, pmd5=False) :
    '''找出两个目录下重复的 .jpg 文件'''

    name_dict = {}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            key = filename
            if pmd5 : key = md5(file_path)
            if name_dict.has_key(key) : name_dict[key].append(file_path)
            else: name_dict[key] = [file_path]
    print "File Num: %d", len(name_dict)

    ret_list = []
    for key, val in name_dict.items():
        if len(val) > 1 : ret_list.append(val)

    for item in ret_list:
        print " ** " + item[0]
        print " -- same as %d files -- " % (len(item) -1)
        for name in item[1:]:
            print "    ", name

    file_num = len(ret_list)

    if raw_input(" -- Delete Duplicated files ? (yes/no)") in ["yes", "YES"]:
        for idx, item in enumerate(ret_list):
            print " **%d/%d**" % (idx, file_num)
            for filepath in item[1:] :
                os.remove(file_path)
                print " -- file deleted: %s" % file_path


    print "Perfect!"

    return True

if __name__ == '__main__' :
    argv = sys.argv
    argc = len(argv)

    if argc == 2 : main(argv[1])
    elif argc == 3: main(argv[1], bool(argv[2]))
    else : print "usage: %s <root> <delete flag T/F>" % argv[0]
