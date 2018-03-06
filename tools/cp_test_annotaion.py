import os
import shutil

name_list = '/home/yansong/FPN-git-unsky2/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt'
old_dir = '/home/yansong/FPN-git-unsky2/data/VOCdevkit2007/VOC2007/Annotations/'
new_dir = '/home/yansong/FPN-git-unsky2/data/VOCdevkit2007/VOC2007/Test_annotations/'
# old_dir = '/home/yansong/FPN-git-unsky2/data/VOCdevkit2007/VOC2007/JPEGImages/'
# new_dir = '/home/yansong/FPN-git-unsky2/data/demo/'


def del_file(dir):
    for i in os.listdir(dir):
        path_file = os.path.join(dir, i)
        print 'path_file', path_file
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)


if __name__ == '__main__':
    del_file(new_dir)
    fr = None
    fr = open(name_list)
    array = fr.readlines()
    for line in array:
        line = line.strip() + '.xml'
        #line = line.strip() + '.jpg'
        oldname = old_dir + line
        newname = new_dir + line
        shutil.copyfile(oldname, newname)
