import os
import sys
import os.path as osp
import BBoxXmlTool as bxt

def get_img_list(img_dir):
    file_list = []
    for root, dirs, files in os.walk(img_dir):
        if (root != img_dir):
            continue
        for ele_f in files:
            if ele_f.endswith('.jpg'):
                file_list.append(ele_f+'\n')

    return file_list

def display(gt_img, result_img, th, iou_th):

    T=0
    result_img.bboxes = [x for x in result_img.bboxes if x.score >= th]
    i=0
    j=0
    k=0
    ## 01,02,03,04
    if len(gt_img.bboxes) >0 and len(gt_img.bboxes) == len(result_img.bboxes):

        for gt_bbox in gt_img.bboxes:
            for result_bbox in result_img.bboxes:
                if calc_iou(gt_bbox, result_bbox) > iou_th and gt_bbox.name==result_bbox.name:
                    j = j + 1
                    T = 2
                elif calc_iou(gt_bbox, result_bbox) > iou_th and gt_bbox.name!=result_bbox.name:
                    i = i + 1
                elif calc_iou(gt_bbox, result_bbox) == 0.0:
                    i = i + 1
        if j > 0 and i > 0:
            T = 3
        if j == max(len(gt_img.bboxes), len(result_img.bboxes)):
            T = 1
        if i == len(gt_img.bboxes) * len(result_img.bboxes):
            T = 4
        return T

    ##02
    if len(gt_img.bboxes) > 0 and len(result_img.bboxes)==0:
        T=2
        return T

    ##01,02,03,04
    if len(gt_img.bboxes) > 1 and len(result_img.bboxes) == 1:
        for gt_bbox in gt_img.bboxes:
            for result_bbox in result_img.bboxes:
                if calc_iou(gt_bbox, result_bbox) > iou_th and gt_bbox.name==result_bbox.name:
                    j = j + 1
                    T = 2
                elif calc_iou(gt_bbox, result_bbox) > iou_th and gt_bbox.name!=result_bbox.name:
                    i = i + 1

        if j > 0 and i > 0:
            T=3
        if j == max(len(gt_img.bboxes),len(result_img.bboxes)):
            T =1
        if i == len(gt_img.bboxes)*len(result_img.bboxes):
            T = 4
        return T

    ## 02,03,04
    if len(gt_img.bboxes) > 0 and  len(gt_img.bboxes) > len(result_img.bboxes) and len(result_img.bboxes) !=0:

        for gt_bbox in gt_img.bboxes:
            for result_bbox in result_img.bboxes:
                if calc_iou(gt_bbox, result_bbox) > iou_th and gt_bbox.name==result_bbox.name:
                    j = j + 1
                    T = 2
                elif calc_iou(gt_bbox, result_bbox) > iou_th and gt_bbox.name!=result_bbox.name:
                    i = i + 1
                # elif calc_iou(gt_bbox, result_bbox) < iou_th and calc_iou(gt_bbox, result_bbox) != 0.0:
                #     T = 0
                #     return T
                elif calc_iou(gt_bbox, result_bbox) == 0.0 :
                    i = i + 1
        if j == max(len(gt_img.bboxes),len(result_img.bboxes)):
            T =1
        if i == len(gt_img.bboxes)*len(result_img.bboxes):
            T = 4
        if j > 0 and i > 0:
            T=3
        return T
    ## 01,03 04
    if len(gt_img.bboxes) > 0 and len(gt_img.bboxes) < len(result_img.bboxes):
        for gt_bbox in gt_img.bboxes:
            for result_bbox in result_img.bboxes:
                if calc_iou(gt_bbox, result_bbox) > iou_th and gt_bbox.name==result_bbox.name:
                    j = j + 1
                    T = 2
                elif calc_iou(gt_bbox, result_bbox) > iou_th and gt_bbox.name!=result_bbox.name:
                    i = i + 1
                elif calc_iou(gt_bbox, result_bbox) < iou_th and gt_bbox.name==result_bbox.name and calc_iou(gt_bbox, result_bbox) != iou_th:
                    T = 0
                    return T
                elif calc_iou(gt_bbox, result_bbox) < iou_th and gt_bbox.name!=result_bbox.name and calc_iou(gt_bbox, result_bbox) != iou_th:
                    T = 3
                    return T
                elif calc_iou(gt_bbox, result_bbox) == 0.0:
                    i = i + 1
        if j > 0 and i > 0:
            T=3
        if j == max(len(gt_img.bboxes),len(result_img.bboxes)):
            T =1
        if i == len(gt_img.bboxes)*len(result_img.bboxes):
            T = 4
        return T


    ## 04
    if len(gt_img.bboxes) == 0 and len(result_img.bboxes) > 0:
        T = 4
        return T

    ## 05
    if 0 == len(gt_img.bboxes) and 0 == len(result_img.bboxes) :
        T = 5
        return T

def calc_iou(bbox1, bbox2) :
    sw = max(min(bbox1.xmax, bbox2.xmax) - max(bbox1.xmin, bbox2.xmin), 0.)
    sh = max(min(bbox1.ymax, bbox2.ymax) - max(bbox1.ymin, bbox2.ymin), 0.)

    area1 = (bbox1.xmax - bbox1.xmin) * (bbox1.ymax - bbox1.ymin)
    area2 = (bbox2.xmax - bbox2.xmin) * (bbox2.ymax - bbox2.ymin)

    return sw * sh / float(area1 + area2 - sw * sh)

def calc_param(img_path,gt_xml_path,result_xml_path, save_dir ,fname,gt_list, result_list, th = 0.5, iou_th = 0.5):
    T=0

    T=display(gt_list, result_list, th, iou_th)

    for i in range(1,6,1):
        if i == T:
            BGR=(0,255,0)
            tmp_img = bxt.IMGBBox(img_path, gt_xml_path)
            tmp_img.showIMG(save_dir+'\\'+str(i)+'\\',BGR,th)

            BGR = (0, 0, 255)
            tmp_img = bxt.IMGBBox(save_dir+'\\'+str(i)+'\\'+fname+'.jpg',result_xml_path)
            tmp_img.showIMG(save_dir+'\\'+str(i)+'\\',BGR,th)


def eval(all_img, gt_dir, result_dir, save_dir , th, iou_th):
    if not osp.isdir(gt_dir) or not osp.isdir(result_dir) :
        print "Import DIR"
        print gt_dir
        print result_dir
        return None


    lines = get_img_list(all_img)
    type(lines)
    # with open(save_dir+"\\1.txt", "w") as f:
    #     for i in lines:
    #         f.write(i)

    if not os.path.isdir(save_dir + '\\5\\'):
        # os.mkdir(save_dir + '\\0\\')
        os.mkdir(save_dir + '\\1\\')
        os.mkdir(save_dir + '\\2\\')
        os.mkdir(save_dir + '\\3\\')
        os.mkdir(save_dir + '\\4\\')
        os.mkdir(save_dir + '\\5\\')


    image_num = len(lines)

    for idx, xml_name in enumerate(lines):
        xml_name, _ = osp.splitext(xml_name)
        # if idx > 400 : break

        fname = xml_name.strip()
        print "[%d/%d] %s" % (idx, image_num, xml_name)

        img_path=osp.join(all_img, fname + ".jpg")

        gt_xml_path = osp.join(gt_dir, fname + ".xml")
        result_xml_path = osp.join(result_dir, fname + ".xml")

        gt_img = bxt.IMGBBox(xml_path= gt_xml_path)
        result_img = bxt.IMGBBox(xml_path=result_xml_path)

        calc_param(img_path,gt_xml_path,result_xml_path, save_dir, fname,gt_img, result_img, th, iou_th)


if __name__ == '__main__':
    argv = sys.argv
    argc = len(argv)

    # eva_pkl()

    all_img=r'D:\D\yisuo_knife_result_180227\yisuo_knife_result_180227\all_img'
    gt_dir=r'D:\D\yisuo_knife_result_180227\yisuo_knife_result_180227\xml'
    result_dir=r'D:\D\yisuo_knife_result_180227\yisuo_knife_result_180227\all_xml'
    save_dir=r'D:\D\yisuo_knife_result_180227\yisuo_knife_result_180227\1'

    th=0.5
    iou_th=0.5

    eval(all_img, gt_dir, result_dir, save_dir, th, iou_th)

    # #if argc == 3 : eval(argv[1], argv[2])
    # #if argc == 5 : eval(argv[1], argv[2], argv[3], argv[4])
    # if argc == 7 : eval(argv[1], argv[2], argv[3], argv[4], argv[5], argv[6])
    # else: print "usage: %s imagelist  gt_ann test_ann <optional resut_save_dir>" % argv[0]
