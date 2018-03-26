# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr
import cv2
from fast_rcnn.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
import random
import os
import math

def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

    # Get the input image blob, formatted for caffe
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    blobs = {'data': im_blob}

    if cfg.TRAIN.HAS_RPN:
        assert len(im_scales) == 1, "Single batch only"
        assert len(roidb) == 1, "Single batch only"
        # gt boxes: (x1, y1, x2, y2, cls)
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
        gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
        gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
        blobs['gt_boxes'] = gt_boxes
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)
    else: # not using RPN
        # Now, build the region of interest and label blobs
        rois_blob = np.zeros((0, 5), dtype=np.float32)
        labels_blob = np.zeros((0), dtype=np.float32)
        bbox_targets_blob = np.zeros((0, 4 * num_classes), dtype=np.float32)
        bbox_inside_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)
        # all_overlaps = []
        for im_i in xrange(num_images):
            labels, overlaps, im_rois, bbox_targets, bbox_inside_weights \
                = _sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image,
                               num_classes)

            # Add to RoIs blob
            rois = _project_im_rois(im_rois, im_scales[im_i])
            batch_ind = im_i * np.ones((rois.shape[0], 1))
            rois_blob_this_image = np.hstack((batch_ind, rois))
            rois_blob = np.vstack((rois_blob, rois_blob_this_image))

            # Add to labels, bbox targets, and bbox loss blobs
            labels_blob = np.hstack((labels_blob, labels))
            bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
            bbox_inside_blob = np.vstack((bbox_inside_blob, bbox_inside_weights))
            # all_overlaps = np.hstack((all_overlaps, overlaps))

        # For debug visualizations
        # _vis_minibatch(im_blob, rois_blob, labels_blob, all_overlaps)

        blobs['rois'] = rois_blob
        blobs['labels'] = labels_blob

        if cfg.TRAIN.BBOX_REG:
            blobs['bbox_targets'] = bbox_targets_blob
            blobs['bbox_inside_weights'] = bbox_inside_blob
            blobs['bbox_outside_weights'] = \
                np.array(bbox_inside_blob > 0).astype(np.float32)

    return blobs

def _sample_rois(roidb, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # label = class RoI has max overlap with
    labels = roidb['max_classes']
    overlaps = roidb['max_overlaps']
    rois = roidb['boxes']

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(
                fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                        bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(
                bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    overlaps = overlaps[keep_inds]
    rois = rois[keep_inds]

    bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(
            roidb['bbox_targets'][keep_inds, :], num_classes)

    return labels, overlaps, rois, bbox_targets, bbox_inside_weights

def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    import copy
    roidb = copy.deepcopy(roidb)
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    rbbs = []
    processed_roidb = []
    cfg.TRAIN.IMAGES_LIST = []


    for i in xrange(num_images):
        print 'i=', i

        roi_rec = roidb[i].copy()
        assert os.path.exists(roi_rec['image']), '%s does not exist'.format(roi_rec['image'])
        im = cv2.imread(roidb[i]['image'])


        print 'flipped:', roidb[i]['flipped']
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        im_COLOR = copy.deepcopy(im)
        im_RANDOM_PAD = copy.deepcopy(im)
        im_RANDOM_CLIP = copy.deepcopy(im)
        im_STRETCH = copy.deepcopy(im)
        im_SQUARE = copy.deepcopy(im)
        im_ROTATE = copy.deepcopy(im)
        im_VIBO = copy.deepcopy(im)


        roi_rec_COLOR = copy.deepcopy(roi_rec)
        roi_rec_RANDOM_PAD = copy.deepcopy(roi_rec)
        roi_rec_RANDOM_CLIP = copy.deepcopy(roi_rec)
        roi_rec_STRETCH = copy.deepcopy(roi_rec)
        roi_rec_SQUARE = copy.deepcopy(roi_rec)
        roi_rec_ROTATE = copy.deepcopy(roi_rec)
        roi_rec_VIBO = copy.deepcopy(roi_rec)

        ### DATAUG

        if cfg.TRAIN.DATAUG:
            ### Display
            if cfg.TRAIN.DATAUG_DEBUG:
                display_datum(roi_rec, im, "orig")
                # cv2.waitKey()

            # display_datum(roi_rec, im, "orig")
            # #roi_rec, im = random_pad(roi_rec, im, config.TRAIN.DATAUG_RANDOM_PAD_RATIO)
            # #roi_rec, im = random_clip(roi_rec, im, config.TRAIN.DATAUG_RANDOM_CLIP_RATIO)
            #
            # # x_ratio = 1.3
            # # y_ratio = 0.8
            # # roi_rec, im = stretch(roi_rec, im, x_ratio, y_ratio)
            #
            # roi_rec, im = vibration(roi_rec, im,
            #                         random.uniform(-config.TRAIN.DATAUG_VIBO_RATIO, config.TRAIN.DATAUG_VIBO_RATIO))
            #
            # display_datum(roi_rec, im, "padded")
            # cv2.waitKey()

            # color
            if cfg.TRAIN.DATAUG_COLOR:
                if True:
                    # if random.sample([0, 1, 2, 3, 4, 5, 6, 7, 8], 1)[0] == 0:
                    #     im = convert_color_ycrcb(im, y_offset=random.uniform(-50, 50), cr_offset=random.uniform(-10, 10), cb_offset=random.uniform(-10, 10))
                    # elif seed[0] == 1:
                    im_COLOR = convert_color_hsv(im_COLOR, h_offset=random.uniform(-3, 6), s_offset=random.uniform(-10, 8),
                                           v_offset=random.uniform(-10, 30))

                    if roi_rec_COLOR is not None:
                        for j in xrange(len(roi_rec_COLOR['boxes'])):
                            [x1, y1, x2, y2] = [int(x) for x in roi_rec_COLOR['boxes'][j, :]]
                            # cv2.rectangle(disp, (x1, y1), (x2, y2), (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)), 2)
                            cv2.rectangle(im_COLOR, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        cv2.imshow('hsv', im_COLOR)
                        cv2.waitKey()

                    # im = convert_color_hsv(im, h_offset=0, s_offset=0, v_offset=-10)
                # elif random.sample([0, 1, 2, 3, 4], 1)[0] == 0:
                #     im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                #     im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)



            ### stretch
            if cfg.TRAIN.DATAUG_STRETCH and random.sample([0, 1, 2, 3], 1)[0] == 0:
                x_ratio = 1. + random.uniform(-cfg.TRAIN.DATAUG_STRETCH_XRATIO,
                                              cfg.TRAIN.DATAUG_STRETCH_XRATIO)
                y_ratio = 1. + random.uniform(-cfg.TRAIN.DATAUG_STRETCH_YRATIO,
                                              cfg.TRAIN.DATAUG_STRETCH_YRATIO)
                roi_rec_STRETCH, im_STRETCH = stretch(roi_rec_STRETCH, im_STRETCH, x_ratio, y_ratio)

                for j in xrange(len(roi_rec_STRETCH['boxes'])):
                    [x1, y1, x2, y2] = [int(x) for x in roi_rec_STRETCH['boxes'][j, :]]
                    # cv2.rectangle(disp, (x1, y1), (x2, y2), (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)), 2)
                    cv2.rectangle(im_STRETCH, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imshow('stretch', im_STRETCH)
                cv2.waitKey()

            #pad to square
            if cfg.TRAIN.DATAUG_SQUARE:
                if cfg.TRAIN.DATAUG_SQUARE_MAXSIDE > 0 and cfg.TRAIN.DATAUG_SQUARE_MAXSIDE < 2048:
                    max_side = cfg.TRAIN.DATAUG_SQUARE_MAXSIDE + random.randrange(
                        -cfg.TRAIN.DATAUG_SQUARE_MAXSIDE_OFFSET, cfg.TRAIN.DATAUG_SQUARE_MAXSIDE_OFFSET)
                    roi_rec_SQUARE, im_SQUARE = make_border(roi_rec_SQUARE, im_SQUARE, max_side, cfg.TRAIN.DATAUG_SQUARE_RANDOM_ALIGN)
                    for j in xrange(len(roi_rec_SQUARE['boxes'])):
                        [x1, y1, x2, y2] = [int(x) for x in roi_rec_SQUARE['boxes'][j, :]]
                        # cv2.rectangle(disp, (x1, y1), (x2, y2), (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)), 2)
                        cv2.rectangle(im_SQUARE, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.imshow('pad to square_1', im_SQUARE)
                    cv2.waitKey()
                else:
                    roi_rec_SQUARE, im_SQUARE = pad_to_square(im_SQUARE)
                    for j in xrange(len(roi_rec_SQUARE['boxes'])):
                        [x1, y1, x2, y2] = [int(x) for x in roi_rec_SQUARE['boxes'][j, :]]
                        # cv2.rectangle(disp, (x1, y1), (x2, y2), (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)), 2)
                        cv2.rectangle(im_SQUARE, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.imshow('pad to square_2', im_SQUARE)
                    cv2.waitKey()

            ## rotate
            if cfg.TRAIN.DATAUG_ROTATE:
                seed = random.sample([0, 1, 2, 3], 1)
                roi_rec_ROTATE, im_ROTATE = rotate_4degree(roi_rec_ROTATE, im_ROTATE, seed[0])
                for j in xrange(len(roi_rec_ROTATE['boxes'])):
                    [x1, y1, x2, y2] = [int(x) for x in roi_rec_ROTATE['boxes'][j, :]]
                    # cv2.rectangle(disp, (x1, y1), (x2, y2), (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)), 2)
                    cv2.rectangle(im_ROTATE, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imshow('rotate', im_ROTATE)
                cv2.waitKey()

            ### viboration
            if cfg.TRAIN.DATAUG_VIBO:
                roi_rec_VIBO, im_VIBO = vibration(roi_rec_VIBO, im_VIBO, random.uniform(-cfg.TRAIN.DATAUG_VIBO_RATIO,
                                                                    cfg.TRAIN.DATAUG_VIBO_RATIO))

                for j in xrange(len(roi_rec_VIBO['boxes'])):
                    [x1, y1, x2, y2] = [int(x) for x in roi_rec_VIBO['boxes'][j, :]]
                    # cv2.rectangle(disp, (x1, y1), (x2, y2), (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)), 2)
                    cv2.rectangle(im_VIBO, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imshow('vibration', im_VIBO)
                cv2.waitKey()


            ### random pad
            pad_flag = False
            if cfg.TRAIN.DATAUG_RANDOM_PAD and random.sample([0, 1, 2], 1)[0] == 0:
            #if True:

                roi_rec_RANDOM_PAD, im_RANDOM_PAD = random_pad(roi_rec_RANDOM_PAD, im_RANDOM_PAD, cfg.TRAIN.DATAUG_RANDOM_PAD_RATIO)

                pad_flag = True

                for j in xrange(len(roi_rec_RANDOM_PAD['boxes'])):
                    [x1, y1, x2, y2] = [int(x) for x in roi_rec_RANDOM_PAD['boxes'][j, :]]
                    # cv2.rectangle(disp, (x1, y1), (x2, y2), (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)), 2)
                    cv2.rectangle(im_RANDOM_PAD, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imshow('random_pad', im_RANDOM_PAD)
                cv2.waitKey()
            #
            ## random clip
            if cfg.TRAIN.DATAUG_RANDOM_CLIP and not pad_flag and random.sample([0, 1, 2], 1)[0] == 0:
            #if True:
                roi_rec_RANDOM_CLIP, im_RANDOM_CLIP = random_clip(roi_rec_RANDOM_CLIP, im_RANDOM_CLIP, cfg.TRAIN.DATAUG_RANDOM_CLIP_RATIO)
                for j in xrange(len(roi_rec_RANDOM_CLIP['boxes'])):
                    [x1, y1, x2, y2] = [int(x) for x in roi_rec_RANDOM_CLIP['boxes'][j, :]]
                    # cv2.rectangle(disp, (x1, y1), (x2, y2), (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)), 2)
                    cv2.rectangle(im_RANDOM_CLIP, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imshow('random_clip', im_RANDOM_CLIP)
                cv2.waitKey()



            # new_boxes = np.empty((0, 4), dtype=np.float32)
            # for j in xrange(len(roi_rec['boxes'])):
            #     [x1, y1, x2, y2] = [int(x) for x in roi_rec['boxes'][j, :]]
            #     new_boxes = np.vstack((new_boxes, np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])))
            #
            # roi_rec['boxes'] = new_boxes

            ### Display
            # if cfg.TRAIN.DATAUG_DEBUG:
            #     display_datum(roi_rec, im, "changed")
            #     cv2.waitKey()
            #
        ### !DATAUG


        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                       cfg.TRAIN.MAX_SIZE,cfg.TRAIN.IMAGE_STRIDE)


        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)
    

    return blob, im_scales

def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights

def _vis_minibatch(im_blob, rois_blob, labels_blob, overlaps):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    for i in xrange(rois_blob.shape[0]):
        rois = rois_blob[i, :]
        im_ind = rois[0]
        roi = rois[1:]
        im = im_blob[im_ind, :, :, :].transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        cls = labels_blob[i]
        plt.imshow(im)
        print 'class: ', cls, ' overlap: ', overlaps[i]
        plt.gca().add_patch(
            plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                          roi[3] - roi[1], fill=False,
                          edgecolor='r', linewidth=3)
            )
        plt.show()



def display_datum(roi_rec, im, name="img"):
    cv2.imshow('img_without_boxs', im)
    disp = im.copy()
    for j in xrange(len(roi_rec['boxes'])):
        [x1, y1, x2, y2] = [int(x) for x in roi_rec['boxes'][j, :]]
        #cv2.rectangle(disp, (x1, y1), (x2, y2), (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)), 2)
        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow(name, disp)
    cv2.waitKey()


def pad_br(im, stride=32, border_type=0):
    if cfg.TRAIN.DATAUG_DEBUG: print "pad_br"
    if im is None: return im
    im_height, im_width = im.shape[:2]
    pad_r, pad_b = 0, 0
    if im_width % int(stride) != 0 : pad_r = int(math.ceil(im_width / float(stride))) * 32 - im_width
    if im_height % int(stride) != 0 : pad_b = int(math.ceil(im_width / float(stride))) * 32 - im_height


    if pad_b > 0 or pad_r > 0:
            cv2.copyMakeBorder(im, 0, pad_b, 0, pad_r, border_type)
            for j in xrange(len(roi_rec['boxes'])):
                [x1, y1, x2, y2] = [int(x) for x in roi_rec['boxes'][j, :]]
                # cv2.rectangle(disp, (x1, y1), (x2, y2), (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)), 2)
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)

    #else: return im
    return im

def convert_color_hsv(im, h_offset = 0., s_offset = 0., v_offset = 0.):
    if cfg.TRAIN.DATAUG_DEBUG: print "convert_color_hsv"
    if im is None: return im
    hue = cv2.cvtColor(im, cv2.COLOR_BGR2HSV_FULL)
    hue_plane = cv2.split(hue)
    hue_plane[0] = (hue_plane[0].astype(np.float) + h_offset).clip(0, 255).astype(np.uint8)
    hue_plane[1] = (hue_plane[1].astype(np.float) + s_offset).clip(0, 255).astype(np.uint8)
    hue_plane[2] = (hue_plane[2].astype(np.float) + v_offset).clip(0, 255).astype(np.uint8)
    hue = cv2.merge(hue_plane)
    return cv2.cvtColor(hue, cv2.COLOR_HSV2BGR_FULL)

def convert_color_ycrcb(im, y_offset = 0., cr_offset = 0., cb_offset = 0.):
    if im is None: return im
    hue = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
    hue_plane = cv2.split(hue)
    hue_plane[0] = (hue_plane[0].astype(np.float) + y_offset).clip(0, 255).astype(np.uint8)
    hue_plane[1] = (hue_plane[1].astype(np.float) + cr_offset).clip(0, 255).astype(np.uint8)
    hue_plane[2] = (hue_plane[2].astype(np.float) + cb_offset).clip(0, 255).astype(np.uint8)
    hue = cv2.merge(hue_plane)
    return cv2.cvtColor(hue, cv2.COLOR_YCrCb2BGR)

# def pad_to_square(im, border_type=0):
#     if cfg.TRAIN.DATAUG_DEBUG: print "pad_to_square"
#
#     if im is None: return im
#     im_height, im_width = im.shape[:2]
#     if im_height == im_width :
#         for j in xrange(len(roi_rec['boxes'])):
#             [x1, y1, x2, y2] = [int(x) for x in roi_rec['boxes'][j, :]]
#             # cv2.rectangle(disp, (x1, y1), (x2, y2), (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)), 2)
#             cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.imshow('pad_to_square', im)
#         return im
#     else :
#         max_side = max(im_width, im_height)
#         top = (max_side - im_height) / 2
#         bottom = max_side - im_height - top
#         left = (max_side - im_width)  / 2
#         right = max_side - im_width - left
#         #return cv2.copyMakeBorder(im, top, bottom, left, right, border_type)
#         cv2.copyMakeBorder(im, top, bottom, left, right, border_type)
#         for j in xrange(len(roi_rec['boxes'])):
#             [x1, y1, x2, y2] = [int(x) for x in roi_rec['boxes'][j, :]]
#             # cv2.rectangle(disp, (x1, y1), (x2, y2), (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)), 2)
#             cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.imshow('pad_to_square', im)
#         return im

def pad_to_square(im, border_type=0):
    if cfg.TRAIN.DATAUG_DEBUG: print "pad_to_square"

    if im is None: return im
    im_height, im_width = im.shape[:2]
    if im_height == im_width : return im
    else :
        max_side = max(im_width, im_height)
        top = (max_side - im_height) / 2
        bottom = max_side - im_height - top
        left = (max_side - im_width)  / 2
        right = max_side - im_width - left
        return cv2.copyMakeBorder(im, top, bottom, left, right, border_type)


def random_pad(roi_rec_RANDOM_PAD, im_RANDOM_PAD, ratio = 0.1, border_type=0):
    #roi_rec_RANDOM_PAD_cp = roi_rec_RANDOM_PAD.copy()
    #im_RANDOM_PAD_cp = im_RANDOM_PAD.copy()
    if cfg.TRAIN.DATAUG_DEBUG: print "random_pad"
    if im_RANDOM_PAD is None or ratio <= 0: return roi_rec_RANDOM_PAD, im_RANDOM_PAD
    im_height, im_width = im_RANDOM_PAD.shape[:2]

    top = random.randrange(0, int(im_height * ratio))
    bottom = random.randrange(0, int(im_height * ratio))
    left = random.randrange(0, int(im_width * ratio))
    right = random.randrange(0, int(im_width * ratio))

    #print 'oi_rec.dtype', roi_rec['boxes'].dtype
    #roi_rec['boxes'] += np.array([left, top, left, top], dtype=roi_rec['boxes'].dtype)
    roi_rec_RANDOM_PAD['boxes'] += np.array([left, top, left, top]).astype(np.uint16)

    im_RANDOM_PAD = cv2.copyMakeBorder(im_RANDOM_PAD, top, bottom, left, right, border_type)
    return roi_rec_RANDOM_PAD, im_RANDOM_PAD


def random_clip(roi_rec, im, ratio = 1.):
    if cfg.TRAIN.DATAUG_DEBUG: print "random_clip"
    if im is None or len(roi_rec['boxes']) <= 0: return roi_rec, im
    im_height, im_width = im.shape[:2]

    rxmin = min(roi_rec['boxes'][:,0])
    xmin = 0 if int(rxmin * ratio) <= 1 else random.randrange(0, int(rxmin*ratio))

    rymin = min(roi_rec['boxes'][:,1])
    ymin = 0 if int(rymin * ratio) <= 1 else random.randrange(0, int(rymin*ratio))

    rxmax = max(roi_rec['boxes'][:,2])
    xmax = im_width if int((im_width - rxmax) * ratio) <= 1 else ( im_width - random.randrange(0, int((im_width - rxmax) * ratio)) )

    rymax = max(roi_rec['boxes'][:,3])
    ymax = im_height if int((im_height - rymax) * ratio) <= 1 else ( im_height - random.randrange(0, int((im_height - rymax) * ratio)) )

    im = im[ymin:ymax,xmin:xmax]

    #roi_rec['boxes'] -= np.array([xmin, ymin, xmin, ymin], dtype=roi_rec['boxes'].dtype)

    roi_rec['boxes'] -= np.array([xmin, ymin, xmin, ymin]).astype(np.uint16)
    return roi_rec, im


def stretch(roi_rec, im, ratio_x=1., ratio_y=1.):
    if cfg.TRAIN.DATAUG_DEBUG: print "stretch"
    if im is None or ratio_x <= 0 or ratio_y <= 0: return roi_rec, im
    im = cv2.resize(im, dsize=None, fx=ratio_x, fy=ratio_y)
    roi_rec['boxes'] = (roi_rec['boxes'] * np.array([ratio_x, ratio_y, ratio_x, ratio_y])).astype(np.uint32)

    return roi_rec, im

def resize_maxside(roi_rec, im, max_side = 1024.):
    if cfg.TRAIN.DATAUG_DEBUG: print "resize_maxside"
    if im is None : return roi_rec, im
    im_height, im_width = im.shape[:2]
    ratio = float(max_side) / max(im_width, im_height)
    return stretch(roi_rec, im, ratio, ratio)

def vibration(roi_rec, im, rotd=0.):
    if cfg.TRAIN.DATAUG_DEBUG: print "vibration"
    if im is None: return roi_rec, im
    im_height, im_width = im.shape[:2]

    M = cv2.getRotationMatrix2D((im_width / 2, im_height / 2), rotd, 1)
    im = cv2.warpAffine(im, M, (im_width, im_height), None, cv2.INTER_LINEAR, 0)
    rotate_boxes = np.empty((0, 4), dtype=np.float32)
    new_classes = []
    for j in xrange(len(roi_rec['boxes'])):
        [x1, y1, x2, y2] = [int(x) for x in roi_rec['boxes'][j, :]]
        new_pt1 = np.dot(M, np.array([x1, y1, 1]).transpose()).astype(np.int32).transpose()
        new_pt2 = np.dot(M, np.array([x2, y2, 1]).transpose()).astype(np.int32).transpose()
        new_pt3 = np.dot(M, np.array([x1, y2, 1]).transpose()).astype(np.int32).transpose()
        new_pt4 = np.dot(M, np.array([x2, y1, 1]).transpose()).astype(np.int32).transpose()
        rect_pts = np.array([[new_pt1, new_pt2, new_pt3, new_pt4]])
        x, y, w, h = cv2.boundingRect(rect_pts)
        new_rect = np.array([x, y, x+w-1, y+h-1])

        new_rect[new_rect < 0] = 0
        new_rect[0] = new_rect[0] if new_rect[0] < im_width else im_width
        new_rect[2] = new_rect[2] if new_rect[2] < im_width else im_width
        new_rect[1] = new_rect[1] if new_rect[1] < im_height else im_height
        new_rect[3] = new_rect[3] if new_rect[3] < im_height else im_height

        if new_rect[0] >= new_rect[2] or new_rect[1] >= new_rect[3] : continue

        new_classes.append(roi_rec['gt_classes'][j])
        rotate_boxes = np.vstack((rotate_boxes, new_rect))
    roi_rec['boxes'] = rotate_boxes
    roi_rec['gt_classes'] = np.array(new_classes)

    return roi_rec, im

# print cv2.ROTATE_90_CLOCKWISE 1
# print cv2.ROTATE_90_COUNTERCLOCKWISE 3
# print cv2.ROTATE_180 2
def rotate_4degree(roi_rec, im, rotd=0.):
    if cfg.TRAIN.DATAUG_DEBUG: print "rotate_4degree"
    if im is None: return roi_rec, im
    im_height, im_width = im.shape[:2]

    # disp = im.copy()
    # for j in xrange(len(roi_rec['boxes'])):
    #     [x1, y1, x2, y2] = [int(x) for x in roi_rec['boxes'][j, :]]
    #     cv2.rectangle(disp, (x1, y1), (x2, y2),
    #                   (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)), 2)
    # cv2.imshow("orig", disp)

    if rotd == 1:
        im = cv2.transpose(im)
        im = cv2.flip(im, 1)
        roi_rec['boxes'] = roi_rec['boxes'][:,[1,0,3,2]] * np.array([-1, 1, -1, 1]) + np.array([im_height, 0, im_height, 0])
        # for j in xrange(len(roi_rec['boxes'])):
        #     [x1, y1, x2, y2] = [int(x) for x in roi_rec['boxes'][j, :]]
        #     # cv2.rectangle(disp, (x1, y1), (x2, y2), (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)), 2)
        #     cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.imshow('rotate_4degree_1', im)
    elif rotd == 2:
        im = cv2.flip(im, 0)
        im = cv2.flip(im, 1)
        roi_rec['boxes'] = roi_rec['boxes'] * np.array([-1, -1, -1, -1]) + np.array([im_width, im_height, im_width, im_height])
        # for j in xrange(len(roi_rec['boxes'])):
        #     [x1, y1, x2, y2] = [int(x) for x in roi_rec['boxes'][j, :]]
        #     # cv2.rectangle(disp, (x1, y1), (x2, y2), (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)), 2)
        #     cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.imshow('rotate_4degree_2', im)
    elif rotd == 3:
        im = cv2.transpose(im)
        im = cv2.flip(im, 0)
        roi_rec['boxes'] = roi_rec['boxes'][:, [1, 0, 3, 2]] * np.array([1, -1, 1, -1]) + np.array([0, im_width, 0, im_width])
        # for j in xrange(len(roi_rec['boxes'])):
        #     [x1, y1, x2, y2] = [int(x) for x in roi_rec['boxes'][j, :]]
        #     # cv2.rectangle(disp, (x1, y1), (x2, y2), (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)), 2)
        #     cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.imshow('rotate_4degree_3', im)
    new_boxes = np.empty((0, 4), dtype=np.float32)
    for j in xrange(len(roi_rec['boxes'])):
        [x1, y1, x2, y2] = [int(x) for x in roi_rec['boxes'][j, :]]
        new_boxes = np.vstack((new_boxes, np.array([min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)])))

    roi_rec['boxes'] = new_boxes

    # disp = im.copy()
    # for j in xrange(len(roi_rec['boxes'])):
    #     [x1, y1, x2, y2] = [int(x) for x in roi_rec['boxes'][j, :]]
    #     cv2.rectangle(disp, (x1, y1), (x2, y2),
    #                   (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)), 2)
    # cv2.imshow("rotated", disp)
    # cv2.waitKey()



    return roi_rec, im

def make_border(roi_rec, im, max_side=1024, random_align=False, border_type=0):
    if cfg.TRAIN.DATAUG_DEBUG: print "make_border"
    if im is None : return roi_rec, im
    im_height, im_width = im.shape[:2]
    if max(im_height, im_width) > max_side : max_side = max(im_height, im_width)

    top, bottom, left, right = 0, 0, 0, 0
    if random_align :
        top = random.randrange(0, (max_side - im_height)) if (max_side - im_height) > 0 else 0
        left = random.randrange(0, (max_side - im_width)) if (max_side - im_width) > 0 else 0
    else:
        top = (max_side - im_height) / 2
        left = (max_side - im_width) / 2

    bottom = max_side - im_height - top
    right = max_side - im_width - left

    im = cv2.copyMakeBorder(im, top, bottom, left, right, border_type)
    roi_rec['boxes'] = (roi_rec['boxes'] + np.array([left, top, left, top])).astype(np.uint32)

    return roi_rec, im

def get_image_filenames(filenames):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(filenames)
    processed_ims = []
    processed_infos = []
    for i in range(num_images):
        filename = filenames[i]
        assert os.path.exists(filename), '%s does not exist'.format(filename)
        im = cv2.imread(filename)
        im = convert_color_hsv(im, h_offset=random.uniform(-10, 10), s_offset=random.uniform(-30, 30),
                               v_offset=random.uniform(-35, 35))
        im_height, im_width = im.shape[:2]

        if cfg.TRAIN.DATAUG_SQUARE:
            top, bottom, left, right = 0, 0, 0, 0
            if im_width > im_height:
                top = (im_width - im_height) / 2
                bottom = (im_width - im_height - top)
            elif im_width < im_height:
                left = (im_height - im_width) / 2
                right = (im_height - im_width - left)
            im = cv2.copyMakeBorder(im, top, bottom, left, right, 0)
            im = cv2.resize(im, (1000, 1000))

        # data argument
        if cfg.TRAIN.DATAUG:
            if random.randint(0, 1) == 1:
                im = im[:, ::-1, :]
            im_height, im_width = im.shape[:2]
            rot_d = np.random.randint(-180, 180)
            M = cv2.getRotationMatrix2D((im_width / 2, im_height / 2), rot_d, 1)
            im = cv2.warpAffine(im, M, (1000, 1000), None, cv2.INTER_LINEAR, cv2.BORDER_REPLICATE)

        #cv2.imshow("im", im)
        #cv2.waitKey()

        scale_ind = random.randrange(len(cfg.SCALES))
        target_size = cfg.SCALES[scale_ind][0]
        max_size = cfg.SCALES[scale_ind][1]
        im, im_scale = resize(im, target_size, max_size, stride=cfg.IMAGE_STRIDE)
        im_tensor = transform(im, cfg.PIXEL_MEANS)
        processed_ims.append(im_tensor)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        processed_infos.append(im_info)
    return processed_ims, processed_infos

def get_test_image(roidb):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims = []
    processed_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i].copy()
        assert os.path.exists(roi_rec['image']), '%s does not exist'.format(roi_rec['image'])
        im = cv2.imread(roi_rec['image'])
        if roi_rec['flipped']:
            im = im[:, ::-1, :]
        new_rec = roi_rec.copy()
        scale_ind = random.randrange(len(cfg.SCALES))
        target_size = cfg.SCALES[scale_ind][0]
        max_size = cfg.SCALES[scale_ind][1]
        im, im_scale = resize(im, target_size, max_size, stride=cfg.IMAGE_STRIDE)
        im_tensor = transform(im, cfg.PIXEL_MEANS)
        processed_ims.append(im_tensor)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['boxes'] = roi_rec['boxes'].copy() * im_scale
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)
    return processed_ims, processed_roidb

def resize(im, target_size, max_size, stride=0):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :param stride: if given, pad the image to designated stride
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

    if stride == 0:
        return im, im_scale
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel))
        padded_im[:im.shape[0], :im.shape[1], :] = im
        return padded_im, im_scale


def transform(im, pixel_means):
    """
    transform into mxnet tensor,
    subtract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :param pixel_means: [B, G, R pixel means]
    :return: [batch, channel, height, width]
    """
    im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
    for i in range(3):
        im_tensor[0, i, :, :] = im[:, :, 2 - i] - pixel_means[2 - i]
    return im_tensor


def transform_inverse(im_tensor, pixel_means):
    """
    transform from mxnet im_tensor to ordinary RGB image
    im_tensor is limited to one image
    :param im_tensor: [batch, channel, height, width]
    :param pixel_means: [B, G, R pixel means]
    :return: im [height, width, channel(RGB)]
    """
    assert im_tensor.shape[0] == 1
    im_tensor = im_tensor.copy()
    # put channel back
    channel_swap = (0, 2, 3, 1)
    im_tensor = im_tensor.transpose(channel_swap)
    im = im_tensor[0]
    assert im.shape[2] == 3
    im += pixel_means[[2, 1, 0]]
    im = im.astype(np.uint8)
    return im


def tensor_vstack(tensor_list, pad=0):
    """
    vertically stack tensors
    :param tensor_list: list of tensor to be stacked vertically
    :param pad: label to pad with
    :return: tensor with max shape
    """
    ndim = len(tensor_list[0].shape)
    dtype = tensor_list[0].dtype
    islice = tensor_list[0].shape[0]
    dimensions = []
    first_dim = sum([tensor.shape[0] for tensor in tensor_list])
    dimensions.append(first_dim)
    for dim in range(1, ndim):
        dimensions.append(max([tensor.shape[dim] for tensor in tensor_list]))
    if pad == 0:
        all_tensor = np.zeros(tuple(dimensions), dtype=dtype)
    elif pad == 1:
        all_tensor = np.ones(tuple(dimensions), dtype=dtype)
    else:
        all_tensor = np.full(tuple(dimensions), pad, dtype=dtype)
    if ndim == 1:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice] = tensor
    elif ndim == 2:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1]] = tensor
    elif ndim == 3:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1], :tensor.shape[2]] = tensor
    elif ndim == 4:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1], :tensor.shape[2], :tensor.shape[3]] = tensor
    else:
        raise Exception('Sorry, unimplemented.')
    return all_tensor