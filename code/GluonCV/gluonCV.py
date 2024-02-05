import gluoncv
from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt

import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms


from gluoncv.data.transforms.presets.segmentation import test_transform
from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg

def ssd_detection():
    net = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True)

    im_fname = "../static/tmpb1lbm2dd.jpg"
    x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)
    print('Shape of pre-processed image:', x.shape)

    class_IDs, scores, bounding_boxes = net(x)

    ax = utils.viz.plot_bbox(img, bounding_boxes[0], scores[0],
                            class_IDs[0], class_names=net.classes)
    plt.savefig("../processed/gluon_ssd.png")

def yolo3_detection():
    net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)
    im_fname = "../static/tmpb1lbm2dd.jpg"
    x, img = data.transforms.presets.yolo.load_test(im_fname, short=512)
    class_IDs, scores, bounding_boxs = net(x)

    ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
                         class_IDs[0], class_names=net.classes)
    plt.savefig("../processed/gluon_yolo3.png")

def faster_rcnn_detection():
    net = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)
    im_fname = "../static/tmpb1lbm2dd.jpg"
    x, orig_img = data.transforms.presets.rcnn.load_test(im_fname)
    box_ids, scores, bboxes = net(x)
    ax = utils.viz.plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], class_names=net.classes)

    plt.savefig("../processed/gluon_faster_rcnn.png")

def mask_rcnn_segmentation():
    net = model_zoo.get_model('mask_rcnn_resnet50_v1b_coco', pretrained=True)
    im_fname = "../static/tmpb1lbm2dd.jpg"
    x, orig_img = data.transforms.presets.rcnn.load_test(im_fname)
    ids, scores, bboxes, masks = [xx[0].asnumpy() for xx in net(x)]

    # paint segmentation mask on images directly
    width, height = orig_img.shape[1], orig_img.shape[0]
    masks, _ = utils.viz.expand_mask(masks, bboxes, (width, height), scores)
    orig_img = utils.viz.plot_mask(orig_img, masks)

    # identical to Faster RCNN object detection
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax = utils.viz.plot_bbox(orig_img, bboxes, scores, ids,
                            class_names=net.classes, ax=ax)
    plt.savefig("../processed/gluon_segmentation.png")

def FCN_sematic():
    ctx = mx.cpu(0)
    img = image.imread("../static/tmpb1lbm2dd.jpg")
    img = test_transform(img, ctx)
    model = gluoncv.model_zoo.get_model('fcn_resnet101_voc', pretrained=True)
    output = model.predict(img)
    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
    mask = get_color_pallete(predict, 'pascal_voc')
    mask.save("../processed/FCN_sematic.png")


if __name__ == "__main__":
    FCN_sematic()