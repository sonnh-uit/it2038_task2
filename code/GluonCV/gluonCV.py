import gluoncv
from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt


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
    
if __name__ == "__main__":
    faster_rcnn_detection()