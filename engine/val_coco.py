import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_json', type=str, default='/home/cvmdsp3/opt/ship_datasets/HRSID/annotations/test.json', help='training model path')
    parser.add_argument('--pred_json', type=str, default='/home/cvmdsp3/opt/ship_yolo/ultralytics/runs/detect/train67/predictions.json', help='data yaml path')
    return parser.parse_known_args()[0]

def parse_opt1():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_json', type=str, default='/home/cvmdsp3/opt/ship_datasets/coco_style/annotations/test.json', help='training model path')
    parser.add_argument('--pred_json', type=str, default='/home/cvmdsp3/opt/ship_yolo/ultralytics/runs/detect/val89/upredictions.json', help='data yaml path')
    return parser.parse_known_args()[0]


def mycocoeval(mypred_json):
    opt = parse_opt()
    anno_json = opt.anno_json
    pred_json = mypred_json
    anno = COCO(anno_json)  # 初始化标注的json文件
    pred = anno.loadRes(pred_json)  # 初始化预测的json文件
    eval = COCOeval(anno, pred, 'bbox')
    eval.evaluate()
    eval.accumulate()
    eval.summarize()


# if __name__ == '__main__':
#     opt = parse_opt()
#     anno_json = opt.anno_json
#     pred_json = opt.pred_json
#     anno = COCO(anno_json)  # 初始化标注的json文件
#     pred = anno.loadRes(pred_json)  # 初始化预测的json文件
#     eval = COCOeval(anno, pred, 'bbox')
#     eval.evaluate()
#     eval.accumulate()
#     eval.summarize()
    #
    # opt = parse_opt1()
    # anno_json = opt.anno_json
    # pred_json = opt.pred_json
    # anno = COCO(anno_json)  # 初始化标注的json文件
    # pred = anno.loadRes(pred_json)  # 初始化预测的json文件
    # eval = COCOeval(anno, pred, 'bbox')
    # eval.evaluate()
    # eval.accumulate()
    # eval.summarize()
