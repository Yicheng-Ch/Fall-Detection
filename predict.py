import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from collections import Counter

if __name__ == '__main__':
    model = YOLO("D:\yolov13-main (2)\yolov13-main\\runs\detect\\train63\weights\\best.pt")
    source = r"D:\our dateset\our dateset\images\test"   # 使用原始字符串避免转义

    results = model.predict(source, save=True, verbose=False)

    # 汇总统计
    total_imgs = len(results)
    total_boxes = 0
    class_counter = Counter()
    confidences = []

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        total_boxes += len(boxes)
        for cls_id, conf in zip(boxes.cls, boxes.conf):
            class_name = model.names[int(cls_id)]
            class_counter[class_name] += 1
            confidences.append(conf.item())

    avg_conf = sum(confidences) / len(confidences) if confidences else 0

    print("\n========== 总体检测结果 ==========")
    print(f"总图片数        : {total_imgs}")
    print(f"总检测目标数    : {total_boxes}")
    print("各类别数量      :")
    for name, cnt in class_counter.items():
        print(f"  - {name}: {cnt}")
    print(f"平均置信度      : {avg_conf:.3f}")
    print(f"结果保存目录    : runs\\detect\\predict")