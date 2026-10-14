import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()   # 必须放在第一行
    model = YOLO('yolov13.yaml')
    model.train(
        data="D:\\our dateset\\our dateset\\data.yaml",
        epochs=100,
        batch=12,
        imgsz=640,
        workers=8,
        device=0,
        optimizer='SGD',
        amp=True,
        cache=False
    )