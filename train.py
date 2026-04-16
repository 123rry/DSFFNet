import os
os.environ["WANDB_MODE"] = "offline"
from ultralytics import YOLO



def train_model():

    model = YOLO("/root/autodl-fs/yolov12-main/ultralytics/cfg/models/11/yolo11s-dctconv-transform.yaml")
    results = model.train(
        data="/root/autodl-fs/yolov12-main/ultralytics/models/yolo/segment/data.yaml",
        epochs=200,               # 训练轮数
        batch=10,                # 批次大小
        imgsz=640,                # 输入图像尺寸
        device="0",               # 使用的 GPU（可设为 "0,1" 多卡训练）
        workers=8,                # Dataloader 线程数
        optimizer="AdamW",         # 自动选择优化器
        lr0=0.01,                 # 初始学习率
        lrf=0.01,                 # 最终学习率
        weight_decay=0.0005,      # 权重衰减
        dropout=0.0,              # Dropout（YOLOv8 默认不会用）
        augment=True,             # 启用数据增强
        rect=False,               # 不使用矩形训练
        cos_lr=True,              # 使用余弦学习率调度
        patience=50,              # 提前停止条件
        seed=1,                   # 随机种子
        deterministic=True,       # 确保训练可复现
        task="segment",           # 明确指定是分割任务
        pretrained=False,           # 如果你想用 COCO 预训练权重初始化，可以设为 True
        amp=False,
        save_period=30
        
           )
