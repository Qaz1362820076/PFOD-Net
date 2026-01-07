from ultralytics import YOLO


 model = YOLO("PFOD-Net.pt")

 model.train(
    data="PolarLITIS.yaml",  # 数据集配置文件
     epochs=100,  # 训练轮数
     imgsz=640,  # 输入图像大小
     batch=32,  # 批量大小
     device="0",  # 使用 GPU（如果是 CPU，设置为 "cpu"）
     workers=0,  # 数据加载的线程数
     patience=100,

 )
