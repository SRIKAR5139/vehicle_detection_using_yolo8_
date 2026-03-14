from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model(
    "images/img_11.jpg",
    conf=0.25,
    save=True,
    show=True,
    classes=[2,5,7]   # car, bus, truck in COCO
)

print("Detection completed!")