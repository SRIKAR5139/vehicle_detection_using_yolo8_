from ultralytics import YOLO

model = YOLO("runs/detect/train7/weights/best.pt")

results = model("images/img_11.jpg", conf=0.25, save=True, show=True)

print("Detection completed!")