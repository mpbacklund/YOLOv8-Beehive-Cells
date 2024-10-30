from ultralytics import YOLO

model = YOLO("yolov8n.pt")
train_results = model.train(data='data.yaml',
    epochs=100,
    imgsz=640,
    device=0)

# evaluate performace on the validation set
metrics = model.val()
# perform obj detection on image
results = model("../data/obj/2013_07_28_CHBRC_010.png")
results[0].show()
# export format
export_model = model.export(format="onnx")