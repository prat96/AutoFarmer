from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("yolo11x-seg.pt")

# Perform object detection on an image
results = model(source="https://www.youtube.com/watch?v=1FfoZpV2VPY", stream=True, show=True)

 #Visualize the results
for result in results:
   result.show()