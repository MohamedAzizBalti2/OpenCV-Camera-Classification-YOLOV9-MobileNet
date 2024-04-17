import cv2
import tensorflow as tf
import pathlib
from ultralytics import YOLO
# List of class names for COCO dataset
COCO_NAMES = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
              'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
              'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
              'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
              'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
              'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
              'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
              'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote',
              'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
              'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Load YOLO model
model = YOLO("yolov9c.pt")


# Function to perform object detection using YOLO model
def predict(chosen_model, img, classes=[], conf=0.5):
    # Predict objects in the image
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)
    return results


# Function to predict and draw bounding boxes around objects
def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf=conf)
    # Iterate over each detected object and draw bounding box
    for result in results:
        for box in result.boxes:
            # Draw bounding box rectangle
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            # Add text label to the bounding box
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    return img, results


# Function to load the detection model
def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name,
        origin=base_url + model_file,
        untar=True)
    model_dir = pathlib.Path(model_dir) / "saved_model"
    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']
    return model


# Model name for object detection
model_name = "ssd_mobilenet_v1_coco_2017_11_17"
# Load the object detection model
detection_model = load_model(model_name)

# Video capture from webcam
vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Main loop to read frames from the webcam
while True:
    # Capture the video frame
    ret, frame = vid.read()

    # Uncomment below lines if using object detection model directly
    """
    # Perform object detection
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = tf.expand_dims(input_tensor, axis=0)
    detections = detection_model(input_tensor)
    # Get bounding box coordinates, classes, and scores
    num_detections = int(detections['num_detections'][0])
    detection_boxes = detections['detection_boxes'][0][:num_detections]
    detection_classes = detections['detection_classes'][0].numpy().astype(int)[:num_detections]
    detection_scores = detections['detection_scores'][0].numpy()[:num_detections]

    # Draw rectangles around detected objects
    for i in range(num_detections):
        score = detection_scores[i]
        if score > 0.5 and detection_classes[i] != 1:
            ymin, xmin, ymax, xmax = detection_boxes[i]
            xmin = int(xmin * frame.shape[1])
            xmax = int(xmax * frame.shape[1])
            ymin = int(ymin * frame.shape[0])
            ymax = int(ymax * frame.shape[0])
            class_id = detection_classes[i]
            label = f"{COCO_NAMES[class_id]}: {score:.2f}"
            color = (0, 255, 0)  # Green color
            thickness = 1
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness)
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    """

    # Call predict_and_detect function to use YOLO model for object detection
    predict_and_detect(model, frame)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
vid.release()
cv2.destroyAllWindows()
