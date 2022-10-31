from io import BytesIO
from PIL import Image
import numpy as np
from constants import PRED_PATH, DETECTION_THRESHOLD, TFLITE_MODEL_PATH, COST_DAMAGE_REPAIR
from services.custom_object_detection import ObjectDetector, ObjectDetectorOptions, visualize

detector = None

def load_model():
    # # Load the TFLite model
    options = ObjectDetectorOptions(
        num_threads=4,
        score_threshold=DETECTION_THRESHOLD,
    )
    detector = ObjectDetector(model_path=TFLITE_MODEL_PATH, options=options)
    return detector

def check_file(file):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG")
    if not extension:
        return "Image must be jpg or png format!"
    else:
        return "ok"

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

def predict(image: Image.Image, filename):
    global detector
    if detector is None:
        detector = load_model()

    image = image.convert('RGB')
    image.thumbnail((512, 512), Image.ANTIALIAS)
    image_np = np.asarray(image)

    # Run object detection estimation using the model.
    detections = detector.detect(image_np)

    # Draw keypoints and edges on input image
    image_np, damaged_measure = visualize(image_np, detections)
    # Show the detection result
    # Image.fromarray(image_np)
    data = Image.fromarray(image_np)
    data.save(PRED_PATH+filename)
    return damaged_measure

def generate_repair_report(image: Image.Image, filename):
    measures = predict(image, filename)
    damages = []
    for i in measures:
        damages.append(i*100*COST_DAMAGE_REPAIR)
    return damages