import subprocess
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
import numpy as np
import cv2

import ssl


ssl._create_default_https_context = ssl._create_unverified_context
MODEL_NAME = "movenet_lightning" #@param ["movenet_lightning", "movenet_thunder", "movenet_lightning.tflite", "movenet_thunder.tflite"]


if "tflite" in MODEL_NAME:
  if "movenet_lightning" in MODEL_NAME:
    # !wget -q -O model.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/3?lite-format=tflite
    subprocess.run(["wget", "-O", "model.tflite", "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/3?lite-format=tflite"])
    input_size = 192
  elif "movenet_thunder" in MODEL_NAME:
    # !wget -q -O model.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/3?lite-format=tflite
    subprocess.run(["wget", "-O", "model.tflite", "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/3?lite-format=tflite"])
    input_size = 256
  else:
    raise ValueError("Unsupported model name: %s" % MODEL_NAME)

  # Initialize the TFLite interpreter
  interpreter = tf.lite.Interpreter(model_path="model.tflite")
  interpreter.allocate_tensors()

  def movenet(input_image):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """
    # TF Lite format expects tensor type of float32.
    input_image = tf.cast(input_image, dtype=tf.float32)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    # Invoke inference.
    interpreter.invoke()
    # Get the model prediction.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores

else:
  if "movenet_lightning" in MODEL_NAME:
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/3")
    input_size = 192
  elif "movenet_thunder" in MODEL_NAME:
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/3")
    input_size = 256
  else:
    raise ValueError("Unsupported model name: %s" % MODEL_NAME)

  def movenet(input_image):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """
    model = module.signatures['serving_default']

    # SavedModel format expects tensor type of int32.
    input_image = tf.cast(input_image, dtype=tf.int32)
    # Run model inference.
    outputs = model(input_image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoint_with_scores = outputs['output_0'].numpy()
    return keypoint_with_scores