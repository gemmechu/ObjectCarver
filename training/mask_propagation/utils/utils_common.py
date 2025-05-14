import json
import numpy as np
import cv2

def save_json(json_dict, outpath):
    with open(outpath, 'w') as fp:
        json.dump(json_dict, fp, indent=2, sort_keys=True)

def convert_image_to_binary(image, target_value):

    target_value = np.array([target_value[-1], target_value[1],target_value[0]])  #convert RGB to BGR
    # print("after:", target_value)
    target_value_broadcasted = target_value.reshape(1, 1, 3)
    
    binary_image = np.where(np.all(image == target_value_broadcasted, axis=2, keepdims=True), 255, 0).astype(np.uint8)
    
    return binary_image

def overlay_images(image1, image2, alpha=0.5):
   

    # Resize image2 to match the dimensions of image1
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    # Create the overlay image
    overlay = cv2.addWeighted(image1, 1 - alpha, image2, alpha, 0)

    return overlay