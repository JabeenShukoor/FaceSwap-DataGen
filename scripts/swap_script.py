#importing packages 
import cv2
import insightface
import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import warnings
from .detection.detection_script import detect_face_buffalo, detect_face_mtcnn

'''
It was noted that for some images, swapping did not occur.
So this is to ensure that only swapped faces are generated.
'''

# function to compute mean squared error between two images
def mse(imageA, imageB):
    imageA_resized = cv2.resize(imageA, (1024, 1024))
    imageB_resized = cv2.resize(imageB, (1024, 1024))
    err = np.sum((imageA_resized.astype("float") - imageB_resized.astype("float")) ** 2)
    err /= float(imageA_resized.shape[0] * imageA_resized.shape[1])
    return err

'''
This function detects the face uisng buffalo_l model, if it does not
detect then mtcnn model is used. Features needed for inswapper model 
are extracted from the detected face and then swapped.
'''

#function to swap the face
def swap_face(img1, img2, swapper_model):
    img1_face = detect_face_buffalo(img1)
    img2_face = detect_face_buffalo(img2)

    if img1_face is None:
        img1_face = detect_face_mtcnn(img1)
    if img2_face is None:
        img2_face = detect_face_mtcnn(img2)

    if img1_face is None or img2_face is None:
        print("Face Detection Failed!")
        return None

    img1_resized = cv2.resize(img1, (1024, 1024))
    img2_resized = cv2.resize(img2, (1024, 1024))

    try:
        img_swap = swapper_model.get(img1_resized.copy(), img1_face, img2_face, paste_back=True)
    except Exception as e:
        print(f"Error during swapping: {e}")
        return None

    if img_swap is None:
        print("Swapper model returned None for the swapped image.")
        return None

    return img_swap

# This function checks if two faces are a match to be swapped. 
def check_swap(face1, face2):
    age_diff = abs(face1['age'] - face2['age']) <= 5
    same_gender = face1['gender'] == face2['gender']
    same_race =  face1['race'] == face2['race']
    same_emotion = face1['emotion'] == face2['emotion']
    return age_diff and same_gender and same_race and same_emotion

def main():
    # Setup

    #To suppress the warning 
    tf.get_logger().setLevel('ERROR') 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['ORT_LOGGING_LEVEL'] = 'ERROR'
    warnings.filterwarnings("ignore")

    #Loading the inswapper model
    swapper_model = insightface.model_zoo.get_model('models/inswapper_128.onnx', download=False, download_zip=False)

    #Load the sampled metadata from phase 2 
    sampled_metadata = pd.read_csv("sampled_metadata.csv")

    #Output directory
    output_directory = "swapped_images/"
    os.makedirs(output_directory, exist_ok=True)

    #number swap faces to be created 
    number_of_swaps = 100

    #to store tuple (source_image_path, target_image_path, swapped_image_path)
    swap_log = []  

    # Main loop
    max_attempts = 1000 
    swaps = 0
    #To track used source images
    used_sources = set() 

    while swaps < number_of_swaps and max_attempts > 0:
        max_attempts -= 1 

        try:
            #Selecting a random source image
            source_row = sampled_metadata.sample().iloc[0]
            img1_path = source_row['image_path']

            #Skip if this source image has been used before
            if img1_path in used_sources:
                continue

            img1 = cv2.imread(img1_path)

            if img1 is None:
                print(f"Error reading source image: {img1_path} not found.")
                continue

            # Find a matching target image that follows the condition
            possible_matches = sampled_metadata[~sampled_metadata['image_path'].isin([img1_path])]
            matched_rows = [match_row for _, match_row in possible_matches.iterrows() if check_swap(source_row, match_row)]

            if not matched_rows:
                print(f"No matching face found for source {img1_path}... Skipping")
                continue 

            # Randomly select a target image from the matched rows of data
            target_row = random.choice(matched_rows)
            img2_path = target_row['image_path']
            img2 = cv2.imread(img2_path)

            if img2 is None:
                print(f"Error reading target image: {img2_path} not found.")
                continue

            # swapping the faces
            swapped_image = swap_face(img1, img2, swapper_model)

            if swapped_image is not None:
                # To Check if the swap result is identical to the source image
                if mse(img1, swapped_image) > 100: 
                    swapped_image_filename = f"swapped_face_{swaps}.jpg"
                    swapped_image_path = os.path.join(output_directory, swapped_image_filename)

                    # Saving
                    saved = cv2.imwrite(swapped_image_path, swapped_image)
                    if saved:
                        print(f"Image successfully saved: {swapped_image_path}")
                        #logging the details
                        swap_log.append((img1_path, img2_path, swapped_image_path))
                        swaps += 1 
                        used_sources.add(img1_path) 
                    else:
                        print(f"Failed to save image: {swapped_image_path}")
                        continue

                else:
                    print(f"Swapped image is too similar to the source. Skipping swap.")

            else:
                print("Face swap returned None. Skipping swap.")

        except Exception as e:
            print(f"An error occurred during face swapping: {e}")

    # Save the swap log to a CSV file
    swap_log_df = pd.DataFrame(swap_log, columns=['source_image_path', 'target_image_path', 'swapped_image_path'])
    swap_log_df.to_csv('face_swap_log.csv', index=False)

if __name__ == "__main__":
    main()
