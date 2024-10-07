from PIL import Image, ImageStat
from deepface import DeepFace
import os
import glob
import pandas as pd
import numpy as np

#funtction to get the brightness of the image
def get_brightness(img):
    grayscale = img.convert('L')  
    stat = ImageStat.Stat(grayscale)
    return stat.mean[0] 

#function to get metadata 
def get_metadata_info(image_paths, max_bright = 180, min_bright = 40):
    metadata_list = []
    for path in image_paths:
        with Image.open(path) as img:
            brightness = get_brightness(img)
            #check if brightness meets the threshold 
            if brightness > max_bright or brightness < min_bright:
                continue

            try:
                metadata = DeepFace.analyze(
                    np.array(img),
                    actions=['age', 'gender', 'race', 'emotion'],
                    silent=True,
                    enforce_detection=True
                )
                metadata_list.append(
                    {
                    'image_path': path,
                    'age': metadata[0]['age'],
                    'race': metadata[0]['dominant_race'],
                    'gender': metadata[0]['dominant_gender'],
                    'emotion': metadata[0]['dominant_emotion'],
                    'resolution': img.size,               
                    'brightness': brightness
                    }
                )
            except Exception as e:
                print(f"Skipping {path}: no face found")
    return metadata_list

def run(dir_path, output_path):
    image_paths  = []
    image_paths += glob.glob(os.path.join(dir_path, "*.jpg"))

    metadata = get_metadata_info(image_paths)
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(output_path, index=False)

def main():
    dir_path = "images"
    output_path = "image_metadata.csv"
    run(dir_path, output_path)

if __name__ == "__main__":
    main()
