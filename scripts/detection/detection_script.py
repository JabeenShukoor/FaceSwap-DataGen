import cv2
from insightface.app import FaceAnalysis
import numpy as np
from mtcnn import MTCNN
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense 
from tensorflow.keras.models import Sequential

#creating a class for structuring the face data 
class FaceData:
    def __init__(self, box, confidence, kps, normed_embedding):
        self.box = np.array(box, dtype=np.float32)
        self.confidence = confidence
        self.kps = np.array(kps, dtype=np.float32)
        self.normed_embedding = np.array(normed_embedding, dtype=np.float32)

    def __str__(self):
        return (f"FaceData(\n"
                f"  Box: {self.box.tolist()},\n"
                f"  Confidence: {self.confidence},\n"
                f"  Keypoints: {self.kps.tolist()},\n"
                f"  Normed Embedding: {self.normed_embedding.tolist()}\n"
                f")")
    
    def __repr__(self):
        return self.__str__()
    
#function to preprocess the images
def image_preprocess(img):
    img_resize = cv2.resize(img, (1024,1024))
    return img_resize

#extract features - if face was detected using FaceAnalysis from insightface 
def extract_features_buffalo(data):
    return FaceData(
        box = data['bbox'].tolist(),
        confidence = data['det_score'],
        kps = data['kps'].tolist(),
        normed_embedding=data['embedding'].tolist()
    )

#extract features - if face was detected using MTCNN
def extract_features_mtcnn(data):
    transformed_data_list = []
    for d in data:
        transformed_data = {
        'box': d['box'],  
        'confidence': d['confidence'],  
        'kps': [  
            d['keypoints']['left_eye'], 
            d['keypoints']['right_eye'], 
            d['keypoints']['nose'], 
            d['keypoints']['mouth_left'],
            d['keypoints']['mouth_right']
        ],
        'normed_embedding': d['normed_embedding']
        }
        transformed_data['kps'] = [list(point) for point in transformed_data['kps']]
        transformed_data_list.append(FaceData(transformed_data['box'], transformed_data['confidence'], transformed_data['kps'], transformed_data['normed_embedding']))
    return transformed_data_list

def reduce_embedding_dimensionality(embeddings, target_dim=512):
    #Apply PCA to reduce dimensionality from 1280 to 512
    model = Sequential()
    model.add(Dense(target_dim, input_shape=(embeddings.shape[-1],), activation=None))
    #Perform dimensionality reduction
    reduced_embeddings = model.predict(embeddings)
    return reduced_embeddings

def detect_face_buffalo(img):
    app = FaceAnalysis(name = 'buffalo_l')
    app.prepare(ctx_id=0, det_size=(1024,1024))

    image_preprocessed = image_preprocess(img)
    faces = app.get(image_preprocessed)

    if len(faces) > 0:
        face_features = extract_features_buffalo(faces[0])
        return face_features
    else:
        return None
    
def detect_face_mtcnn(img):
    detect_model = MTCNN()
    image_preprocessed = image_preprocess(img)
    faces = detect_model.detect_faces(cv2.cvtColor(image_preprocessed, cv2.COLOR_BGR2RGB))

    if len(faces)>0:

        #to get the embeddings
        base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
        img_resized = cv2.resize(img, (512, 512))
        img_preprocessed = preprocess_input(img_resized.astype(np.float32))
        img_expanded = np.expand_dims(img_preprocessed, axis=0)
        embeddings = base_model.predict(img_expanded)
        embeddings_512 = reduce_embedding_dimensionality(embeddings, target_dim=512)

        for im in faces:
            im['normed_embedding'] = embeddings_512

        #extract features
        face_features = extract_features_mtcnn(faces)[0]
        return face_features 
    else:
        return None
