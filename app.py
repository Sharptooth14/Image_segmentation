import os
import numpy as np
import cv2
import pandas as pd
import torch
import streamlit as st
from torchvision.models.detection import maskrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

#Image segmentation
def segment_image(image):
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    
    image_tensor = F.to_tensor(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)

    masks = output[0]['masks']
    segmented_images = []
    for i in range(masks.shape[0]):
        mask = masks[i, 0].mul(255).byte().cpu().numpy()
        segmented_images.append(mask)

    return segmented_images

def extract_and_save_objects(segmented_images, original_image, master_id):
    if not os.path.exists('extracted_objects'):
        os.makedirs('extracted_objects')
    
    object_images = []
    for i, mask in enumerate(segmented_images):
        object_image = cv2.bitwise_and(original_image, original_image, mask=mask)
        object_id = f'obj_{i}_master_{master_id}.png'
        cv2.imwrite(os.path.join('extracted_objects', object_id), object_image)
        object_images.append(os.path.join('extracted_objects', object_id))

    return object_images
#Image detection
def identify_objects(object_images):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    descriptions = []
    for img_path in object_images:
        image = cv2.imread(img_path)
        image_tensor = F.to_tensor(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image_tensor)

        labels = output[0]['labels'].cpu().numpy()
        descriptions.append(labels)  # Replace with label names 

    return descriptions
#Object analysis
def summarize_attributes(identified_objects):
    
    COCO_LABELS = {1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
                   6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light", 
                   11: "fire hydrant", 12: "stop sign", 13: "parking meter", 14: "bench", 
                   15: "bird", 16: "cat", 17: "dog", 18: "horse", 19: "sheep", 
                   20: "cow", 21: "elephant", 22: "bear", 23: "zebra", 24: "giraffe"}

    summaries = []
    for obj in identified_objects:
        description = [COCO_LABELS.get(label, "unknown") for label in obj]  # Map labels to names
        summaries.append({
            'object_id': obj,
            'description': description
        })
    return summaries


def main():
    st.title("Image Segmentation and Object Analysis")

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png"])

    if uploaded_file is not None:

        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption='Uploaded Image', channels='BGR')

        
        segmented_images = segment_image(image)

        
        object_images = extract_and_save_objects(segmented_images, image, master_id=1)

        
        identified_objects = identify_objects(object_images)

        
        summarized_attributes = summarize_attributes(identified_objects)


        st.subheader("Segmented Images and Their Descriptions")
        for obj_id, (seg_img, desc) in enumerate(zip(object_images, summarized_attributes)):
            st.image(seg_img, caption=f"Object ID: {desc['object_id']} - Description: {', '.join(desc['description'])}")

        #Display results
        st.subheader("Object Summaries")
        df = pd.DataFrame(summarized_attributes)
        st.dataframe(df)

if __name__ == "__main__":
    main()
