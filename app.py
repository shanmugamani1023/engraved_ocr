import sys
import pickle
sys.path.append('/app/yolov5/')
sys.path.append('/app/parseq/') # Adjust here
import streamlit as st
import torch
from models.common import DetectMultiBackend
import cv2
import numpy as np
from custom_utils import get_bboxes,get_yolo_results,draw_detections_with_text,remove_lowest_area_bbox,process_detections
device = torch.device('cpu')
weights = '/app/best.pt' # Adjust here
dnn= False
fp16=False 
data = '/app/data.yaml' # Adjust here
model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=fp16)
imgsz=(640, 640)
pt = model.pt
model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))

# Define the path to the checkpoint
checkpoint_path = '/app/epoch=16-step=2086-val_accuracy=100.0000-val_NED=100.0000.ckpt'

# Define the model parameters
model_args = {
    'data_root': 'data',
    'batch_size': 1,  # Set batch size to 1 for inference on singular images
    'num_workers': 4,
    'cased': False,
    'punctuation': False,
    'new': False,  # Set to True if you want to evaluate on new benchmark datasets
    'rotation': 0,
    'device': 'cpu'  # Use 'cuda' or 'cpu' depending on your environment
}

# Load the model checkpoint
#model_ocr = torch.hub.load('baudm/parseq', 'parseq', pretrained=True,map_location=torch.device('cpu'))  # Example: Replace with your model loading code
with open('/app/tokenizer.pkl', 'rb') as tokenizer_file:
    loaded_tokenizer = pickle.load(tokenizer_file)
model_ocr = torch.jit.load('/app/ParseqPretrained.pth').eval().to('cpu')
model_ocr.load_state_dict(torch.load(checkpoint_path,map_location=torch.device('cpu'))['state_dict'])
model_ocr.eval()
model_ocr.to(model_args['device'])



def process_images(image_path_1, image_path_2,model_ocr,loaded_tokenizer):
    detections_left = get_bboxes(image_path_1, get_yolo_results(image_path_1,model))
    detections_right = get_bboxes(image_path_2, get_yolo_results(image_path_2,model))
    detections_left = remove_lowest_area_bbox(detections_left)
    detections_right = remove_lowest_area_bbox(detections_right)
    ocr_results_left = process_detections(detections_left,image_path_1,model_ocr, model_args,loaded_tokenizer)
    ocr_results_right = process_detections(detections_right,image_path_2,model_ocr, model_args,loaded_tokenizer)
    marked_left_image = draw_detections_with_text(cv2.imread(image_path_1),ocr_results_left)
    marked_right_image = draw_detections_with_text(cv2.imread(image_path_2),ocr_results_right)
    return ocr_results_left,marked_left_image,ocr_results_right,marked_right_image

def main():
    st.title("OCR")
    
    # Image upload buttons
    uploaded_image1 = st.file_uploader("Upload Image 1", type=["jpg", "png", "jpeg"])
    uploaded_image2 = st.file_uploader("Upload Image 2", type=["jpg", "png", "jpeg"])
                                                         
    
    # Submit button
    if st.button("Submit"):
        if uploaded_image1 and uploaded_image2:
            left_image = cv2.imdecode(np.asarray(bytearray(uploaded_image1.read()), dtype=np.uint8), 1)
            cv2.imwrite('/app/left-image.jpg',left_image)     # Adjust here
            right_image = cv2.imdecode(np.asarray(bytearray(uploaded_image2.read()), dtype=np.uint8), 1)
            cv2.imwrite('/app/right-image.jpg',right_image)        # Adjust here
            # Call the function and get data
            ocr_results_left,marked_left_image,ocr_results_right,marked_right_image = process_images('/app/left-image.jpg', '/app/right-image.jpg',model_ocr,loaded_tokenizer)
            
            col1, col2 = st.columns(2)
            
            # Column 1
            with col1:
                st.image(marked_left_image,  caption="Left Workstation Processed Image",use_column_width=True)
                st.markdown(f"**MachinePeenText:** **{ ocr_results_left['MachinePeenText'][0]}**")
                st.markdown(f"**DotPeenText:** **{ ocr_results_left['DotPeenText'][0]}**")
            
            # Column 2
            with col2:
                st.image(marked_right_image, caption="Right Workstation Processed Image", use_column_width=True)
                st.markdown(f"**MachinePeenText:** **{ ocr_results_right['MachinePeenText'][0]}**")
                st.markdown(f"**DotPeenText:** **{ ocr_results_right['DotPeenText'][0]}**")
        else:
            st.warning("Please upload both images.")
        if (ocr_results_right['MachinePeenText'][0][0] == ocr_results_left['MachinePeenText'][0][0]) and (ocr_results_right['DotPeenText'][0][0] == ocr_results_left['DotPeenText'][0][0]) :
            st.markdown("**Match Status : OK**")
        else:
            st.markdown("**Match Status : NOT OK**")
        # Refresh button
        if st.button("Refresh"):
            st.caching.clear_cache()


if __name__ == "__main__":
    main()
