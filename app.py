
import numpy as np
import streamlit as st
import cv2
from google.colab.patches import cv2_imshow
import pytesseract
from PIL import Image
st.title('Optical Character Recognition of Car Number plate')
st.write('[This WebApp detects Car number plate and Converts it into Text]')
up_img = st.sidebar.file_uploader('Upload a Image Consisting of Number Plate')
if up_img is not None:
  read_img = Image.open(up_img)
  img_np = np.array(read_img)
  st.image(img_np, caption = 'Your Uploaded Image')
 
  if st.button('Detect Number Plate & Extract Text from Number Plate'):
    num_p = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
    d_nump = num_p.detectMultiScale(img_np, 1.04, 25)
    for (x,y,w,h) in d_nump:
      cv2.rectangle(img_np , (x,y), (x+w,y+h), (68,219, 121), 3)
      nump_det = img_np[y:y+h, x:x+w]
      st.image(nump_det, caption = 'Detected Number Plate')
      buton = pytesseract.image_to_string(nump_det, lang = 'eng')
      
      det_box = pytesseract.image_to_boxes(nump_det)
 
      hImg,wImng,d = nump_det.shape 
      for b in det_box.splitlines():
        b = b.split()
        x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
        cv2.rectangle(nump_det,(x,hImg-y),(w,hImg-h),(255,15,15),1)
  
      st.image(nump_det , 'Bound Boxes of Recognized Image')
      st.write('Text extracted from Number Plate  ~-',buton)
