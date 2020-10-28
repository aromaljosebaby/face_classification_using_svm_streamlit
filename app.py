import streamlit as st
import pickle
import numpy as np
import json
import joblib
from PIL import Image
import pywt
import cv2


#model=pickle.load(open('model.pickle','rb'))

model = joblib.load( 'saved_model.pkl')



import time








def predicting_images_functions(img_path):






    lol = []
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(
        'haarcascade_eye.xml')

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    x, y, w, h = faces[0]

    for (x, y, w, h) in faces:
        face_img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = face_img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cropped_img = np.array(roi_color)

    def w2d(img, mode='haar', level=1):
        imArray = img
        # Datatype conversions
        # convert to grayscale
        imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
        # convert to float
        imArray = np.float32(imArray)
        imArray /= 255;
        # compute coefficients
        coeffs = pywt.wavedec2(imArray, mode, level=level)

        # Process Coefficients
        coeffs_H = list(coeffs)
        coeffs_H[0] *= 0;

        # reconstruction
        imArray_H = pywt.waverec2(coeffs_H, mode);
        imArray_H *= 255;
        imArray_H = np.uint8(imArray_H)

        return imArray_H

    scalled_raw_img = cv2.resize(cropped_img, (32, 32))
    img_har = w2d(cropped_img, 'db1', 5)

    scalled_img_har = cv2.resize(img_har, (32, 32))
    combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32,
                                                                                               1)))  # stacking the vectorizede img and raw img one over the other that is the x
    lol.append(combined_img)
    lol = np.array(lol).reshape(len(lol), 4096).astype(float)  # so it will be one row and 4097 columns




    return lol

















def main():
    st.title("Image Classification")

    html_temp = """
    <div style="background-color:#e63946 ;padding:10px">
    <h2 style="color:white;text-align:center;">Image Classification Prediction ML App </h2>
    </div><br><br>
    """




    st.markdown(html_temp, unsafe_allow_html=True)




    st.image(['maria.jpg','roger.jpg','virat.jpg','serena.jpg','messi.jpg'],width=100,height=50)




    safe_html="""  
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:white;text-align:center;"> Your forest is safe</h2>
       </div>
    """
    danger_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> Your forest is in danger</h2>
       </div>
    """

    st.subheader('Upload images of any of the above 5 people here to begin classification ')
    img_file=st.file_uploader('.')





    if st.button("Predict"):

        if img_file==None:
            st.error('Please upload an Image')

            st.image(['man.png'], width=150, height=50)



        else:

            image=Image.open(img_file)



            img_array=np.array(image)

            cv2.imwrite('out.jpg',cv2.cvtColor(img_array,cv2.COLOR_RGB2BGR))

            #cv_img = np.asarray(image)

            st.image(image,use_column_width=True)

            bar=st.progress(0)
            for i in range(100):
                bar.progress(i+1)
                time.sleep(0.01)

            #output=predicting_using_model(total_sqft, no_of_bathroom, no_of_balcony, area_type, bhk, location)
            #st.success(f'The predicted house price is {output} Indian rupees')






            lol = predicting_images_functions('out.jpg')

            print(model.predict(lol))

            prediction = model.predict(lol)[0]
            # print(prediction,'*****************')

            f = open('class_dictionary.json')

            class_dict = json.load(f)



            prdection_name = class_dict[str(prediction)]

            st.success(f'The person in the picture is  {prdection_name} ')


if __name__=='__main__':
    main()