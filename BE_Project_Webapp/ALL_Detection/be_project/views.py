from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from segmentation import Segmentation
from .forms import UploadImageForm
import numpy as np
import cv2
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import save_img


def preprocess_image(uploaded_file):
    # Read the image from InMemoryUploadedFile
    image_bytes = uploaded_file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert the image to grayscale and resize if needed
    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_image = cv2.resize(processed_image, (256, 256))  # Resize as required by your model

    return processed_image

@api_view(['GET'])
def main_page(request):
    return render(request,'main.html')


@api_view(['POST'])
def process_data(request):
    form = UploadImageForm(request.POST, request.FILES)
    uploaded_image = request.FILES.get('image')

    processed_image = preprocess_image(uploaded_image)

    cv2.imwrite("C:/Users/hp/Desktop/Leukemia_detection_BE_Project_Code/BE_Project_Webapp/ALL_Detection/be_project/static/original_image.jpg",processed_image)

    segmentation = Segmentation(processed_image)

    best_image_segment = segmentation.get_image_segment()

    model = load_model("C:/Users/hp/Desktop/Leukemia_detection_BE_Project_Code/cnn_model.h5")

    path = "C:/Users/hp/Desktop/Leukemia_detection_BE_Project_Code/BE_Project_Webapp/ALL_Detection/be_project/static/temp_image.jpg"

    cv2.imwrite(path,best_image_segment)


    original_image = load_img(path)
    original_image_array = img_to_array(original_image)

    rotated_image_array = np.rot90(original_image_array, k=0)
        
    rotated_image = array_to_img(rotated_image_array)

    cv2.imwrite(path,img_to_array(rotated_image))

    image = load_img(path)
    image_array = img_to_array(image)
    image_array /= 255.0
    img_input = np.expand_dims(image_array, axis=0)

    pred = model.predict(img_input)

    pred = list(pred[0])

    stage = pred.index(max(pred))+1

    message = ""

    if stage==1:
        message = """FAB L1 (children): \n
Chemotherapy is the mainstay of treatment for children with ALL. It usually involves a combination of drugs given in different phases, including induction, consolidation, maintenance, and sometimes central nervous system prophylaxis.
Depending on the risk stratification, patients may receive additional treatments such as corticosteroids, intrathecal chemotherapy (chemotherapy injected into the spinal fluid), and targeted therapies like monoclonal antibodies.
Stem cell transplantation may be considered for patients with high-risk disease or those who experience relapse."""

    elif stage==2:
        message = """FAB L2 (older children and adults):\n
Similar to L1, chemotherapy remains the primary treatment approach for L2 ALL. The regimen may be adjusted based on the patient's age, overall health, and risk factors.
In some cases, targeted therapies like monoclonal antibodies (e.g., rituximab) may be used in combination with chemotherapy.
Stem cell transplantation may be considered for eligible patients, particularly those with high-risk features or those who relapse."""

    else:
        message = """FAB L3 (patients with leukemia):\n
FAB L3, also known as Burkitt's leukemia, is a subtype of ALL characterized by a specific genetic abnormality called the Philadelphia chromosome.
Treatment typically involves aggressive chemotherapy regimens, often including high-dose methotrexate and cytarabine, as well as targeted therapies such as tyrosine kinase inhibitors (e.g., imatinib, dasatinib) to target the Philadelphia chromosome.
Intensive supportive care is crucial due to the high risk of tumor lysis syndrome, a condition caused by the rapid breakdown of cancer cells.
Stem cell transplantation may be considered for eligible patients, particularly those with high-risk features or those who relapse.
It's important to note that treatment plans are highly individualized, and patients should discuss their specific situation and treatment options with their healthcare team. Additionally, clinical trials may offer access to novel therapies and treatment approaches for ALL."""


    label = "Stage of ALL: L" + str(stage)

    context = {'label': label,'message': message}

    return render(request, 'stage.html', context)

