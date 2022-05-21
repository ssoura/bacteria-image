from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from fastai.vision.image import open_image

from django.conf import settings

import os
from fastai import *
from fastai.vision import *
from skimage import measure, filters
import cv2
from scipy import ndimage
import numpy
from fastai.basic_train import load_learner



def index(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']

        learner = load_learner(settings.BASE_DIR+'/bacteriaclassifier/extra/')
        file = request.FILES['myfile']
        
        image_extensions=['ras','xwd', 'bmp', 'jpe', 'jpg', 'jpeg', 'xpm', 'ief', 'pbm', 'tif', 'gif', 'ppm', 'xbm', 'tiff', 'rgb', 'pgm', 'png', 'pnm']    
        
        #if file.split('.')[1] not in image_extensions:
        result = 'Please upload an appropriate image file'
        img = open_image(file)
        pred = learner.predict(img)
        result = str(pred[0])

        return render(request, 'home.html', {
            'result': result
        })
    return render(request, 'home.html')

def result(request):
    return render(request, 'home.html')
