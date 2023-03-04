from fastapi import APIRouter, UploadFile, File
from models.name import Name
from services import predict
from core.config import settings
from PIL import Image
from io import BytesIO
import numpy as np


router = APIRouter()

@router.get('/', response_model=Name)
def name_response_example():
    return Name()

@router.post('/predict', response_model=Name)
async def name_it(file: UploadFile = File(...)):
    name = Name()
    name.name = predict.predict_name_pet_from_img(await file.read())
    name.message = 'Success'
    return name