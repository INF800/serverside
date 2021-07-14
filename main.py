from fastapi import FastAPI, File ,UploadFile
from fastapi import Request # for get
from pydantic import BaseModel # for post

import os, time, cv2, shutil, base64
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf

from movenet import movenet


app = FastAPI()


class FrameRequest(BaseModel):
    data_b64: str
    conn_id: int
    frame_id: int


def post_proc(kps):
    """ kps is output of `movenet`. plug-in business logic here."""
    return kps.tolist()


@app.post("/uploadfile/")
def create_upload_file(req: FrameRequest):
    """
    to test the endpoint manually, goto https://elmah.io/tools/base64-image-encoder/ 
    and create base64 string and use it in swagger ui 
    """
    try:

        # preprocessing
        im_bytes = base64.b64decode(req.data_b64)
        im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
        img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
        img = tf.convert_to_tensor(img)
        input_image = tf.expand_dims(img, axis=0)
        input_image = tf.image.resize_with_pad(input_image, 192, 192) # 256 for thunder

        out = movenet(input_image)
        return post_proc(out)

    except:
        # todo: log exceptions
        print(f'could not process frame', req.frame_id)