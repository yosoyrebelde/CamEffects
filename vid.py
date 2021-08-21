import cv2
import ctypes
import os
import pyvirtualcam
import importlib
import numpy as np
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
from numpy.ctypeslib import ndpointer
from queue import Empty as EmptyQueueException

import config # PATH?

class CShape(ctypes.Structure):
    _fields_ = [('y', ctypes.c_int),
                ('x', ctypes.c_int)]

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL = load_model(download_model(BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16))

# import mfilter from c lib
mfilter_dll = ctypes.CDLL(ROOT_PATH + '/mfilter.dll')
mfilter = mfilter_dll.mfilter
ptr_2d = np.ctypeslib.ndpointer(dtype=np.uintp)
mfilter.argtypes = [ptr_2d, ptr_2d, ptr_2d,
                    ctypes.POINTER(CShape),
                    ctypes.POINTER(CShape),
                    ctypes.c_float] 
mfilter.restype = None

def read_config():
    global CONF
    importlib.reload(config)
    CONF = config.config

def move_filter(array, prev_array, conv_shape, threshold=0.5):
    ''' 
    conv_shape: (y, x) -- subarrays shape
    threshold: float from 0 to 1 (0--old, 1--new)
    '''
    def ptr(arr):
        arr_pp = (arr.__array_interface__['data'][0]
                  + np.arange(arr.shape[0]) * arr.strides[0]
                 ).astype(np.uintp)
        return arr_pp

    array = cv2.pyrDown(array)
    prev_array = cv2.pyrDown(prev_array)
    shape_orig = array.shape
    shape_2d = (array.shape[0], array.shape[1])
    array = np.reshape(array, shape_2d).astype(ctypes.c_int)
    prev_array = np.reshape(prev_array, shape_2d).astype(ctypes.c_int)

    out_array = np.zeros(shape_2d, dtype=ctypes.c_int)
    arr_size = CShape(*shape_2d)
    conv_size = CShape(*conv_shape)

    mfilter(ptr(array), ptr(prev_array), ptr(out_array),
            ctypes.byref(arr_size), ctypes.byref(conv_size),
            ctypes.c_float(threshold))

    out_array = np.reshape(out_array, shape_orig).astype(np.uint8)
    out_array = cv2.pyrUp(out_array)

    return out_array

def mask_smoothing(mask):

    def change(ker, sX, sY):
        ker = (ker[0] * 3, ker[1] * 3)
        sX = sX - 1
        sY = sY - 1
        return ker, sX, sY

    if 'prev_mask' in globals():
        f_mask = move_filter(mask,
                             globals()['prev_mask'],
                             CONF['mfilt_kernel'],
                             CONF['mfilt_threshold'])
        mask = np.copy(f_mask) #?
    globals()['prev_mask'] = mask

    ker = CONF['mfilt_kernel']
    sX, sY = ker[1], ker[0]
    mask = cv2.GaussianBlur(mask, ker, sX, sY, cv2.BORDER_ISOLATED)
    ker, sX, sY = change(ker, sX, sY)
    mask = cv2.GaussianBlur(mask, ker, sX, sY, cv2.BORDER_ISOLATED)
    ker, sX, sY = change(ker, sX, sY)
    mask = cv2.GaussianBlur(mask, ker, sX, sY, cv2.BORDER_REFLECT_101)

    return mask

def inverse_mask(mask):
    neg = np.add(mask, -1)
    inverse = np.where(neg==-1, 1, neg).astype(np.uint8)
    return inverse

def mashup(frame, background):
    # Predict
    result = MODEL.predict_single(frame)
    mask = result.get_mask(
        threshold=CONF['mask_threshold']
        ).numpy().astype(np.uint8)
    mask = mask_smoothing(mask)

    # Get masked frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Get masked background
    masked_background = cv2.bitwise_and(background,
                                        background,
                                        mask=inverse_mask(mask))
    
    # Put all together
    final = cv2.add(masked_frame, masked_background)
    return final

def blur_background(image, blur_size):
    bg = cv2.blur(image, (blur_size, blur_size))
    return bg

def read(conn):
    global CONF
    read_config()

    cam_orig = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam_orig.set(cv2.CAP_PROP_FRAME_WIDTH, CONF['width'])
    cam_orig.set(cv2.CAP_PROP_FRAME_HEIGHT, CONF['height'])
    cam_orig.set(cv2.CAP_PROP_FPS, CONF['fps'])
    
    if CONF['background_img']:
        background = cv2.resize(
            cv2.imread(CONF['background_img']),
            (CONF['width'], CONF['height'])
        )

    with pyvirtualcam.Camera(CONF['width'],
                             CONF['height'],
                             CONF['fps'],
                             #print_fps=True,
                             fmt=pyvirtualcam.PixelFormat.BGR) as cam:
        while cam_orig.isOpened():
            if check_to_stop(conn):
                break
            
            retval, image = cam_orig.read()
            #if not retval:
            #    raise RuntimeError('Error fetching frame')
            if not CONF['background_img']:
                if CONF['background_blur']:
                    background = blur_background(image, CONF['background_blur'])
                else:
                    background = image

            new_image = mashup(image, background)
            if CONF['mirror']:
                new_image = cv2.flip(new_image, 1)
            cam.send(new_image)
            cam.sleep_until_next_frame()
        
            if CONF['imshow']:
                cv2.imshow("Press 'q' to close", new_image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cam_orig.release()
    cv2.destroyAllWindows()
    conn.send(True)

def check_to_stop(conn):
    stop = False
    if conn.poll():
        if conn.recv():
            stop = True
    return stop
