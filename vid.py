import sys, os
import cv2
import ctypes
import pyvirtualcam
import importlib
import numpy as np
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
from numpy.ctypeslib import ndpointer

import config # PATH?
from webcamvideostream import WebcamVideoStream


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

def inverse_mask(mask):
    res = np.absolute(mask - 1).astype(np.uint8)
    return res

def xor_mask(mask1, mask2):
    res = np.absolute(mask1 - mask2).astype(np.uint8)
    return res

def blur_effect_():
    pass

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

    #array = cv2.pyrDown(array)
    #prev_array = cv2.pyrDown(prev_array)
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
    #out_array = cv2.pyrUp(out_array)
    return out_array

def mask_smoothing(mask):

    def odd(num):
        """ Return the nearest positive odd number """
        if num < 1:
            return 1
        num = round(num)
        if not num % 2:
            return num - 1
        return num

    # Resize (just for optimization)
    mask = cv2.pyrDown(mask)

    # Filter by current and previous frame
    if 'prev_mask' in globals():
        f_mask = move_filter(mask,
                             globals()['prev_mask'],
                             CONF['mfilt_kernel'],
                             CONF['mfilt_threshold'])
        mask = f_mask
    globals()['prev_mask'] = mask

    # Smooth after move_filter
    mask = cv2.boxFilter(mask, -1, CONF['mfilt_kernel'])

    gker = CONF['mfilt_kernel']
    gker = odd(gker[0]/2), odd(gker[1]/2)
    sx, sy = gker[0] * 5, gker[1] * 5
    mask = cv2.GaussianBlur(mask, gker, sx, sy)

    gker2 = gker[0] * 7, gker[1] * 7
    sx, sy = gker[0] * 4, gker[1] * 4
    mask = cv2.GaussianBlur(mask, gker2, sx, sy)
    
    # Restore the size
    mask = cv2.pyrUp(mask)
    return mask

def soft_contour(img, background, mask):

    mask = np.copy(mask)
    masked_bg = cv2.bitwise_and(background, background, 
                                mask=inverse_mask(mask))

    # Form the list of alphas
    da = cura = 0.2
    alpha = []
    while cura < 1:
        alpha.append(cura)
        cura += da

    # Form a contour for every alpha
    #blur_ksize = (4, 4)
    #blurred_img = cv2.blur(img, blur_ksize)

    ksize = (3, 3)
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    final_contour = np.zeros(img.shape, np.uint8)

    for a in alpha:

        inner_mask = cv2.erode(mask, ker)
        contour_mask = xor_mask(mask, inner_mask)
        mask = inner_mask

        contour_img = cv2.bitwise_and(img, img, mask=contour_mask)
        contour_bg = cv2.bitwise_and(background, background, mask=contour_mask)
        contour = cv2.addWeighted(contour_img, a, contour_bg, 1-a, 0)

        final_contour = cv2.add(final_contour, contour)

    # Apply the mask and add the contour to the image
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    masked_img = cv2.add(masked_img, final_contour)

    # Add the background
    final = cv2.add(masked_img, masked_bg)

    return final

def mashup(frame, background):
    # Predict (get mask)
    result = MODEL.predict_single(frame)
    mask = result.get_mask(
        threshold=CONF['mask_threshold']
        ).numpy().astype(np.uint8)

    mask = mask_smoothing(mask)

    #masked_background = cv2.bitwise_and(background,
    #                                    background,
    #                                    mask=inverse_mask(mask))
    #masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    #final = cv2.add(masked_frame, masked_background)

    final = soft_contour(frame, background, mask)
    
    return final

def read(conn):

    def check_to_stop(conn):
        stop = False
        if conn.poll():
            if conn.recv():
                stop = True
        return stop

    global CONF
    read_config()

    flags = [
        ('CAP_PROP_FPS', CONF['fps']),
        ('CAP_PROP_FRAME_WIDTH', CONF['width']),
        ('CAP_PROP_FRAME_HEIGHT', CONF['height']),
    ]

    if CONF['background_img']:
        background = cv2.resize(
            cv2.imread(CONF['background_img']),
            (CONF['width'], CONF['height'])
        )

    with WebcamVideoStream(flags=flags, 
                           print_fps=False) as cam_orig:
        with pyvirtualcam.Camera(CONF['width'],
                                 CONF['height'],
                                 CONF['fps'],
                                 print_fps=False,
                                 fmt=pyvirtualcam.PixelFormat.BGR) as cam:
            while True:
                if check_to_stop(conn):
                    break
                
                image = cam_orig.read()

                if CONF['mirror']:
                    new_image = cv2.flip(new_image, 1)

                if not CONF['background_img']:
                    if CONF['background_blur']:
                        background = cv2.blur(
                            image,
                            (CONF['background_blur'], CONF['background_blur'])
                        )
                    else:
                        background = image

                new_image = mashup(image, background)

                cam.send(new_image)
                cam.sleep_until_next_frame()
            
                if CONF['imshow']:
                    cv2.imshow("Press 'q' to close", new_image)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
'''
def soft_contour(img, background, mask):
    # Downsize all

    # Get masked background
    masked_background = cv2.bitwise_and(background,
                                        background,
                                        mask=inverse_mask(mask))
    img_orig = np.copy(img)
    mask_orig = np.copy(mask)

    # Resize (just for optimization)
    #size_orig = img.shape[1], img.shape[0]
    img = cv2.pyrDown(img)
    background = cv2.pyrDown(background)
    mask = cv2.pyrDown(mask)

    # Form the list of alphas
    da = cura = 0.2
    alpha = []
    while cura < 1:
        alpha.append(cura)
        cura += da

    # Form a contour for every alpha
    #blur_ksize = (4, 4)
    #blurred_img = cv2.blur(img, blur_ksize)

    ksize = (3, 3)
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    #final_contour = np.zeros(img.shape, np.uint8)

    for a in alpha:

        inner_mask = cv2.erode(mask, ker)
        contour_mask = xor_mask(mask, inner_mask)
        mask = inner_mask

        contour_img = cv2.bitwise_and(img, img, mask=contour_mask)
        contour_bg = cv2.bitwise_and(background, background, mask=contour_mask)
        contour = cv2.addWeighted(contour_img, a, contour_bg, 1-a, 0)

        background = cv2.bitwise_and(background,
                                     background,
                                     mask=inverse_mask(contour_mask))
        background = cv2.add(background, contour)

    # Restore the size
    mask = cv2.pyrUp(mask)
    background = cv2.pyrUp(background)

    contour_mask = xor_mask(mask_orig, mask)
    final_contour = cv2.bitwise_and(background, background, mask=contour_mask)

    # Apply the mask and add the contour to the image
    masked_img = cv2.bitwise_and(img_orig, img_orig, mask=mask)

    final = cv2.add(masked_background, final_contour)
    final = cv2.add(final, masked_img)

    return final
'''
