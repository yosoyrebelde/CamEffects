import sys, os
import cv2
import ctypes
import pyvirtualcam
import importlib
import numpy as np
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
from numpy.ctypeslib import ndpointer

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
    mask = cv2.GaussianBlur(mask, (21,21), 12, 12)
    
    # Restore the size
    mask = cv2.pyrUp(mask)
    return mask

def soft_contour(img, background, mask):

    # Form the list of alphas
    da = cura = 0.2
    alpha = []
    while cura < 1:
        alpha.append(cura)
        cura += da

    # Form a contour for every alpha
    blur_ksize = (4, 4)
    blurred_img = cv2.blur(img, blur_ksize)

    ksize = (3, 3)
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    final_contour = np.zeros(img.shape, np.uint8)

    for a in alpha:

        inner_mask = cv2.erode(mask, ker)
        contour_mask = xor_mask(mask, inner_mask)
        mask = inner_mask

        contour_img = cv2.bitwise_and(blurred_img, blurred_img, mask=contour_mask)
        contour_bg = cv2.bitwise_and(background, background, mask=contour_mask)
        contour = cv2.addWeighted(contour_img, a, contour_bg, 1-a, 0)

        final_contour = cv2.add(final_contour, contour)

    # Apply the mask and add the contour to the image
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    masked_img = cv2.add(masked_img, final_contour)

    return masked_img

def mashup(frame, background):
    # Predict (get mask)
    result = MODEL.predict_single(frame)
    mask = result.get_mask(
        threshold=CONF['mask_threshold']
        ).numpy().astype(np.uint8)

    mask = mask_smoothing(mask)

    # Get masked background
    masked_background = cv2.bitwise_and(background,
                                        background,
                                        mask=inverse_mask(mask))

    # Get masked frame
    #masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    masked_frame = soft_contour(frame, background, mask)

    # Put all together
    final = cv2.add(masked_frame, masked_background)
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

    cam_orig.release()
    cv2.destroyAllWindows()
    #conn.send(True)

'''
def odd(num):
    if num < 0:
        return 0
    num = round(num)
    if num > 2:
        if not num % 2:
            return num - 1
    return num

def diff_filter(array, prev_array):
    #diff = np.absolute(array - prev_array)
    diff = array - prev_array
    diff = np.where(diff==-1, 1, diff).astype(np.uint8)
    mask = cv2.bitwise_or(array, prev_array, mask=inverse_mask(diff)).reshape(diff.shape)
    #rand = np.random.randint(0, 2, diff.shape)
    #diff = cv2.bitwise_and(rand, rand, mask=diff)
    out = np.add(mask, diff)
    return out

def soft_contour(img, background, mask, ksize, alpha=0.5):
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    inner_mask = cv2.erode(mask, ker)
    contour_mask = xor_mask(mask, inner_mask)

    masked_img = cv2.bitwise_and(img, img, mask=inner_mask)
    contour = cv2.addWeighted(img, alpha, background, 1-alpha, 0)
    contour = cv2.bitwise_and(contour, contour, mask=contour_mask)
    masked_img = cv2.add(masked_img, contour)
    return masked_img, inner_mask

def brightness_corr(img):
    max_intensity = 255.0
    phi = 1
    theta = 1
    new_img = (max_intensity/phi)*(img/(max_intensity/theta))**2
    new_img = np.array(new_img,dtype=np.uint8)
    return new_img

'''
