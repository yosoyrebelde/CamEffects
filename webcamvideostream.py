'''Use threading to increase webcam FPS
Based on https://github.com/jrosebr1/imutils/blob/master/imutils/video/webcamvideostream.py
'''

from threading import Thread
import cv2
import time

class WebcamVideoStream:
    def __init__(self, src=0, name="WebcamVideoStream", flags=[], print_fps=False):
        # flags -- list, opencv videoio flag sets,
        # e.g. [('CAP_PROP_FRAME_WIDTH', 1280),]

        # initialize the video camera stream and 
        self.stream = cv2.VideoCapture(src)

        # Set flags
        # Use 'CAP_PROP_FPS' flag to initialize a fps limiter
        self.fpslimiter = lambda: None
        for flag in flags:
            exec(f'self.stream.set(cv2.{flag[0]}, {flag[1]})')
            if 'CAP_PROP_FPS' in flag:
                if flag[1] > 0:
                    delay = 1 / (float(flag[1]) + 0.05)
                    self.fpslimiter = lambda: time.sleep(delay)

        # read the first frame from the stream
        self.grabbed, self.frame = self.stream.read()

        # initialize the fps counter
        self.fpscounter = FPSCounter.on_off(print_fps)
        self.fpscounter.frame_received()

        # initialize the thread name
        self.name = name

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                self.stream.release()
                cv2.destroyAllWindows()
                return

            # otherwise, read the next frame from the stream
            self.grabbed, self.frame = self.stream.read()
            self.fpscounter.frame_received()
            self.fpslimiter()

    def read(self):
        # return the frame most recently read
        self.fpscounter.frame_used()
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def __enter__(self):
        return self.start()

    def __exit__(self, type, value, tb):
        self.stop()


class FPSCounter:
    @classmethod
    def on_off(cls, on=True):
        def dummy(self):
            pass
        if not on:
            callable_attrs = {k:v for k, v in cls.__dict__.items() 
                                   if callable(v)}
            for name, _ in callable_attrs.items():
                setattr(cls, name, dummy)
        return cls()

    def __init__(self):
        self.dump()

    def frame_received(self):
        self.fps_received += 1
        self.print_every_sec()

    def frame_used(self):
        self.fps_used += 1
        self.print_every_sec()

    def print_every_sec(self):
        t1 = time.perf_counter()
        if t1 - self.t0 >= 1.:
            print(f'FPS received {self.fps_received}, used {self.fps_used}')
            self.dump()

    def dump(self):
        self.fps_received = 0
        self.fps_used = 0
        self.t0 = time.perf_counter()
