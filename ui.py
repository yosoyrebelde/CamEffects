import sys, os
import subprocess
import time
from multiprocessing import Process, Pipe
from threading import Thread
from PyQt5.QtWidgets import (QApplication, QMenu, 
                             QAction, QSystemTrayIcon, 
                             QMessageBox)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QObject, pyqtSignal

import vid

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

def path(file):
    global ROOT_PATH
    return f"{ROOT_PATH}\\{file}"

class ProcManager(QObject):

    procStateChanged = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        # Create pipes
        # To camera_proc
        self.ctrl_camera_child_conn, self.ctrl_camera_parent_conn = Pipe()
        # To manager
        self.ctrl_manag_child_conn, self.ctrl_manag_parent_conn = Pipe()
        # Signals to manager:
        # 0 quit
        # 1 start camera
        # 2 stop camera

        self.camera_proc = None
        self.prev_state = None

        self.manager = Thread(target=self._start_manager)
        self.manager.daemon = True
        self.manager.start()

    def _start_manager(self):
        while True:
            # Get the state of the camera process
            if self.camera_proc is not None:
                state = self.camera_proc.is_alive()
            else:
                state = None
            # Check if the state have been changed
            if state != self.prev_state:
                self.procStateChanged.emit(state)
            self.prev_state = state

            # Check if there is a request to start/stop the camera process
            # or to quit 
            if self.ctrl_manag_child_conn.poll():
                signal = self.ctrl_manag_child_conn.recv()
                if signal == 0:
                    if state:
                        self._stop_camera_proc()
                    break
                elif signal == 1:
                    self._start_camera_proc()
                elif signal == 2:
                    self._stop_camera_proc()
            # Reduce CPU usage
            time.sleep(0.2)

    def _start_camera_proc(self):
        self.camera_proc = Process(target=vid.read, 
                            args=(self.ctrl_camera_child_conn,))
        self.camera_proc.start()

    def _stop_camera_proc(self):
        self.ctrl_camera_parent_conn.send(True)
        if not self._wait_until_timeout(wanted_alive=False):
            # Kill the child process anyway
            self.camera_proc.terminate()

    def start(self):
        self.ctrl_manag_parent_conn.send(1)

    def stop(self):
        self.ctrl_manag_parent_conn.send(2)

    def quit(self):
        self.ctrl_manag_parent_conn.send(0)

    def _wait_until_timeout(self, t=0.5, wanted_alive=True):
        t0 = t1 = time.perf_counter()
        while (t1 - t0 < t):
            if self.camera_proc.is_alive() == wanted_alive:
                return True
            t1 = time.perf_counter()
        return False

class TrayUI:

    def __init__(self):
        self.manager = ProcManager()
        self.manager.procStateChanged.connect(self.change_start_stop)

        self.tray = QSystemTrayIcon()
        self.tray.setIcon(QIcon(path("icon.png")))
        self.tray.setVisible(True)
        
        self.a_start = QAction("Start")
        self.a_start.setIcon(QIcon(path("start.png")))
        self.a_start.triggered.connect(self.start)

        self.a_stop = QAction("Stop")
        self.a_stop.setIcon(QIcon(path("stop.png")))
        self.a_stop.setVisible(False)
        self.a_stop.triggered.connect(self.stop)

        self.a_config = QAction("Open config file")
        self.a_config.triggered.connect(self.open_config_file)

        self.a_quit = QAction("Quit")
        self.a_quit.setIcon(QIcon(path("close.png")))
        self.a_quit.triggered.connect(self.close)

        self.menu = QMenu()
        self.menu.addAction(self.a_start)
        self.menu.addAction(self.a_stop)
        self.menu.addAction(self.a_config)
        self.menu.addAction(self.a_quit)

        self.tray.setContextMenu(self.menu)
        self.menu.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tray.activated.connect(lambda: self.menu.exec_())

    def change_start_stop(self, is_running):
        self.a_start.setVisible(not is_running)
        self.a_stop.setVisible(is_running)
        self.camera_is_running = is_running

    def start(self):

        ### FOR TESTING, DELETE ON RELEASE #
        import importlib
        importlib.reload(vid)
        ###

        self.manager.start()

    def stop(self):
        self.manager.stop()

    def open_config_file(self):
        subprocess.Popen(path('config.py'), shell=True)
        title = "Config has been changed?"
        message = "Relaunch the camera (Stop -> Start) to apply the new configuration."
        self.tray.showMessage(title, message)

    def close(self):
        self.manager.quit()
        QApplication.quit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    ui = TrayUI()
    sys.exit(app.exec_())

'''
class Log:
    """ Writing the console output to file """
    def __init__(self):
        log = path("log.txt")
        if os.path.isfile(log):
            os.remove(log)

        self.stdout = sys.stdout
        self.log = open(log, "w", encoding='utf-8')

    def write(self, msg):
        self.stdout.write(msg)
        #str.decode('utf8')
        self.log.write(msg)

    def flush(self):
        """ The interpreter insists that I really need this method ¯\\_(ツ)_/¯ """
        pass

    def __del__(self):
        """ Let's hope this method will be called. """
        self.log.close()

    ##########################
    def start(self):
        if self.camera_is_running:
            title = "Unable to proceed!"
            message = "The camera is already running."
            self.tray.showMessage(title, message)
            return 0
        #self.queue = Queue()
        #self.proc = Process(target=CameraProc, args=(self.queue,))
        self.camera_proc = CameraProc()
        self.camera_proc.procStateChanged.connect(self.change_start_stop)
        self.camera_proc.start()

        #self.proc = SignalizedProcess(target=CameraProc, args=(self.queue,))
        #self.proc.target_obj.signal.procStateChanged.connect(self.change_start_stop)
        #self.proc.start()
        if not self.wait_until_timeout():
            msg = QMessageBox()
            msg.setWindowTitle("Unknown error!")
            msg.setText("The camera cannot be started.\n"
                        "Check log for more information.")
            msg.exec()
            return 0

    def stop(self):
        if not self.camera_is_running:
            title = "Unable to proceed!"
            message = "The camera is already stopped."
            self.tray.showMessage(title, message)
            return 0
        self.camera_proc.parent_conn.send(True)
        if not self.wait_until_timeout(wanted_alive=False):
            # Kill the child process anyway
            self.camera_proc.proc.terminate()
            msg = QMessageBox()
            msg.setWindowTitle("Unknown error!")
            msg.setText("Some troubles appeared while stopping the camera.\n"
                        "Check log for more information.")
            msg.exec()
        del self.camera_proc
'''

