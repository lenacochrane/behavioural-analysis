import time
from gpiozero import LED
import picamera
from datetime import datetime
import socket

ir = LED(5)
white_led = LED(6)

rig_name = socket.getfqdn()
date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
video_path = f"/home/startle/data/{date}_{rig_name}.h264"

camera = picamera.PiCamera()
camera.resolution = (1920, 1080)
camera.framerate = 5
camera.rotation = 0

try:
    ir.on()

    camera.start_preview(alpha=100)
    time.sleep(2)

    camera.start_recording(video_path)

    white_led.on()
    time.sleep(10)
    white_led.off()

    time.sleep(24 * 60 * 60)

finally:
    camera.stop_recording()
    camera.stop_preview()
    ir.off()
    white_led.off()
    camera.close()