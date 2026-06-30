import time
from time import sleep
from time import strftime
import RPi.GPIO as GPIO
from datetime import datetime
import socket
import subprocess
from gpiozero import LED
import csv
import picamera

###################
#### LED SETUP ####
###################
ir = LED(5)
ir.off()

white_led = LED(6)
white_led.off()

###################
#### PIN SETUP ####
###################
shutdown_pin = 2
record_pin = 27
stop_pin = 22
red_led = 23
yellow_led = 24

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

GPIO.setup(shutdown_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(record_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(stop_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

GPIO.setup(red_led, GPIO.OUT)
GPIO.setup(yellow_led, GPIO.OUT)

GPIO.output(red_led, GPIO.LOW)
GPIO.output(yellow_led, GPIO.LOW)

#######################
#### CAMERA SET UP ####
#######################
rig_name = socket.getfqdn()

camera = picamera.PiCamera()
camera.resolution = (1920, 1080)
camera.framerate = 5
camera.rotation = 0  # change if needed

##############################
#### EXPERIMENT INTERVALS ####
##############################
initial_interval = 600 # 10 min
n_exp = 20
flash_interval = 2
habituation = 30 # 30s between flashes
habituation_memory_interval = 600

try:
    while True:
        time.sleep(0.2)

        if GPIO.input(record_pin) == False: # if record button is pressed
            GPIO.output(red_led, GPIO.HIGH)

            camera.start_preview(alpha=240)   # optional
            sleep(2)                 # allow camera to warm up

            date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            fps = 5
            t0 = time.monotonic()
            flash_frames = []

            video_path = f"/home/startle/data/{date}_{rig_name}.h264"
            csv_path = f"/home/startle/data/{date}_{rig_name}.csv"

            camera.start_recording(video_path)
            ir.on()

            try:

                sleep(initial_interval)

                for i in range(n_exp):

                    start_frame = int((time.monotonic() - t0) * fps)
                    flash_frames.append(start_frame)

                    white_led.on()
                    sleep(flash_interval)
                    white_led.off()

                    sleep(habituation)

                    # ---- STOP BUTTON ----
                    if GPIO.input(stop_pin) == False:
                        time.sleep(0.2)
                        if GPIO.input(stop_pin) == False:
                            break

                sleep(habituation_memory_interval)

                start_frame = int((time.monotonic() - t0) * fps)
                flash_frames.append(start_frame)

                white_led.on()
                sleep(flash_interval)
                white_led.off()
                sleep(habituation)


            finally:
                camera.stop_recording()
                camera.stop_preview()

                GPIO.output(red_led, GPIO.LOW)
                ir.off()
                white_led.off()

                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["flash_start_frame"])
                    for frame in flash_frames:
                        writer.writerow([frame])

finally:
    camera.close()
    GPIO.cleanup()