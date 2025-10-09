from picamera2 import Picamera2
from libcamera import controls
from datetime import datetime
import RPi.GPIO as GPIO
import time, socket

# --- Constants ---
experiment_duration = 1800  # 30 min
framerate = 40
resolution = (1400, 1400)
exposure = 5000
lensPosition = 12
rig_name = socket.getfqdn()

# --- GPIO Setup ---
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
GPIO.output(yellow_led, GPIO.HIGH)

# --- Picamera Setup ---
picam2 = Picamera2()
video_config = picam2.create_video_configuration({'size': resolution})
picam2.configure(video_config)
picam2.set_controls({
    "AfMode": controls.AfModeEnum.Manual,
    "LensPosition": lensPosition,
    "FrameRate": framerate,
    "ExposureTime": exposure
})

# --- Main Loop ---
while True:
    time.sleep(0.1)
    if GPIO.input(record_pin) == False:
        GPIO.output(red_led, GPIO.HIGH)
        date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = f"/home/adultrig/data/{date}_{rig_name}.h264"
        print("Recording to:", output_path)

        picam2.start_and_record_video(output=output_path, duration=experiment_duration, show_preview=False)

        # Optional: listen for stop button during recording
        start_time = time.time()
        while time.time() - start_time < experiment_duration:
            if GPIO.input(stop_pin) == False:
                time.sleep(0.2)
                if GPIO.input(stop_pin) == False:
                    picam2.stop_recording()
                    break
            time.sleep(0.1)

        if picam2.recording:
            picam2.stop_recording()

        GPIO.output(red_led, GPIO.LOW)
        print("Recording complete")
