import time
from time import sleep
from time import strftime
import RPi.GPIO as GPIO
from picamera import PiCamera
from datetime import datetime
import socket
import subprocess

experiment_duration = 605
Framerate = 2
rig_name = socket.getfqdn()

#exp='Test'
#lighting = 'white'
#lighting_paradigm = 'ON'
#experiment_name = parent_directory + '/' + date + '_' + lighting + '_' + lighting_paradigm + '_'+ exp
#experiment_name = parent_directory 

# Pin definitions
shutdown_pin = 2
record_pin = 27
stop_pin = 22
red_led = 23
yellow_led = 24

# Suppress warnings
GPIO.setwarnings(False)

# Use "GPIO" pin numbering
GPIO.setmode(GPIO.BCM)

# Set up GPIO input pins with internal pull-up resistors (Push buttons)
GPIO.setup(shutdown_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(record_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(stop_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Set up GPIO output pins (LED's)
GPIO.setup(red_led, GPIO.OUT)
GPIO.setup(yellow_led, GPIO.OUT)

# Start with both LED's off
GPIO.output(red_led,GPIO.LOW)
GPIO.output(yellow_led,GPIO.LOW)

#Setup the camera prefences
camera = PiCamera()
camera.resolution = (1300, 1300)
camera.framerate = Framerate
camera.rotation = 90
#camera.color_effects=(128,128) #Black and white

# modular function to shutdown Pi
def shut_down():
    GPIO.output(yellow_led, GPIO.HIGH) # Turn on yellow LED to indicate shutdown process
    print("shutting down")
    command = "/usr/bin/sudo /sbin/shutdown -h now"
    try:
        subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    except Exception as e:
        print(f"Error shutting down: {e}")


try:
    while True:
         #short delay, otherwise this code will take up a lot of the Pi's processing power
        time.sleep(0.2)

        if GPIO.input(record_pin) == False:
            GPIO.output(red_led,GPIO.HIGH) # Red LED ON to indicate the device is recording
            camera.start_preview(alpha=220)
            
            #Labels the recording year-month-day_hour-minute-second
            date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            #Saves the recording into the experimens folder
            camera.start_recording(f"/home/topdown/data/{date}_{rig_name}.h264")

            # implement experiment duration accurately, while allowing the stop button to stop acquisition if desired
            start_time = time.time()
            while time.time() - start_time < experiment_duration:
                camera.wait_recording(0.1)  # Wait for a short period
                if GPIO.input(stop_pin) == False:

                    # Debounce the button, make sure you don't accidentally stop acquisition with a short press
                    time.sleep(0.2)  
                    if GPIO.input(stop_pin) == False:
                        break  # Stop recording if button is pressed

            camera.stop_recording()
            camera.stop_preview()
            GPIO.output(red_led,GPIO.LOW) # Red LED OFF to indicate the recording stopped
            

        # Check button if we want to shutdown the Pi safely
        if GPIO.input(shutdown_pin) == False:
        # Debounce the button, makes sure you don't accidentally shut down the device with a short press
            time.sleep(0.2) 
            if GPIO.input(shutdown_pin) == False:
                shut_down()
finally:
    GPIO.cleanup()
