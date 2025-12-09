from gpiozero import LED, PWMLED
from picamera2 import Picamera2, Preview
import time
import numpy as np
import datetime as dt
import os

# ---------------- SETTINGS YOU CAN EDIT ----------------
IR_PIN = 5             # BCM numbering
WHITE_PIN = 6          # BCM numbering (kept OFF)
TOGGLE_SECONDS = 3
N_TOGGLES = 4          # number of OFF->ON cycles per test

DO_PWM_SWEEP = True
PWM_LEVELS = [0.0, 0.1, 0.2, 0.4, 0.7, 1.0]
PWM_HOLD = 3           # seconds to hold each PWM level

SAVE_FRAMES = False
SAVE_DIR_BASE = "/home/sv18/Desktop/data"
# ------------------------------------------------------

timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = os.path.join(SAVE_DIR_BASE, f"ir_diag_{timestamp}")

def mean_brightness(cam, tag=None):
    frame = cam.capture_array()
    bright = float(frame.mean(axis=2).mean())
    if SAVE_FRAMES and tag:
        os.makedirs(SAVE_DIR, exist_ok=True)
        np.save(os.path.join(SAVE_DIR, f"{tag}.npy"), frame)
    return bright

def set_ir_pwm(ir_pwm, value):
    value = max(0.0, min(1.0, value))
    ir_pwm.value = value

def ir_toggle_test(cam, ir_device, label):
    print("\n==============================")
    print(f"{label}")
    print("==============================")
    print(f"Doing {N_TOGGLES} cycles of IR OFF -> ON.")
    deltas = []

    for i in range(N_TOGGLES):
        # OFF
        if isinstance(ir_device, PWMLED):
            set_ir_pwm(ir_device, 0.0)
        else:
            ir_device.off()
        time.sleep(0.4)
        b_off = mean_brightness(cam, tag=f"{label}_cycle{i}_off" if SAVE_FRAMES else None)
        print(f" Cycle {i}: IR OFF brightness = {b_off:.2f}")
        time.sleep(TOGGLE_SECONDS)

        # ON (full)
        if isinstance(ir_device, PWMLED):
            set_ir_pwm(ir_device, 1.0)
        else:
            ir_device.on()
        time.sleep(0.4)
        b_on = mean_brightness(cam, tag=f"{label}_cycle{i}_on" if SAVE_FRAMES else None)
        print(f" Cycle {i}: IR ON  brightness = {b_on:.2f}")
        delta = b_on - b_off
        deltas.append(delta)
        print(f"   Δ (ON-OFF) = {delta:.2f}\n")
        time.sleep(TOGGLE_SECONDS)

    avg_delta = float(np.mean(deltas)) if deltas else 0.0
    print(f"Average Δ brightness = {avg_delta:.2f}")

    if avg_delta > 2.0:
        print("✅ Camera sees a strong IR effect here.")
    elif avg_delta > 0.5:
        print("⚠️ Weak IR effect here.")
    else:
        print("❌ No detectable IR effect here.")

    return avg_delta, deltas

def lock_exposure(cam):
    meta = cam.capture_metadata()
    exp = meta.get("ExposureTime", 8000)
    gain = meta.get("AnalogueGain", 1.0)
    cam.set_controls({
        "AeEnable": False,
        "ExposureTime": exp,
        "AnalogueGain": gain
    })
    print(f"Locked exposure at ExposureTime={exp}, AnalogueGain={gain}")

def unlock_exposure(cam):
    cam.set_controls({"AeEnable": True})
    print("Auto-exposure ENABLED.")

def pwm_sweep_test(cam, ir_pwm):
    print("\n==============================")
    print("PWM SWEEP (dim → bright)")
    print("==============================")
    print("This tests whether IR output scales with duty cycle.")
    sweep = []

    for lvl in PWM_LEVELS:
        set_ir_pwm(ir_pwm, lvl)
        time.sleep(0.5)
        b = mean_brightness(cam, tag=f"pwm_{int(lvl*100):03d}" if SAVE_FRAMES else None)
        sweep.append((lvl, b))
        print(f" PWM {lvl:.2f} -> brightness {b:.2f} (hold {PWM_HOLD}s)")
        time.sleep(PWM_HOLD)

    set_ir_pwm(ir_pwm, 0.0)
    print("\nSweep results:")
    for lvl, b in sweep:
        print(f"  {lvl:.2f}: {b:.2f}")

    bs = [b for _, b in sweep]
    span = max(bs) - min(bs) if bs else 0.0
    print(f"Brightness span across PWM levels = {span:.2f}")

    if span > 2.0:
        print("✅ Brightness follows PWM → IR is producing light.")
    elif span > 0.5:
        print("⚠️ Slight PWM response → IR weak or partly blocked.")
    else:
        print("❌ No PWM response → IR likely not producing or not reaching view.")

def pause(msg):
    input(f"\n--- {msg} ---\nPress ENTER to continue...\n")

def main():
    # LEDs
    white_led = LED(WHITE_PIN)
    white_led.off()

    # Use PWMLED for IR so we can dim/sweep.
    ir_pwm = PWMLED(IR_PIN)
    set_ir_pwm(ir_pwm, 0.0)

    # Camera + preview
    cam = Picamera2()
    config = cam.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
    cam.configure(config)
    cam.start_preview(Preview.QTGL)
    cam.start()
    time.sleep(1.5)

    print("\nIR DIAGNOSTIC STARTED.")
    print("Preview window should be open now.")
    print("White LED stays OFF for all tests.\n")

    try:
        # STEP 0: baseline
        unlock_exposure(cam)
        pause("STEP 1: Keep your setup EXACTLY as it currently is (arena ON). We’ll test IR with AUTO-EXPOSURE ON")

        avg1, _ = ir_toggle_test(cam, ir_pwm, "TEST 1 (Arena ON, AE ON)")

        pause("STEP 2: Do NOT move anything. We’ll now lock exposure and repeat.\nIf AE was hiding IR, you’ll SEE it now.")

        lock_exposure(cam)
        avg2, _ = ir_toggle_test(cam, ir_pwm, "TEST 2 (Arena ON, AE LOCKED)")

        pause("STEP 3: WITHOUT stopping preview, REMOVE the arena/dish so camera sees straight down.\nThis tests if arena is blocking IR.")

        # keep AE locked for a fair comparison
        avg3, _ = ir_toggle_test(cam, ir_pwm, "TEST 3 (Arena REMOVED, AE LOCKED)")

        pause("STEP 4: Put the arena back ON.\nWe’ll do a PWM brightness sweep so you can watch scaling.")

        if DO_PWM_SWEEP:
            pwm_sweep_test(cam, ir_pwm)

        print("\n==============================")
        print("FINAL INTERPRETATION GUIDE")
        print("==============================")
        print(f"Test1 avg Δ (arena on, AE on)     = {avg1:.2f}")
        print(f"Test2 avg Δ (arena on, AE locked) = {avg2:.2f}")
        print(f"Test3 avg Δ (arena removed)       = {avg3:.2f}\n")

        if avg3 > 2.0 and avg2 < 0.5:
            print("➡️ IR is being produced, but ARENA is blocking/absorbing it.")
        elif avg3 > 2.0 and avg2 > 2.0:
            print("➡️ IR is strong and reaching camera. If preview looked same before, AE/preview tone-mapping hid it.")
        elif avg3 < 0.5:
            print("➡️ IR is NOT reaching view even with arena removed → power/wiring/IR board output issue.")
        else:
            print("➡️ Mixed result. Use PWM sweep trend to decide weak-IR vs partial blocking.")

        if SAVE_FRAMES:
            print(f"\nFrames saved in: {SAVE_DIR}")

    finally:
        set_ir_pwm(ir_pwm, 0.0)
        white_led.off()
        cam.stop()
        cam.stop_preview()
        cam.close()

if __name__ == "__main__":
    main()
