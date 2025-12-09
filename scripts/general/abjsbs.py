"""
GPIO LED diagnostic script for IR (GPIO5) and white LED (GPIO6).

Run it with:
    python led_diagnostic.py

It will show a menu of tests. Pick a number and press Enter.

Notes:
- Uses BCM numbering (gpiozero default).
- IR is assumed to be on GPIO5.
- White LED on GPIO6.
- Cleans up LEDs on exit.
"""

from gpiozero import LED
from time import sleep, monotonic
import sys

IR_PIN = 5
WHITE_PIN = 6

def make_leds(ir_active_high=False):
    """
    Create LED objects.
    For IR, we may flip active_high to test polarity.
    White LED is assumed normal active_high=True unless you know otherwise.
    """
    ir = LED(IR_PIN, active_high=ir_active_high)
    white = LED(WHITE_PIN)
    return ir, white

def cleanup(ir, white):
    try:
        ir.off()
    except Exception:
        pass
    try:
        white.off()
    except Exception:
        pass

def test_ir_steady(ir, seconds=5):
    print(f"[IR steady] active_high={ir.active_high} -> ON for {seconds}s")
    ir.on()
    sleep(seconds)
    ir.off()
    print("[IR steady] done")

def test_ir_heartbeat(ir, seconds=10, poke_s=0.1):
    print(f"[IR heartbeat] active_high={ir.active_high} -> forcing ON for {seconds}s, poke every {poke_s}s")
    t0 = monotonic()
    while monotonic() - t0 < seconds:
        ir.on()
        sleep(poke_s)
    ir.off()
    print("[IR heartbeat] done")

def test_ir_blink(ir, on_s=1, off_s=1, cycles=5):
    print(f"[IR blink] active_high={ir.active_high} -> {cycles} cycles of ON {on_s}s / OFF {off_s}s")
    for i in range(cycles):
        print(f"  cycle {i+1}/{cycles}: IR ON")
        ir.on()
        sleep(on_s)
        print(f"  cycle {i+1}/{cycles}: IR OFF")
        ir.off()
        sleep(off_s)
    print("[IR blink] done")

def test_white_steady(white, seconds=5):
    print(f"[White steady] ON for {seconds}s")
    white.on()
    sleep(seconds)
    white.off()
    print("[White steady] done")

def test_white_blink(white, on_s=0.5, off_s=0.5, cycles=10):
    print(f"[White blink] {cycles} cycles of ON {on_s}s / OFF {off_s}s")
    for i in range(cycles):
        print(f"  cycle {i+1}/{cycles}: White ON")
        white.on()
        sleep(on_s)
        print(f"  cycle {i+1}/{cycles}: White OFF")
        white.off()
        sleep(off_s)
    print("[White blink] done")

def test_both_ir_constant_white_flash(ir, white, flash_duration=1, interval=3, flashes=5, poke_s=0.2):
    print("[Both] IR forced ON continuously (heartbeat), white flashes")
    print(f"       flashes={flashes}, flash_duration={flash_duration}s, interval={interval}s, IR poke={poke_s}s")
    for i in range(flashes):
        print(f"  flash {i+1}/{flashes}: keep IR alive for interval, then flash white")

        # keep IR alive during interval before flash
        t0 = monotonic()
        while monotonic() - t0 < interval:
            ir.on()
            sleep(poke_s)

        # flash white
        print("    White ON")
        white.on()
        # keep IR alive while white on too
        t1 = monotonic()
        while monotonic() - t1 < flash_duration:
            ir.on()
            sleep(poke_s)
        print("    White OFF")
        white.off()

    ir.off()
    white.off()
    print("[Both] done")

def menu():
    print("\n========== LED DIAGNOSTIC MENU ==========")
    print("IR polarity tests:")
    print(" 1) IR steady ON (active_high=False)  [your usual wiring]")
    print(" 2) IR steady ON (active_high=True)   [polarity flip test]")
    print(" 3) IR heartbeat ON (active_high=False)")
    print(" 4) IR heartbeat ON (active_high=True)")
    print("\nWhite LED tests:")
    print(" 5) White steady ON")
    print(" 6) White blink")
    print("\nBoth LEDs tests:")
    print(" 7) IR constant (heartbeat) + White flashes")
    print("\nExit:")
    print(" 0) Quit")
    print("========================================\n")

def main():
    while True:
        menu()
        choice = input("Pick a test number: ").strip()

        if choice == "0":
            print("Quitting.")
            return

        try:
            if choice == "1":
                ir, white = make_leds(ir_active_high=False)
                try:
                    test_ir_steady(ir)
                finally:
                    cleanup(ir, white)

            elif choice == "2":
                ir, white = make_leds(ir_active_high=True)
                try:
                    test_ir_steady(ir)
                finally:
                    cleanup(ir, white)

            elif choice == "3":
                ir, white = make_leds(ir_active_high=False)
                try:
                    test_ir_heartbeat(ir)
                finally:
                    cleanup(ir, white)

            elif choice == "4":
                ir, white = make_leds(ir_active_high=True)
                try:
                    test_ir_heartbeat(ir)
                finally:
                    cleanup(ir, white)

            elif choice == "5":
                ir, white = make_leds(ir_active_high=False)
                try:
                    test_white_steady(white)
                finally:
                    cleanup(ir, white)

            elif choice == "6":
                ir, white = make_leds(ir_active_high=False)
                try:
                    test_white_blink(white)
                finally:
                    cleanup(ir, white)

            elif choice == "7":
                ir, white = make_leds(ir_active_high=False)
                try:
                    test_both_ir_constant_white_flash(ir, white)
                finally:
                    cleanup(ir, white)

            else:
                print("Unknown choice. Try again.")

        except KeyboardInterrupt:
            # If you Ctrl+C mid-test, we return to menu safely
            print("\n[Interrupted] Returning to menu.\n")
            try:
                cleanup(ir, white)
            except Exception:
                pass

if __name__ == "__main__":
    main()
