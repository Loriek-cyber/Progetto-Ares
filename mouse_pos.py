"""
mouse_pos.py — Utility per trovare le coordinate del mouse
Muovi il mouse sulla posizione desiderata e premi SPAZIO per registrarla.
Premi ESC per uscire.
"""
import time
try:
    import pyautogui
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyautogui"])
    import pyautogui

try:
    import keyboard
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "keyboard"])
    import keyboard

print("=" * 50)
print("  MOUSE POSITION TRACKER")
print("=" * 50)
print("  Muovi il mouse e premi:")
print("  [SPAZIO] → registra posizione corrente")
print("  [ESC]    → esci")
print("=" * 50)

saved = []

while True:
    x, y = pyautogui.position()
    print(f"\r  Pos: X={x:5d}  Y={y:5d}    ", end="", flush=True)

    if keyboard.is_pressed("space"):
        saved.append((x, y))
        print(f"\n  >>> SALVATA #{len(saved)}: X={x}, Y={y}")
        time.sleep(0.3)  # debounce

    if keyboard.is_pressed("esc"):
        break

    time.sleep(0.05)

print("\n\n--- Posizioni registrate ---")
for i, (x, y) in enumerate(saved, 1):
    print(f"  #{i}: pyautogui.click({x}, {y})")
print("\nCopia le coordinate nel file utils/driver.py → AC_RESTART_CLICK_POS")
