from pynput import mouse
from datetime import datetime

def on_click(x, y, button, pressed):
    """Print mouse click events with timestamp and coordinates."""
    event_type = "Pressed" if pressed else "Released"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{event_type:<8} {button:<10} at (x={x}, y={y})\n")

def main():
    print("🖱️  Mouse click tracker started. Press Ctrl+C to stop.\n")
    try:
        # Start listening for clicks
        with mouse.Listener(on_click=on_click) as listener:
            listener.join()
    except KeyboardInterrupt:
        print("\nStopped tracking clicks.")

if __name__ == "__main__":
    main()
