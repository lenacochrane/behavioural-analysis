
import tkinter as tk
from sidehole_gui import SideHoleGui  # Import the HoleGui class from the hole_gui module

if __name__ == "__main__":
    root = tk.Tk()  # Create the main root window
    app = SideHoleGui(root)  # Create an instance of the HoleGui class with the root window
    root.mainloop()  # Start the Tkinter main loop
