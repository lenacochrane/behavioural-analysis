
import tkinter as tk
from sidehole_gui import SideHoleGui  # Import the HoleGui class from the hole_gui module
from sidehole_analyse import Side_hole_analysis 
import os 
import seaborn as sns
import matplotlib.pyplot as plt


# call function to run the gui 
def run_gui():
    root = tk.Tk()  # Create the main root window
    app = SideHoleGui(root)  # Create an instance of the SideHoleGui class with the root window
    root.mainloop()  # Start the Tkinter main loop
    print("GUI closed.")


def perform_analysis(directory):
    if not os.path.isdir(directory):
        print(f"The directory {directory} does not exist.")
        return

    print("Starting analysis...")
    analysis = Side_hole_analysis(directory)

    analysis.distance_from_hole()
    print("Analysis Completed")


if __name__ == "__main__":

    run_gui_option = input("Do you want to run the GUI for hole drawing? (y/n): ").strip().lower()
    if run_gui_option == 'y':
        run_gui()

    perform_analysis("/Volumes/lab-windingm/home/users/cochral/AttractionRig/modelling-behaviour/extract/test-for-script")







