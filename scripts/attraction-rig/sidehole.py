
import tkinter as tk
from hole_gui import HoleGui  # Import the HoleGui class from the hole_gui module
from hole_analysis import HoleAnalysis 
import os 
import seaborn as sns
import matplotlib.pyplot as plt


# call function to run the gui 
def run_gui():
    root = tk.Tk()  # Create the main root window
    app = HoleGui(root)  # Create an instance of the SideHoleGui class with the root window
    root.mainloop()  # Start the Tkinter main loop
    print("GUI closed.")


def perform_analysis(directory):
    if not os.path.isdir(directory):
        print(f"The directory {directory} does not exist.")
        return

    print("Starting analysis...")
    analysis = HoleAnalysis(directory)

    analysis.returns()

    print("Analysis Completed")


if __name__ == "__main__":

    run_gui_option = input("Do you want to run the GUI for hole drawing? (y/n): ").strip().lower()
    if run_gui_option == 'y':
        run_gui()

    perform_analysis("/Volumes/lab-windingm/home/users/cochral/AttractionRig/modelling-behaviour/michael-test-sleap-extrac/ptc")




