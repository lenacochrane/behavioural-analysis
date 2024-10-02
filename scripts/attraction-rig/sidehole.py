
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

    # analysis.post_processing()

    analysis.hole_boundary(scale_factor=1.5)

    #METHOD TO SHORTEN
    # analysis.shorten(frame=100)
    
    # CALL METHODS FOR LARVAEL BEHAVIOUR 
    # analysis.trajectory()
    # analysis.time_average_msd(list(range(1, 101, 1)))
    # analysis.speed()
    # analysis.ensemble_msd()
    # analysis.acceleration()
    # analysis.euclidean_distance()
    # analysis.euclidean_distance_variance(200, 600) # currently plotting the plataeu but arbitary time start for plateu 
    # analysis.distance_from_centre()
    # analysis.proximity()

    # DIGGING IN ABSENCE OF HOLES
    # analysis.number_digging(10)

    # HOLE ANALYSIS METHODS 

    # requires polygon
    # analysis.returns()
    # analysis.time_to_enter() # SCALE FACTOR LESS THAN 1.5?
    analysis.hole_counter()
    # analysis.hole_departures()
    
    # requires hole centroid 
    # analysis.hole_orientation()
    # analysis.distance_from_hole()


    print("Analysis Completed")


if __name__ == "__main__":

    run_gui_option = input("Do you want to run the GUI for hole drawing? (y/n): ").strip().lower()
    if run_gui_option == 'y':
        run_gui()    


    # perform_analysis("/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n1")
    # perform_analysis("/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n2")
    # perform_analysis("/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n3")
    # perform_analysis("/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n4")
    # perform_analysis('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n5')
    # perform_analysis("/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n6")
    # perform_analysis("/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n7")
    # perform_analysis("/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n8")
    # perform_analysis("/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n9")
    # perform_analysis("/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n10")
 
    # perform_analysis("/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-food-behaviour-day4/n1")
    # perform_analysis("/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-food-behaviour-day4/n2")
    # perform_analysis("/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-food-behaviour-day4/n3")
    # perform_analysis("/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-food-behaviour-day4/n4")
    # perform_analysis('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-food-behaviour-day4/n5')
    # perform_analysis("/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-food-behaviour-day4/n6")
    # perform_analysis("/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-food-behaviour-day4/n7")
    # perform_analysis("/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-food-behaviour-day4/n8")
    # perform_analysis("/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-food-behaviour-day4/n9")
    # perform_analysis("/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-food-behaviour-day4/n10")

    
    # perform_analysis('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/plain-petri/1-hour-agarose-n10')
    # perform_analysis('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/plain-petri/1-hour-food-n10')
    # perform_analysis('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/plain-petri/30-minutes-agarose-n10')
    # perform_analysis('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/plain-petri/30-minutes-agarose-n10')

    perform_analysis('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/side-hole/agarose')
    perform_analysis('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/side-hole/food')

    perform_analysis('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/middle-hole/agarose')
    perform_analysis('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/middle-hole/food')


    # perform_analysis('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/test-fake_data')




    





