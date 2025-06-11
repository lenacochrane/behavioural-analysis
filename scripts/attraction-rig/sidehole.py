
import tkinter as tk
from hole_gui import HoleGui  # Import the HoleGui class from the hole_gui module
from hole_analysis import HoleAnalysis 
import os 
import seaborn as sns
import matplotlib.pyplot as plt


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

    analysis.digging_mask()


    # analysis.post_processing()
    # analysis.quality_control()

    # analysis.nearest_neighbour()

    # analysis.hole_boundary(scale_factor=1.5)

    #METHOD TO SHORTEN
    # analysis.shorten(frame=600)
    
    # CALL METHODS FOR LARVAEL BEHAVIOUR 
    # analysis.trajectory()
    # analysis.time_average_msd(list(range(1, 101, 1)))
    # analysis.speed()
    # analysis.ensemble_msd()
    # analysis.acceleration()
    # analysis.euclidean_distance()
    # analysis.euclidean_distance_variance(200, 600) # currently plotting the plataeu but arbitary time start for plateu 
    # analysis.distance_from_centre()


    ### PSEUDO POPULATION MODEL

    # analysis.pseudo_population_model()
    # analysis.pseudo_population_model(number_of_iterations=1, number_of_animals=2)


    # DIGGING IN ABSENCE OF HOLES
    # analysis.total_digging(cleaned=True) #HAVE TO MODIFY



    


    # analysis.hole_euclidean_distance()

    # analysis.interaction_types()


    analysis.interactions()
    
    # analysis.contacts(proximity_threshold=5)

    # analysis.total_digging(cleaned=True)
    # analysis.digging_behaviour()


    ##### --- HOLES --- ####
    # analysis.compute_hole()
    # analysis.hole_counter()
    # analysis.returns()
    # analysis.hole_departures()
    # analysis.time_to_enter()
    # analysis.hole_entry_probability()

    # analysis.hole_mask()
    # analysis.hole_orientation()
    # analysis.distance_from_hole()
    


    print("Analysis Completed")


if __name__ == "__main__":

    run_gui_option = input("Do you want to run the GUI for hole drawing? (y/n): ").strip().lower()
    if run_gui_option == 'y':
        run_gui()    



    # perform_analysis("/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n1/group-housed")
    # perform_analysis("/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n1/socially-isolated")
    perform_analysis("/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/group-housed")
    perform_analysis("/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/socially-isolated")
    perform_analysis("/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed")
    perform_analysis("/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/socially-isolated")

    # perform_analysis('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/holes/N10-1-HOLE/GROUP-HOUSED')
    # perform_analysis('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/holes/N10-1-HOLE/SOCIAL-ISOLATION')


    # perform_analysis('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-digging-mask/diff-video')




    







    





