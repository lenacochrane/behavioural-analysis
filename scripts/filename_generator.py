import os
import csv

def list_files_in_directory(directory_path, output_csv):
    # Get a list of all files in the directory
    files = os.listdir(directory_path)

    # Filter out directories, keeping only files
    files = [f for f in files if os.path.isfile(os.path.join(directory_path, f))]
        
        # Sort the files by name
    files.sort()

    # Create a CSV file and write the filenames to it
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['File Name'])  # Write the header
        for filename in files:
            writer.writerow([filename])  # Write each filename

    print(f"CSV file '{output_csv}' created successfully.")



# Example usage:

directory_path = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/Videos/Holes/1H/PLAIN-PETRI-N10/AGAROSE'
output_csv = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/Videos/Holes/1H/PLAIN-PETRI-N10/agarose.csv'
list_files_in_directory(directory_path, output_csv)
