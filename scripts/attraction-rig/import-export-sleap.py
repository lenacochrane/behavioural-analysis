import sleap

labels = sleap.Labels.load_file("/Users/cochral/Desktop/SLAEP/Topdown/test.slp")

# Export the tracked data to CSV
# This will create a CSV file for each video in the SLEAP project
labels.export_csv("/Users/cochral/Desktop/SLAEP/Topdown/")