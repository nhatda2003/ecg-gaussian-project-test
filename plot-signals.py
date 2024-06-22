import os
import sys
import re
import numpy as np
import matplotlib.pyplot as plt

def plot_npy_files(data_folder):
    # Define a regex pattern to match filenames like "_1_raw100.npy", "_2_raw100.npy", etc.
    pattern_raw = re.compile(r'^_\d+_raw100\.npy$')
    pattern_label = re.compile(r'^_\d+_label\.npy$')

    # Get a list of all .npy files in the directory that match the pattern
    raw_files = [file for file in os.listdir(data_folder) if pattern_raw.match(file)]
    label_files = [file for file in os.listdir(data_folder) if pattern_label.match(file)]

    # Sort the file lists to ensure files are processed in order
    raw_files.sort()
    label_files.sort()

    # Check if the number of raw files and label files match
    if len(raw_files) != len(label_files):
        print("Error: Number of raw data files does not match number of label files.")
        return

    # Loop through each matching .npy file
    for idx, (raw_file, label_file) in enumerate(zip(raw_files, label_files)):
        # Construct the full file paths
        raw_file_path = os.path.join(data_folder, raw_file)
        label_file_path = os.path.join(data_folder, label_file)
        
        # Load the data from the .npy files, allowing pickling
        raw_data = np.load(raw_file_path, allow_pickle=True)
        label_data = np.load(label_file_path, allow_pickle=True)
        
        # Determine if the signal is normal or abnormal based on label
        if label_data == 0:
            label_text = "Normal"
        elif label_data == 1:
            label_text = "Abnormal"
        else:
            label_text = "Unknown"  # Handle unexpected labels
        
        # Plot the data
        plt.figure(figsize=(24, 6))  # Set figure size (width, height)
        plt.plot(raw_data)
        plt.title(f'Plot of {raw_file} ({label_text})')  # Use the file name and label as the plot title
        plt.xlabel('Sample Index')
        plt.ylabel('Signal Value')
        plt.grid(True)
        
        # Save the plot as a PNG file in the same folder
        plot_filename = os.path.splitext(raw_file)[0] + f'_{label_text}.png'  # Append label to file name
        plot_path = os.path.join(data_folder, plot_filename)
        plt.savefig(plot_path)
        
        # Close the plot to free up memory
        plt.close()

    print(f"Plots saved successfully for files matching the pattern in the '{data_folder}' folder.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_npy_files.py <data_folder>")
        sys.exit(1)

    data_folder = sys.argv[1]
    plot_npy_files(data_folder)

