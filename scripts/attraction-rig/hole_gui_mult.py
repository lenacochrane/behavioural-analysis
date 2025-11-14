import tkinter as tk # Python's standard GUI library
from tkinter import filedialog, messagebox
import cv2 
import pandas as pd
import os 
import csv

class HoleGui:
    def __init__(self, root):
        self.root = root
        self.root.title("Hole Analyser")
        self.directory = ""
        self.video_files = []
        self.hole_coordinates = []
        self.holes = []  # NEW: store finalized holes
        self.drawing = False
        self.current_video_index = 0

        self.label = tk.Label(root, text="Select Video Directory")
        self.label.pack(pady=10)

        self.select_button = tk.Button(root, text="Select Directory", command=self.select_directory)
        self.select_button.pack(pady=10)

        self.process_button = tk.Button(root, text="Process Videos", command=self.process_videos)
        self.process_button.pack(pady=10)

    def select_directory(self):
        self.directory = filedialog.askdirectory()
        self.video_files = [f for f in os.listdir(self.directory) if f.endswith('.mp4')]
        self.video_files.sort()
        print(f"Selected directory: {self.directory}")
        print(f"Video files found: {self.video_files}")

    def process_videos(self):
        if self.current_video_index < len(self.video_files):
            video_file = self.video_files[self.current_video_index]
            video_path = os.path.join(self.directory, video_file)
            print(f"Processing video: {video_path}")
            self.process_video(video_path)
        else:
            print("All videos processed.")
            messagebox.showinfo("Processing Complete", "All videos have been processed.")
            self.root.destroy()  # Force close the Tkinter window

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Cannot open video: {video_path}")
            messagebox.showerror("Error", f"Cannot open video: {video_path}")
            self.current_video_index += 1
            self.process_videos()
            return

        ret, frame = cap.read()
        if not ret:
            print(f"Error: Cannot read video: {video_path}")
            messagebox.showerror("Error", f"Cannot read video: {video_path}")
            self.current_video_index += 1
            self.process_videos()
            return

        self.temp_frame = frame.copy()
        self.holes = []                 # NEW: reset holes per video
        self.hole_coordinates = []      # Reset coordinates for each video

        cv2.imshow("Draw Hole Perimeter", self.temp_frame)
        cv2.setMouseCallback("Draw Hole Perimeter", self.draw_hole_perimeter, self.temp_frame)
        cv2.waitKey(0)

    def draw_hole_perimeter(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # ✅ Auto-finalize the previously closed hole before starting a new one
            if not self.drawing and len(self.hole_coordinates) >= 3:
                self.holes.append(self.hole_coordinates[:])
                print(f"Auto-finalized hole #{len(self.holes)} on new draw")
                self.hole_coordinates = []

            self.drawing = True
            self.hole_coordinates.append((x, y))
            print(f"Point added: {(x, y)}")

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.hole_coordinates:
                prev_point = self.hole_coordinates[-1]
                self.hole_coordinates.append((x, y))
                cv2.line(self.temp_frame, prev_point, (x, y), (0, 0, 255), 2)
                cv2.imshow("Draw Hole Perimeter", self.temp_frame)
                print(f"Drawing line from {prev_point} to {(x, y)}")

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if len(self.hole_coordinates) > 1:
                prev_point = self.hole_coordinates[-1]
                cv2.line(self.temp_frame, prev_point, self.hole_coordinates[0], (0, 0, 255), 2)
                cv2.imshow("Draw Hole Perimeter", self.temp_frame)
                print(f"Closed the hole perimeter by drawing line from {prev_point} to {self.hole_coordinates[0]}")
                # Not finalized yet; next LBUTTONDOWN (or middle-click) will finalize

        elif event == cv2.EVENT_MBUTTONDOWN:
            # Finalize current hole and start a new one (optional shortcut)
            if len(self.hole_coordinates) >= 3:
                cv2.line(self.temp_frame, self.hole_coordinates[-1], self.hole_coordinates[0], (0, 0, 255), 2)
                cv2.imshow("Draw Hole Perimeter", self.temp_frame)
                self.holes.append(self.hole_coordinates[:])
                print(f"Finalized hole #{len(self.holes)} with {len(self.hole_coordinates)} points")
                self.hole_coordinates = []
            else:
                print("Middle-click ignored: need at least 3 points to finalize a hole.")

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Auto-finalize in-progress hole if valid, then save & next
            if len(self.hole_coordinates) >= 3:
                cv2.line(self.temp_frame, self.hole_coordinates[-1], self.hole_coordinates[0], (0, 0, 255), 2)
                cv2.imshow("Draw Hole Perimeter", self.temp_frame)
                self.holes.append(self.hole_coordinates[:])
                print(f"Auto-finalized hole #{len(self.holes)} with {len(self.hole_coordinates)} points before saving")
                self.hole_coordinates = []

            print("Right button clicked, saving coordinates and moving to next video.")
            self.save_coordinates()
            cv2.destroyAllWindows()
            self.current_video_index += 1
            self.process_videos()


    def save_coordinates(self):
        # If user drew only one hole and never middle-clicked, handle it
        if not self.holes and len(self.hole_coordinates) >= 3:
            self.holes.append(self.hole_coordinates[:])

        if self.holes:
            video_file = self.video_files[self.current_video_index]
            video_path = os.path.join(self.directory, video_file)
            base = os.path.splitext(os.path.basename(video_path))[0]

            for i, hole in enumerate(self.holes, start=1):
                hole_file = os.path.join(self.directory, f"{base}_hole{i}.csv")  # NEW: per-hole files
                print(f"Saving hole #{i} to {hole_file} with {len(hole)} points")
                try:
                    with open(hole_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows(hole)
                    print(f"Coordinates saved to {hole_file}")
                except Exception as e:
                    print(f"Error saving coordinates to {hole_file}: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    HoleGui(root)
    root.mainloop()
