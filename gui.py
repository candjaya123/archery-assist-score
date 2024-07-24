import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class FileWatcher(FileSystemEventHandler):
    def __init__(self, app):
        self.app = app
    
    def on_modified(self, event):
        if event.src_path.endswith("file.txt"):
            self.app.update_table_data()

class ArcheryScoringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Automatic Archery Scoring")
        self.root.geometry("1300x750")  # Adjusted for better layout with camera frame size
        self.root.configure(bg="#a1c4fd")  # Set background color for the root window
        
        # Camera Live View
        self.camera_frame = tk.LabelFrame(root, text="Camera Live View", width=640, height=640, bg="#8bc34a", fg="#ffffff", font=("Helvetica", 12, "bold"))
        self.camera_frame.grid(row=0, column=0, padx=10, pady=10)
        self.camera_label = tk.Label(self.camera_frame, width=640, height=640, bg="#ffffff")
        self.camera_label.pack()

        # Score Table
        self.table_frame = tk.LabelFrame(root, text="Score", width=640, height=640, bg="#4caf50", fg="#ffffff", font=("Helvetica", 12, "bold"))
        self.table_frame.grid(row=0, column=1, padx=10, pady=10)
        
        columns = ["Shot", "First Session", "Second Session", "Third Session", "Fourth Session", "Fifth Session", "Sixth Session"]
        self.treeview = ttk.Treeview(self.table_frame, columns=columns, show="headings", height=10)
        
        for col in columns:
            self.treeview.heading(col, text=col)
            self.treeview.column(col, width=80, anchor='center')
        
        style = ttk.Style()
        style.configure("Treeview.Heading", background="#2e7d32", foreground="black", font=("Helvetica", 10, "bold"))
        style.configure("Treeview", background="#a5d6a7", foreground="black", fieldbackground="#a5d6a7", font=("Helvetica", 10))
        
        self.treeview.pack(fill="both", expand=True)

        # Total Score
        self.total_frame = tk.Frame(self.table_frame, bg="#4caf50")
        self.total_frame.pack(fill="x", pady=5)
        
        self.total_label = tk.Label(self.total_frame, text="TOTAL ALL SCORE", bg="#ffeb3b", font=("Helvetica", 12, "bold"))
        self.total_label.pack(fill="x")

        # Timer
        self.timer_frame = tk.Frame(root, bg="#a1c4fd")
        self.timer_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        self.timer_label = tk.Label(self.timer_frame, text="0", bg="#a1c4fd", font=("Helvetica", 16, "bold"))
        self.timer_label.pack()

        self.start_time = time.time()
        self.update_timer()

        # Start webcam thread
        self.cap = cv2.VideoCapture("input_cepat.mp4")
        self.show_frame()

        # Load table data
        self.update_table_data()

        # Start file watcher
        self.observer = None
        self.start_file_watcher()

        # Start timer to write to file
        self.stop_event = threading.Event()
        self.start_timer()

    def show_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 640))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)
        self.camera_label.after(10, self.show_frame)

    def update_table_data(self):
        # Clear existing data
        for item in self.treeview.get_children():
            self.treeview.delete(item)
        
        with open("file.txt", 'r') as file:
            lines = file.readlines()
            for line in lines:
                self.treeview.insert('', 'end', values=line.strip().split(','))

    def start_file_watcher(self):
        event_handler = FileWatcher(self)
        self.observer = Observer()
        self.observer.schedule(event_handler, path='.', recursive=False)
        self.observer.start()
        # Ensure the observer is stopped properly when the app is closed
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        if self.observer is not None:
            self.observer.stop()
            self.observer.join()
        self.stop_event.set()  # Signal the thread to stop
        self.cap.release()
        self.root.destroy()

    def start_timer(self):
        def write_to_file():
            # Define the schedule as a list of (delay, content) tuples
            schedule = [
                (6, "1,10,0,0,0,0,0"),
                (16, "2,5,0,0,0,0,0"),
                (13, "3,6,0,0,0,0,0"),
                (10, "4,9,0,0,0,0,0"),
                (11, "5,9,0,0,0,0,0"),
                (11, "6,8,0,0,0,0,0"),
                # Add more schedules as needed
            ]

            for delay, content in schedule:
                if self.stop_event.wait(delay):
                    break
                with open("file.txt", 'a') as file:
                    file.write(content + "\n")

        threading.Thread(target=write_to_file, daemon=True).start()

    def update_timer(self):
        elapsed_time = int(time.time() - self.start_time)
        self.timer_label.configure(text=str(elapsed_time))
        self.root.after(1000, self.update_timer)

if __name__ == "__main__":
    root = tk.Tk()
    app = ArcheryScoringApp(root)
    root.mainloop()
