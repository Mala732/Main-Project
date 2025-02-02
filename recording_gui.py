import tkinter as tk
from itertools import cycle
import random
import csv
from datetime import datetime
import winsound  # Module to play sound on Windows

# Create the Tkinter application
root = tk.Tk()
root.title("Cyclic Display with Buzzer")
csv_file_name = "left_hand_trial1.csv"
# Set the window to full screen
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{screen_width}x{screen_height}")

# Create a label to display the content
label = tk.Label(root, text="", font=("Arial", 72), bg="white")
label.pack(expand=True, fill="both")

# Variable to store the selected cue
selected_cue = tk.StringVar(value="←")  # Default cue

# Default durations (in milliseconds)
durations = {
    "fixation": 3000,  # Fixation cross
    "cue": 1000,       # Cue
    "record": 4000,    # Record
    "rest": 2000       # Rest
}

# Sequence of states
sequence = []
sequence_cycle = None

# Flags and variables
is_started = False
is_paused = False
current_content = None
start_time = None
log_data = []  # To store log entries
current_task = None  # Reference to the active Tkinter `after` task

# Mapping for descriptive labels
symbol_to_description = {
    "+": "Fixation cross",
    "←": "left",
    "→": "right",
    ".": "cue",
    "": "rest"
}

def play_short_buzzer():
    """Play a short buzzer sound."""
    winsound.Beep(1000, 500)  # 1000 Hz for 500 ms

# def play_long_buzzer():
#     """Play a long buzzer sound."""
#     winsound.Beep(1000, 1000)  # 1000 Hz for 600 ms

def log_time(content):
    """Log the display timings."""
    global current_content, start_time, log_data

    if current_content is not None:
        stop_time = datetime.now()
        log_data.append({
            "Sign": symbol_to_description[current_content],
            "Start Time": start_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
            "Stop Time": stop_time.strftime("%Y-%m-%d %H:%M:%S.%f")
        })

    # Update current content and start time for the new sign
    current_content = content
    start_time = datetime.now()

def update_display():
    """Update the display with the next item in the sequence."""
    global current_task, is_started, is_paused, current_content

    if not is_started or is_paused:
        return  # Exit if the loop is paused or stopped

    content, duration = next(sequence_cycle)  # Get the next state and duration

    # Handle pre-cue beep while displaying the fixation cross
    if content == "fixation_beep":
        #play_short_buzzer()
        content = "+"  # Keep displaying the fixation cross
    elif content == "CUE":
        content = "."
    elif content == "record":
        play_short_buzzer()
        current_content = random.choice(["←", "→"]) if selected_cue.get() == "random" else selected_cue.get()
        content = current_content
        #content = current_content  # Keep displaying the cue during the record state
    elif content == "rest":
        content = ""  # Rest displays a blank screen

    # Log the current display content
    log_time(content)

    # Update the label's text and schedule the next update
    label.config(text=content)
    current_task = root.after(duration, update_display)

def start_loop():
    """Starts the loop when the Start button is pressed."""
    global is_started, is_paused, log_data, sequence, sequence_cycle, current_task

    if not is_started or is_paused:
        is_started = True
        is_paused = False

        # Build the sequence with updated durations
        sequence = [
            ("+", durations["fixation"] - 1000),  # Fixation without beep
            ("fixation_beep", 1000),             # Beep during the last second of fixation
            ("CUE", durations["cue"]),
            ("record", durations["record"]),     # Record while keeping the cue visible
            ("rest", durations["rest"])
        ]
        sequence_cycle = cycle(sequence)

        # Start the loop
        if current_task is None:  # Prevent duplicate `after` tasks
            update_display()




def pause_loop():
    """Pauses the loop when the Pause button is pressed."""
    global is_paused, current_task

    if is_started and not is_paused:
        is_paused = True
        if current_task is not None:
            root.after_cancel(current_task)  # Cancel the current scheduled task
            current_task = None

def resume_loop():
    """Resumes the loop when the Resume button is pressed."""
    global is_paused

    if is_started and is_paused:
        is_paused = False
        update_display()

def stop_and_restart_loop():
    """Stops the loop, saves data, and resets the state."""
    global is_started, is_paused, current_task, log_data

    if is_started:
        is_started = False
        is_paused = False

        # Cancel any pending task
        if current_task is not None:
            root.after_cancel(current_task)
            current_task = None

        # Save log data to CSV
        with open(csv_file_name, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["Sign", "Start Time", "Stop Time"])
            writer.writeheader()
            writer.writerows(log_data)

        print("Log saved to sign_timings.csv")
        log_data = []  # Clear the log data

        # Clear the display
        label.config(text="")

def update_durations():
    """Updates the durations based on user input."""
    try:
        durations["fixation"] = int(fixation_entry.get())
        durations["cue"] = int(cue_entry.get())
        durations["record"] = int(record_entry.get())
        durations["rest"] = int(rest_entry.get())
        print("Durations updated:", durations)
    except ValueError:
        print("Invalid duration values entered. Please enter integers.")

# Dropdown menu for selecting a cue
def create_cue_selector():
    """Create a dropdown menu to select the cue."""
    frame = tk.Frame(root, bg="white")
    frame.pack(side="top", fill="x", padx=10, pady=10)

    tk.Label(frame, text="Select Cue:", font=("Arial", 20), bg="white").pack(side="left", padx=5)

    cues = ["←","→","random"]
    dropdown = tk.OptionMenu(frame, selected_cue, *cues)
    dropdown.config(font=("Arial", 16))
    dropdown.pack(side="left", padx=5)

# Duration adjustment inputs
def create_duration_inputs():
    """Create inputs to adjust the durations."""
    frame = tk.Frame(root, bg="white")
    frame.pack(side="top", fill="x", padx=10, pady=10)

    tk.Label(frame, text="Fixation (ms):", font=("Arial", 16), bg="white").grid(row=0, column=0, padx=5, pady=5)
    tk.Label(frame, text="Cue (ms):", font=("Arial", 16), bg="white").grid(row=0, column=2, padx=5, pady=5)
    tk.Label(frame, text="Record (ms):", font=("Arial", 16), bg="white").grid(row=1, column=0, padx=5, pady=5)
    tk.Label(frame, text="Rest (ms):", font=("Arial", 16), bg="white").grid(row=1, column=2, padx=5, pady=5)

    global fixation_entry, cue_entry, record_entry, rest_entry
    fixation_entry = tk.Entry(frame, font=("Arial", 14), width=10)
    fixation_entry.insert(0, str(durations["fixation"]))
    fixation_entry.grid(row=0, column=1, padx=5, pady=5)

    cue_entry = tk.Entry(frame, font=("Arial", 14), width=10)
    cue_entry.insert(0, str(durations["cue"]))
    cue_entry.grid(row=0, column=3, padx=5, pady=5)

    record_entry = tk.Entry(frame, font=("Arial", 14), width=10)
    record_entry.insert(0, str(durations["record"]))
    record_entry.grid(row=1, column=1, padx=5, pady=5)

    rest_entry = tk.Entry(frame, font=("Arial", 14), width=10)
    rest_entry.insert(0, str(durations["rest"]))
    rest_entry.grid(row=1, column=3, padx=5, pady=5)

    update_button = tk.Button(frame, text="Update Durations", font=("Arial", 12), command=update_durations, bg="#2196F3", fg="white")
    update_button.grid(row=2, columnspan=4, pady=10)

# Control buttons (Start, Pause, Resume, Stop and Restart)
def create_control_buttons():
    """Create control buttons for the loop."""
    button_frame = tk.Frame(root, bg="white")
    button_frame.pack(side="top", pady=10)

    start_button = tk.Button(button_frame, text="Start", font=("Arial", 16), command=start_loop, bg="#4CAF50", fg="white")
    start_button.pack(side="left", padx=10)

    pause_button = tk.Button(button_frame, text="Pause", font=("Arial", 16), command=pause_loop, bg="#FFC107", fg="black")
    pause_button.pack(side="left", padx=10)

    resume_button = tk.Button(button_frame, text="Resume", font=("Arial", 16), command=resume_loop, bg="#FF9800", fg="white")
    resume_button.pack(side="left", padx=10)

    stop_button = tk.Button(button_frame, text="Stop", font=("Arial", 16), command=stop_and_restart_loop, bg="#F44336", fg="white")
    stop_button.pack(side="left", padx=10)

# Build the UI
create_cue_selector()
create_duration_inputs()
create_control_buttons()

# Run the application
root.mainloop()
