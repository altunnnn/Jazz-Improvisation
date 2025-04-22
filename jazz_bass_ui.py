import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pygame.midi
import threading
import time
import json
import os
import sys
import numpy as np

# This would link to the RL agent code
# For now, we'll just import the necessary parts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jazz_improvisation import JazzEnvironment, RLJazzAgent, STATE_SHAPE_MELODY, STATE_SHAPE_CHORD, STATE_SHAPE_BASS, STATE_SHAPE_CONTEXT

class JazzBassUI(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # Initialize pygame for MIDI handling
        pygame.midi.init()
        
        # UI Configuration
        self.title("Jazz Bass Improvisation Plugin")
        self.geometry("800x600")
        self.configure(bg="#2a2a2a")
        self.resizable(True, True)
        
        # Main variables
        self.input_device_id = None
        self.output_device_id = None
        self.is_running = False
        self.midi_thread = None
        self.model_loaded = False
        self.midi_in = None
        self.midi_out = None
        self.chord_progression = ['Cmaj7', 'Dm7', 'G7', 'Cmaj7']  # Default progression
        self.current_genre = "Jazz"
        self.saved_settings_path = "jazz_bass_settings.json"
        
        # Agent and environment
        self.agent = None
        self.env = None
        
        # Create the interface
        self.create_widgets()
        
        # Load saved settings
        self.load_settings()
        
        # Protocol for closing the window
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_widgets(self):
        # Main frame with padding
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Style configuration
        style = ttk.Style()
        style.configure("TLabel", foreground="#cccccc", background="#2a2a2a", font=("Arial", 10))
        style.configure("TButton", font=("Arial", 10), padding=5)
        style.configure("TFrame", background="#2a2a2a")
        style.configure("Header.TLabel", font=("Arial", 14, "bold"), padding=10)
        style.configure("Bold.TLabel", font=("Arial", 10, "bold"))
        style.configure("Status.TLabel", foreground="#00ff00", background="#2a2a2a")
        
        # ----- Header -----
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        header_label = ttk.Label(header_frame, text="Jazz Bass Improvisation", style="Header.TLabel")
        header_label.pack()
        
        # ----- MIDI Device Selection -----
        device_frame = ttk.LabelFrame(main_frame, text="MIDI Configuration", padding="10")
        device_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Input device selection
        input_frame = ttk.Frame(device_frame)
        input_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(input_frame, text="Input Device:").pack(side=tk.LEFT, padx=(0, 10))
        self.input_device_var = tk.StringVar()
        self.input_device_menu = ttk.Combobox(input_frame, textvariable=self.input_device_var, width=30)
        self.input_device_menu.pack(side=tk.LEFT)
        ttk.Button(input_frame, text="Refresh", command=self.refresh_midi_devices).pack(side=tk.RIGHT)
        
        # Output device selection
        output_frame = ttk.Frame(device_frame)
        output_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(output_frame, text="Output Device:").pack(side=tk.LEFT, padx=(0, 10))
        self.output_device_var = tk.StringVar()
        self.output_device_menu = ttk.Combobox(output_frame, textvariable=self.output_device_var, width=30)
        self.output_device_menu.pack(side=tk.LEFT)
        
        # ----- Improvisation Settings -----
        settings_frame = ttk.LabelFrame(main_frame, text="Improvisation Settings", padding="10")
        settings_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Genre selection
        genre_frame = ttk.Frame(settings_frame)
        genre_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(genre_frame, text="Genre:").pack(side=tk.LEFT, padx=(0, 10))
        self.genre_var = tk.StringVar(value="Jazz")
        genres = ["Jazz", "Blues", "Funk", "Latin"]
        genre_menu = ttk.Combobox(genre_frame, textvariable=self.genre_var, values=genres, width=15)
        genre_menu.pack(side=tk.LEFT)
        genre_menu.bind("<<ComboboxSelected>>", self.on_genre_change)
        
        # Improvisation mode
        mode_frame = ttk.Frame(settings_frame)
        mode_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(mode_frame, text="Mode:").pack(side=tk.LEFT, padx=(0, 10))
        self.mode_var = tk.StringVar(value="Automatic")
        modes = ["Automatic", "Chord-Based", "Scale-Based", "Advanced"]
        mode_menu = ttk.Combobox(mode_frame, textvariable=self.mode_var, values=modes, width=15)
        mode_menu.pack(side=tk.LEFT)
        
        # ----- Model Settings -----
        model_frame = ttk.LabelFrame(main_frame, text="Model Configuration", padding="10")
        model_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Model selection
        ttk.Button(model_frame, text="Load Model", command=self.load_model).pack(side=tk.LEFT, padx=(0, 10))
        self.model_path_var = tk.StringVar(value="No model loaded")
        model_path_label = ttk.Label(model_frame, textvariable=self.model_path_var)
        model_path_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # ----- Chord Progression -----
        chord_frame = ttk.LabelFrame(main_frame, text="Chord Progression", padding="10")
        chord_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Chord progression entry
        ttk.Label(chord_frame, text="Enter chord progression (comma separated):").pack(anchor=tk.W)
        self.chord_entry = ttk.Entry(chord_frame)
        self.chord_entry.pack(fill=tk.X, pady=5)
        self.chord_entry.insert(0, "Cmaj7, Dm7, G7, Cmaj7")
        
        # Preset chord progressions
        preset_frame = ttk.Frame(chord_frame)
        preset_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(preset_frame, text="Presets:").pack(side=tk.LEFT, padx=(0, 10))
        presets = ["II-V-I", "Blues", "Rhythm Changes", "Minor II-V-I"]
        self.preset_var = tk.StringVar()
        preset_menu = ttk.Combobox(preset_frame, textvariable=self.preset_var, values=presets, width=20)
        preset_menu.pack(side=tk.LEFT)
        preset_menu.bind("<<ComboboxSelected>>", self.on_preset_selected)
        
        # ----- Control Buttons -----
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.start_button = ttk.Button(control_frame, text="Start Improvisation", command=self.toggle_improvisation)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(control_frame, text="Save Settings", command=self.save_settings).pack(side=tk.LEFT, padx=(0, 10))
        
        # ----- Status Display -----
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.pack(fill=tk.X)
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, style="Status.TLabel")
        self.status_label.pack(fill=tk.X)
        
        # ----- Log Display -----
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(log_frame, height=10, bg="#333333", fg="#ffffff", wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Initialize the device lists
        self.refresh_midi_devices()
    
    def refresh_midi_devices(self):
        # Clear current devices
        self.input_device_menu['values'] = []
        self.output_device_menu['values'] = []
        
        # Get available MIDI devices
        input_devices = []
        output_devices = []
        
        for i in range(pygame.midi.get_count()):
            info = pygame.midi.get_device_info(i)
            device_name = info[1].decode('utf-8')
            is_input = info[2]
            is_output = info[3]
            
            if is_input:
                input_devices.append((i, device_name))
            if is_output:
                output_devices.append((i, device_name))
        
        # Populate comboboxes
        self.input_device_menu['values'] = [f"{name} (ID: {id})" for id, name in input_devices]
        self.output_device_menu['values'] = [f"{name} (ID: {id})" for id, name in output_devices]
        
        if input_devices:
            self.input_device_menu.current(0)
        if output_devices:
            self.output_device_menu.current(0)
        
        self.log("MIDI devices refreshed")
    
    def on_genre_change(self, event=None):
        self.current_genre = self.genre_var.get()
        self.log(f"Genre changed to {self.current_genre}")
    
    def on_preset_selected(self, event=None):
        preset = self.preset_var.get()
        if preset == "II-V-I":
            self.chord_entry.delete(0, tk.END)
            self.chord_entry.insert(0, "Dm7, G7, Cmaj7")
        elif preset == "Blues":
            self.chord_entry.delete(0, tk.END)
            self.chord_entry.insert(0, "C7, F7, C7, C7, F7, F7, C7, C7, G7, F7, C7, G7")
        elif preset == "Rhythm Changes":
            self.chord_entry.delete(0, tk.END)
            self.chord_entry.insert(0, "Bbmaj7, G7, Cm7, F7, Bbmaj7, G7, Cm7, F7, Fm7, Bb7, Ebmaj7, Ab7, Dm7, G7, Cm7, F7")
        elif preset == "Minor II-V-I":
            self.chord_entry.delete(0, tk.END)
            self.chord_entry.insert(0, "Dm7b5, G7alt, Cm")
        
        self.log(f"Loaded preset: {preset}")
    
    def toggle_improvisation(self):
        if not self.is_running:
            self.start_improvisation()
        else:
            self.stop_improvisation()
    
    def start_improvisation(self):
        # Parse input device
        input_selection = self.input_device_var.get()
        output_selection = self.output_device_var.get()
        
        try:
            self.input_device_id = int(input_selection.split("ID: ")[1].rstrip(')'))
            self.output_device_id = int(output_selection.split("ID: ")[1].rstrip(')'))
        except (ValueError, IndexError):
            messagebox.showerror("Error", "Please select valid MIDI input and output devices")
            return
        
        # Parse chord progression
        chord_text = self.chord_entry.get()
        self.chord_progression = [chord.strip() for chord in chord_text.split(',')]
        
        if not self.model_loaded:
            messagebox.showwarning("Warning", "No model loaded. Will use basic improvisation.")
        
        # Setup environment and initialize agent if needed
        if not self.env:
            # In a real implementation, we'd use real-time MIDI input as melody
            # For now, we'll just use a simple placeholder melody
            placeholder_melody = [60, 62, 64, 65] * len(self.chord_progression)
            self.env = JazzEnvironment(
                chord_progression=self.chord_progression, 
                melody_sequence=placeholder_melody,
                tempo=120
            )
        
        if not self.agent and self.model_loaded:
            state_shapes = {
                "melody": (1, *STATE_SHAPE_MELODY),
                "chord": (1, *STATE_SHAPE_CHORD),
                "bass": (1, *STATE_SHAPE_BASS),
                "context": (1, *STATE_SHAPE_CONTEXT)
            }
            self.agent = RLJazzAgent(state_shapes=state_shapes, num_actions=28)  # Assuming NUM_BASS_NOTES
            self.agent.load(self.model_path_var.get())
            self.agent.epsilon = 0.0  # Disable exploration for performance
        
        try:
            # Open MIDI devices
            self.midi_in = pygame.midi.Input(self.input_device_id)
            self.midi_out = pygame.midi.Output(self.output_device_id)
            
            # Start the processing thread
            self.is_running = True
            self.midi_thread = threading.Thread(target=self.process_midi)
            self.midi_thread.daemon = True
            self.midi_thread.start()
            
            # Update UI
            self.start_button.config(text="Stop Improvisation")
            self.status_var.set("Running")
            self.log("Improvisation started")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start improvisation: {str(e)}")
            self.log(f"Error: {str(e)}")
    
    def stop_improvisation(self):
        self.is_running = False
        
        # Clean up MIDI devices
        if self.midi_in:
            self.midi_in.close()
            self.midi_in = None
        
        if self.midi_out:
            self.midi_out.close()
            self.midi_out = None
        
        # Wait for thread to end
        if self.midi_thread:
            self.midi_thread.join(timeout=1.0)
            self.midi_thread = None
        
        # Update UI
        self.start_button.config(text="Start Improvisation")
        self.status_var.set("Stopped")
        self.log("Improvisation stopped")
    
    def load_model(self):
        model_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("HDF5 Files", "*.h5"), ("All Files", "*.*")]
        )
        
        if model_path:
            self.model_path_var.set(model_path)
            self.model_loaded = True
            self.log(f"Model loaded: {os.path.basename(model_path)}")
            
            # In a real implementation, we would load the weights here
            # self.agent.load(model_path)
    
    def process_midi(self):
        """Process MIDI input and generate bass output"""
        self.log("MIDI processing started")
        
        # Simple detection of quarter notes for synchronization
        last_beat_time = time.time()
        beat_interval = 60.0 / 120.0  # 120 BPM default
        state = self.env.reset() if self.env else None
        
        # Track notes to avoid hanging notes
        active_notes = set()
        current_chord_idx = 0
        chord_duration_beats = 4  # Assume 4 beats per chord
        beat_count = 0
        
        # Tracked melody notes for context
        melody_notes = []
        respond_to_melody = True  # New flag to trigger response to played notes
        
        try:
            while self.is_running:
                # Process incoming MIDI if available
                if self.midi_in and self.midi_in.poll():
                    midi_events = self.midi_in.read(10)
                    for event in midi_events:
                        # event[0] is [status, note, velocity, 0]
                        status = event[0][0]
                        note = event[0][1]
                        velocity = event[0][2]
                        
                        # Note on events (status & 0xF0 == 0x90)
                        if (status & 0xF0) == 0x90 and velocity > 0:
                            # Store melody notes for context
                            melody_notes.append(note)
                            if len(melody_notes) > 16:  # Keep last 16 notes
                                melody_notes.pop(0)
                            respond_to_melody = True  # Trigger a response
                    
                # Generate bass note on each beat or when new melody note is played
                now = time.time()
                if now - last_beat_time >= beat_interval or respond_to_melody:
                    if now - last_beat_time >= beat_interval:
                        last_beat_time = now
                        beat_count += 1
                        
                        # Update chord if needed
                        if beat_count % chord_duration_beats == 0:
                            current_chord_idx = (current_chord_idx + 1) % len(self.chord_progression)
                            current_chord = self.chord_progression[current_chord_idx]
                            self.log(f"Current chord: {current_chord}")
                    
                    respond_to_melody = False  # Reset the flag
                    
                    # Generate bass note with the agent if available
                    bass_note = None
                    if self.agent and state:
                        # Update state with new melody notes if available
                        if melody_notes:
                            self.env.update_melody(melody_notes[-8:])  # Use last 8 notes
                        
                        action = self.agent.choose_action(state)
                        state, reward, done, _ = self.env.step(action)
                        bass_note = action + 32  # Convert back to MIDI note (BASS_NOTE_MIN)
                    else:
                        # Simple fallback if no model
                        try:
                            from music21 import chord as m21chord
                            c = m21chord.Chord(self.chord_progression[current_chord_idx])
                            root = c.root().midi
                            
                            # If we have melody notes, try to create a more responsive bass
                            if melody_notes:
                                # Simple approach: use root or fifth based on last melody note
                                if melody_notes[-1] % 12 in [0, 4, 7]:  # If last note is part of the chord
                                    bass_note = root - 24  # root two octaves down
                                else:
                                    bass_note = root - 24 + 7  # fifth
                            else:
                                bass_note = root - 24  # Two octaves down
                        except Exception:
                            bass_note = 36  # Default to C2 if parsing fails
                    
                    # Send MIDI note if we have an output and a valid note
                    if self.midi_out and bass_note:
                        # Stop previous note to avoid overlaps
                        for note_to_stop in active_notes:
                            self.midi_out.note_off(note_to_stop, 0)
                        active_notes.clear()
                        
                        # Play new note
                        self.midi_out.note_on(bass_note, 100)  # velocity 100
                        active_notes.add(bass_note)
                        
                        # Log what's happening
                        from music21 import note as m21note
                        try:
                            note_name = m21note.Note(bass_note).nameWithOctave
                            self.log(f"Playing bass note: {note_name} (MIDI: {bass_note})")
                        except Exception:
                            self.log(f"Playing bass note: MIDI {bass_note}")
                
                # Small sleep to prevent CPU hogging
                time.sleep(0.01)
                
        except Exception as e:
            self.log(f"Error in MIDI processing: {str(e)}")
            
        finally:
            # Stop all notes in case of shutdown
            if self.midi_out:
                for note_to_stop in active_notes:
                    try:
                        self.midi_out.note_off(note_to_stop, 0)
                    except Exception:
                        pass
    
    def log(self, message):
        """Add message to log with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)  # Scroll to end
    
    def save_settings(self):
        """Save current settings to file"""
        settings = {
            "input_device": self.input_device_var.get(),
            "output_device": self.output_device_var.get(),
            "genre": self.genre_var.get(),
            "mode": self.mode_var.get(),
            "chord_progression": self.chord_entry.get(),
            "model_path": self.model_path_var.get()
        }
        
        try:
            with open(self.saved_settings_path, 'w') as f:
                json.dump(settings, f)
            self.log("Settings saved")
        except Exception as e:
            self.log(f"Error saving settings: {str(e)}")
    
    def load_settings(self):
        """Load settings from file if exists"""
        try:
            if os.path.exists(self.saved_settings_path):
                with open(self.saved_settings_path, 'r') as f:
                    settings = json.load(f)
                
                # Apply settings
                if "input_device" in settings and settings["input_device"] in self.input_device_menu["values"]:
                    self.input_device_var.set(settings["input_device"])
                
                if "output_device" in settings and settings["output_device"] in self.output_device_menu["values"]:
                    self.output_device_var.set(settings["output_device"])
                
                if "genre" in settings:
                    self.genre_var.set(settings["genre"])
                
                if "mode" in settings:
                    self.mode_var.set(settings["mode"])
                
                if "chord_progression" in settings:
                    self.chord_entry.delete(0, tk.END)
                    self.chord_entry.insert(0, settings["chord_progression"])
                
                if "model_path" in settings and os.path.exists(settings["model_path"]):
                    self.model_path_var.set(settings["model_path"])
                    self.model_loaded = True
                
                self.log("Settings loaded")
        except Exception as e:
            self.log(f"Error loading settings: {str(e)}")
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_running:
            self.stop_improvisation()
        
        # Clean up pygame MIDI
        pygame.midi.quit()
        
        self.destroy()


if __name__ == "__main__":
    app = JazzBassUI()
    app.mainloop()