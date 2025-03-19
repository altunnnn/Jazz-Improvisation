import pygame
import pygame.midi
import tkinter as tk
from tkinter import ttk
import numpy as np
import threading
import time
import queue
import mido
from mido import MidiFile, Message, MetaMessage
from music21 import chord, note
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import rtmidi
import os
import sys
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JazzImproModelInterface:
    def __init__(self, model_path='improved_jazz_model.h5'):
        # Initialize basic components
        self.init_pygame()
        
        # Load the model
        try:
            self.model = keras.models.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Creating new model instance...")
            from jazz_improvisation_model import JazzImprovisationModel
            jazz_model = JazzImprovisationModel()
            self.model = jazz_model.model
        
        # Constants
        self.sequence_length = 32  # Number of notes to consider for prediction
        self.tempo = 120  # BPM
        self.is_playing = False
        self.using_virtual_input = False
        
        # Communication queues
        self.note_queue = queue.Queue()
        self.chord_queue = queue.Queue()
        
        # Buffers for storing input data
        self.notes_buffer = []
        self.chords_buffer = []
        self.current_chord = "Cmaj7"  # Default chord
        
        # Virtual keyboard state
        self.virtual_keyboard_notes = []
        
        # Find available MIDI devices
        self.input_device_id = None
        self.output_device_id = None
        self.input_device = None
        self.output_device = None
        
        # Virtual MIDI port setup
        self.virtual_input_port = None
        self.virtual_output_port = None
        
        # For RtMidi virtual ports
        self.rtmidi_in = None
        self.rtmidi_out = None
        
        # Create visualization data
        self.visualization_data = {
            'input_notes': [],
            'output_notes': [],
            'timestamps': [],
            'chord_changes': []
        }
        
        # Create threads
        self.processing_thread = None
        self.virtual_input_thread = None
        self.stop_event = threading.Event()
        
        # Setup UI
        self.setup_user_interface()
        
        # Find MIDI devices
        self.find_midi_devices()
        
    def init_pygame(self):
        """Initialize pygame for MIDI and audio"""
        try:
            pygame.init()
            pygame.midi.init()
            logger.info("Pygame initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing pygame: {e}")
            
    def find_midi_devices(self):
        """Find available MIDI input and output devices"""
        logger.info("\nSearching for MIDI devices...")
        
        # Get physical MIDI devices from pygame
        physical_inputs = []
        physical_outputs = []
        
        try:
            for i in range(pygame.midi.get_count()):
                info = pygame.midi.get_device_info(i)
                device_name = info[1].decode('utf-8')
                is_input = info[2]
                is_output = info[3]
                is_opened = info[4]
                
                status = "open" if is_opened else "available"
                direction = ""
                if is_input:
                    direction += "input "
                    physical_inputs.append((i, device_name))
                if is_output:
                    direction += "output"
                    physical_outputs.append((i, device_name))
                    
                logger.info(f"ID: {i}, Name: {device_name}, {direction}, {status}")
            
            # Update device lists in UI
            self.update_device_lists(physical_inputs, physical_outputs)
            
        except Exception as e:
            logger.error(f"Error finding MIDI devices: {e}")
    
    def setup_virtual_midi(self):
        """Set up virtual MIDI ports using rtmidi"""
        try:
            # Create virtual MIDI ports
            self.rtmidi_in = rtmidi.MidiIn()
            self.rtmidi_out = rtmidi.MidiOut()
            
            # Create virtual ports
            virtual_in_name = "JazzAI_VirtualInput"
            virtual_out_name = "JazzAI_VirtualOutput"
            
            try:
                # Check if port already exists
                for port_name in self.rtmidi_in.get_ports():
                    if virtual_in_name in port_name:
                        logger.info(f"Virtual input port {virtual_in_name} already exists")
                        break
                else:
                    self.rtmidi_in.open_virtual_port(virtual_in_name)
                    logger.info(f"Created virtual input port: {virtual_in_name}")
                
                for port_name in self.rtmidi_out.get_ports():
                    if virtual_out_name in port_name:
                        logger.info(f"Virtual output port {virtual_out_name} already exists")
                        break
                else:
                    self.rtmidi_out.open_virtual_port(virtual_out_name)
                    logger.info(f"Created virtual output port: {virtual_out_name}")
                
                # Set callback for input port
                self.rtmidi_in.set_callback(self.virtual_midi_callback)
                
                # Update device lists
                self.find_midi_devices()
                
                self.status_var.set(f"Virtual MIDI ports created. Connect to {virtual_in_name} and {virtual_out_name}")
                
            except rtmidi.InvalidPortError as e:
                logger.error(f"Error creating virtual ports: {e}")
                self.status_var.set("Error creating virtual MIDI ports")
                
        except Exception as e:
            logger.error(f"Error setting up virtual MIDI: {e}")
            self.status_var.set("Error: RtMidi not available")
    
    def virtual_midi_callback(self, message, time_stamp=None):
        """Callback for virtual MIDI input"""
        if message[0][0] & 0xF0 == 0x90:  # Note On
            note = message[0][1]
            velocity = message[0][2]
            
            if velocity > 0:
                self.process_note_input(note)
    
    def setup_user_interface(self):
        """Setup the UI with tkinter"""
        self.root = tk.Tk()
        self.root.title("Jazz Improvisation AI")
        self.root.geometry("1000x700")
        
        # Create main frames
        self.control_frame = ttk.Frame(self.root, padding=10)
        self.control_frame.pack(fill='x')
        
        self.midi_frame = ttk.Frame(self.root, padding=10)
        self.midi_frame.pack(fill='x')
        
        self.chord_frame = ttk.Frame(self.root, padding=10)
        self.chord_frame.pack(fill='x')
        
        self.virtual_keyboard_frame = ttk.Frame(self.root, padding=10)
        self.virtual_keyboard_frame.pack(fill='x')
        
        visualization_frame = ttk.Frame(self.root)
        visualization_frame.pack(fill='both', expand=True)
        
        # Control panel
        ttk.Label(self.control_frame, text="Jazz Improvisation AI", 
                 font=('TkDefaultFont', 16, 'bold')).grid(
                 row=0, column=0, columnspan=5, padx=5, pady=5)
        
        ttk.Button(self.control_frame, text="Start", command=self.toggle_playing).grid(
                 row=1, column=0, padx=5, pady=5)
        
        ttk.Label(self.control_frame, text="Tempo (BPM):").grid(row=1, column=1, padx=5, pady=5)
        self.tempo_var = tk.IntVar(value=self.tempo)
        tempo_scale = ttk.Scale(self.control_frame, from_=40, to=240, variable=self.tempo_var, 
                               orient='horizontal', command=self.update_tempo)
        tempo_scale.grid(row=1, column=2, padx=5, pady=5, sticky='ew')
        self.tempo_label = ttk.Label(self.control_frame, text=f"{self.tempo}")
        self.tempo_label.grid(row=1, column=3, padx=5, pady=5)
        
        # MIDI device selection
        ttk.Label(self.midi_frame, text="MIDI Input:").grid(row=0, column=0, padx=5, pady=5)
        self.input_device_var = tk.StringVar()
        self.input_device_combo = ttk.Combobox(self.midi_frame, textvariable=self.input_device_var, width=30)
        self.input_device_combo.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(self.midi_frame, text="MIDI Output:").grid(row=0, column=2, padx=5, pady=5)
        self.output_device_var = tk.StringVar()
        self.output_device_combo = ttk.Combobox(self.midi_frame, textvariable=self.output_device_var, width=30)
        self.output_device_combo.grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Button(self.midi_frame, text="Refresh Devices", 
                 command=self.refresh_devices).grid(row=0, column=4, padx=5, pady=5)
        
        ttk.Button(self.midi_frame, text="Create Virtual MIDI", 
                 command=self.setup_virtual_midi).grid(row=1, column=0, padx=5, pady=5)
        
        ttk.Button(self.midi_frame, text="Use Virtual Keyboard", 
                 command=self.toggle_virtual_keyboard).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Button(self.midi_frame, text="Simulate MIDI Input", 
                 command=self.start_input_simulation).grid(row=1, column=2, padx=5, pady=5)
        
        # Chord selection
        ttk.Label(self.chord_frame, text="Current Chord:").grid(row=0, column=0, padx=5, pady=5)
        
        # Common jazz chords
        self.chord_buttons = []
        common_chords = [
            "Cmaj7", "Dm7", "Em7", "Fmaj7", "G7", "Am7", "Bm7b5",
            "Ebmaj7", "Fm7", "Gm7", "Abmaj7", "Bb7", "Cm7", "Dm7b5"
        ]
        
        row, col = 1, 0
        for chord_name in common_chords:
            btn = ttk.Button(self.chord_frame, text=chord_name, 
                           command=lambda c=chord_name: self.set_current_chord(c),
                           width=8)
            btn.grid(row=row, column=col, padx=2, pady=2)
            self.chord_buttons.append(btn)
            col += 1
            if col > 6:  # 7 buttons per row
                col = 0
                row += 1
        
        # Custom chord entry
        ttk.Label(self.chord_frame, text="Custom Chord:").grid(row=row+1, column=0, padx=5, pady=5)
        self.custom_chord_var = tk.StringVar()
        custom_chord_entry = ttk.Entry(self.chord_frame, textvariable=self.custom_chord_var)
        custom_chord_entry.grid(row=row+1, column=1, columnspan=2, padx=5, pady=5, sticky='ew')
        ttk.Button(self.chord_frame, text="Set", 
                 command=lambda: self.set_current_chord(self.custom_chord_var.get())).grid(
                 row=row+1, column=3, padx=5, pady=5)
        
        # Current chord display
        self.current_chord_var = tk.StringVar(value=self.current_chord)
        ttk.Label(self.chord_frame, textvariable=self.current_chord_var, 
                font=('TkDefaultFont', 12, 'bold')).grid(
                row=row+1, column=4, columnspan=3, padx=5, pady=5)
        
        # Virtual keyboard (initially hidden)
        self.virtual_keyboard_frame.pack_forget()
        
        # Setup visualization
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 6))
        self.fig.tight_layout(pad=3.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=visualization_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Status messages
        self.status_var = tk.StringVar(value="Ready. Select MIDI devices and press Start.")
        self.status_label = ttk.Label(self.root, textvariable=self.status_var, 
                                     font=('TkDefaultFont', 10))
        self.status_label.pack(side='bottom', fill='x', padx=10, pady=5)
        
        # Update tempo callback
        def delayed_tempo_update(*args):
            self.tempo = self.tempo_var.get()
            self.tempo_label.config(text=f"{self.tempo}")
        
        self.tempo_var.trace_add("write", delayed_tempo_update)
        
        # Periodic UI update
        self.update_visualization()
        
        # When window is closed
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_virtual_keyboard(self):
        """Create a virtual keyboard UI"""
        # Clear existing keyboard
        for widget in self.virtual_keyboard_frame.winfo_children():
            widget.destroy()
        
        # Create piano keys
        # We'll use a simplified layout with one octave
        key_width = 40
        white_key_height = 120
        black_key_height = 80
        
        # White keys
        white_keys = [60, 62, 64, 65, 67, 69, 71]  # C4, D4, E4, F4, G4, A4, B4
        black_keys = [61, 63, 66, 68, 70]  # C#4, D#4, F#4, G#4, A#4
        
        key_frame = ttk.Frame(self.virtual_keyboard_frame)
        key_frame.pack(pady=10)
        
        # Create white keys
        for i, note_num in enumerate(white