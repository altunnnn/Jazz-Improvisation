import sys
import os
import numpy as np
import pygame
import tensorflow as tf
from tensorflow import keras
from music21 import chord, note, stream, scale, interval, pitch
import time
import matplotlib.pyplot as plt
from collections import deque
import random
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import rtmidi
import queue

# --- Constants ---
SEQUENCE_LENGTH = 16
BASS_NOTE_MIN = 32  # E1
BASS_NOTE_MAX = 60  # C4
NUM_BASS_NOTES = BASS_NOTE_MAX - BASS_NOTE_MIN + 1
STATE_SHAPE_MELODY = (SEQUENCE_LENGTH, 128)
STATE_SHAPE_CHORD = (SEQUENCE_LENGTH, 24)
STATE_SHAPE_BASS = (SEQUENCE_LENGTH, 1)
STATE_SHAPE_CONTEXT = (3,)


class JazzEnvironment:
    """
    Simulates the musical environment for the RL agent.
    Provides states and rewards.
    """
    def __init__(self, chord_progression, melody_sequence, tempo=120):
        self.chord_progression = chord_progression
        self.melody_sequence = melody_sequence
        self.tempo = tempo
        self.beats_per_chord = 4

        self.current_step = 0
        self.current_chord_index = 0
        self.current_melody_index = 0
        self.max_steps = len(melody_sequence)

        # State components
        self.melody_history = deque(maxlen=SEQUENCE_LENGTH)
        self.chord_history = deque(maxlen=SEQUENCE_LENGTH)
        self.bass_note_history = deque(maxlen=SEQUENCE_LENGTH)

        # Initialize histories with padding
        for _ in range(SEQUENCE_LENGTH):
            self.melody_history.append(-1)
            self.chord_history.append(None)
            self.bass_note_history.append(-1)

    def reset(self):
        """Reset the environment to the beginning of the sequence."""
        self.current_step = 0
        self.current_chord_index = 0
        self.current_melody_index = 0

        self.melody_history.clear()
        self.chord_history.clear()
        self.bass_note_history.clear()
        for _ in range(SEQUENCE_LENGTH):
            self.melody_history.append(-1)
            self.chord_history.append(None)
            self.bass_note_history.append(-1)

        # Add the first melody note and chord
        self._update_history()

        return self._get_state()

    def _update_history(self):
        """Internal function to update state histories based on current step."""
        # Update melody
        current_melody_note = self.melody_sequence[self.current_melody_index] if self.current_melody_index < len(self.melody_sequence) else -1
        self.melody_history.append(current_melody_note)

        # Update chord
        chord_idx = (self.current_step // self.beats_per_chord) % len(self.chord_progression)
        current_chord_name = self.chord_progression[chord_idx]
        self.chord_history.append(current_chord_name)

    def _get_state(self):
        """Compile the current state information for the agent."""
        # --- Melody ---
        melody_state = np.zeros(STATE_SHAPE_MELODY)
        for i, note_val in enumerate(self.melody_history):
            if 0 <= note_val < 128:
                melody_state[i, note_val] = 1

        # --- Chord ---
        chord_state = np.zeros(STATE_SHAPE_CHORD)
        for i, chord_name in enumerate(self.chord_history):
            if chord_name:
                try:
                    c = chord.Chord(chord_name)
                    root = c.root().pitchClass
                    chord_state[i, root] = 1  # One-hot encode root

                    # Simplified chord type encoding
                    if c.isDominantSeventh(): type_idx = 12
                    elif c.isMinorSeventh(): type_idx = 13
                    elif c.isMajorSeventh(): type_idx = 14
                    elif c.isMinorTriad(): type_idx = 15
                    elif c.isMajorTriad(): type_idx = 16
                    else: type_idx = 17  # Other/Unknown
                    chord_state[i, type_idx] = 1
                except Exception:
                    chord_state[i, 0] = 1  # Default to C if parsing fails
                    chord_state[i, 16] = 1  # Default to Major

        # --- Bass History ---
        bass_state = np.array(list(self.bass_note_history)).reshape(STATE_SHAPE_BASS)
        bass_state = (bass_state - BASS_NOTE_MIN) / NUM_BASS_NOTES  # Normalize 0-1
        bass_state[bass_state < 0] = -1  # Keep rests/padding as -1

        # --- Context ---
        tempo_norm = self.tempo / 240.0  # Normalize tempo
        current_chord_name = self.chord_history[-1]
        root_norm = 0.0
        type_norm = 0.0  # 0: Major, 0.33: Minor, 0.66: Dominant, 1: Other
        if current_chord_name:
            try:
                c = chord.Chord(current_chord_name)
                root_norm = c.root().pitchClass / 12.0
                if c.isMajorSeventh() or c.isMajorTriad(): type_norm = 0.0
                elif c.isMinorSeventh() or c.isMinorTriad(): type_norm = 0.33
                elif c.isDominantSeventh(): type_norm = 0.66
                else: type_norm = 1.0
            except Exception: pass
        context_state = np.array([tempo_norm, root_norm, type_norm]).reshape(STATE_SHAPE_CONTEXT)

        # Return state components as a dictionary
        return {
            "melody": melody_state.reshape(1, *STATE_SHAPE_MELODY),
            "chord": chord_state.reshape(1, *STATE_SHAPE_CHORD),
            "bass": bass_state.reshape(1, *STATE_SHAPE_BASS),
            "context": context_state.reshape(1, *STATE_SHAPE_CONTEXT)
        }

    def _calculate_reward(self, bass_action):
        """
        Calculate the reward for the chosen bass note (action).
        """
        reward = 0.0
        bass_note_midi = bass_action + BASS_NOTE_MIN

        current_chord_name = self.chord_history[-1]
        prev_bass_note_midi = self.bass_note_history[-1] if len(self.bass_note_history) > 0 else -1

        if not current_chord_name:
            return 0.0

        try:
            c = chord.Chord(current_chord_name)
            chord_pitches = c.pitches
            chord_tone_classes = {p.pitchClass for p in chord_pitches}
            root_pc = c.root().pitchClass
            bass_note_obj = note.Note(midi=bass_note_midi)
            bass_pc = bass_note_obj.pitch.pitchClass

            # 1. Chord Tone Bonus
            if bass_pc in chord_tone_classes:
                reward += 0.5
                # Bonus for being the root
                if bass_pc == root_pc:
                    reward += 0.3
                # Bonus for being 3rd or 5th
                elif bass_pc == (root_pc + 4) % 12 or bass_pc == (root_pc + 3) % 12 or bass_pc == (root_pc + 7) % 12:
                     reward += 0.1

            # 2. Voice Leading Reward
            if prev_bass_note_midi > 0:
                prev_bass_note_obj = note.Note(midi=prev_bass_note_midi)
                inter = interval.Interval(prev_bass_note_obj, bass_note_obj)
                # Penalize large leaps
                if abs(inter.semitones) > 12:
                    reward -= 0.3
                # Reward steps
                elif abs(inter.semitones) <= 2:
                    reward += 0.2
                # Small reward for consonant leaps
                elif abs(inter.semitones) <= 9:
                    reward += 0.05

            # 3. Dissonance Penalty
            clash_penalty = 0
            for p in chord_pitches:
                 if p.pitchClass in [root_pc, (root_pc + 3)%12, (root_pc + 4)%12, (root_pc + 7)%12]:
                     diff = abs(bass_pc - p.pitchClass)
                     if diff == 1 or diff == 11:  # Semitone clash
                         clash_penalty = -0.6
                         break
            reward += clash_penalty

        except Exception:
            pass

        # Normalize reward
        reward = np.clip(reward, -1.0, 1.0)
        return reward

    def step(self, action):
        """
        Apply the agent's action, update state, calculate reward, and return results.
        """
        if not (0 <= action < NUM_BASS_NOTES):
            raise ValueError(f"Invalid action: {action}")

        # Calculate reward based on the current state before updating history
        reward = self._calculate_reward(action)

        # Update bass note history with the chosen action
        bass_note_midi = action + BASS_NOTE_MIN
        self.bass_note_history.append(bass_note_midi)

        # Advance time
        self.current_step += 1
        self.current_melody_index += 1

        # Update melody and chord histories for the next state
        self._update_history()

        # Check if done
        done = self.current_step >= self.max_steps

        # Get the next state
        next_state = self._get_state()

        return next_state, reward, done, {}


class RLJazzAgent:
    """
    The Reinforcement Learning Agent that learns to improvise bass lines.
    Uses a Deep Q-Network (DQN).
    """
    def __init__(self, state_shapes, num_actions, learning_rate=0.001, gamma=0.95, epsilon=0.1):
        self.state_shapes = state_shapes
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.midi_handler = MidiHandler()
        self.agent = None
        self.environment = None
        self.is_improvising = False
        self.improv_thread = None
        self.is_waiting_for_input = False  
        self.received_melody_notes = []    # Storage for received melody notes

    def _build_model(self):
        """Build the Q-Network model."""
        # Inputs for each part of the state
        melody_input = keras.layers.Input(shape=self.state_shapes["melody"][1:], name="melody_input")
        chord_input = keras.layers.Input(shape=self.state_shapes["chord"][1:], name="chord_input")
        bass_input = keras.layers.Input(shape=self.state_shapes["bass"][1:], name="bass_input")
        context_input = keras.layers.Input(shape=self.state_shapes["context"][1:], name="context_input")

        # Process melody
        m = keras.layers.LSTM(64, return_sequences=True)(melody_input)
        m = keras.layers.LSTM(32)(m)
        m = keras.layers.LayerNormalization()(m)

        # Process chords
        c = keras.layers.LSTM(32, return_sequences=True)(chord_input)
        c = keras.layers.LSTM(16)(c)
        c = keras.layers.LayerNormalization()(c)

        # Process bass history
        b = keras.layers.LSTM(16)(bass_input)
        b = keras.layers.LayerNormalization()(b)

        # Process context
        ctx = keras.layers.Dense(16, activation='relu')(context_input)

        # Combine features
        combined = keras.layers.Concatenate()([m, c, b, ctx])

        # Dense layers
        x = keras.layers.Dense(128, activation='relu')(combined)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(64, activation='relu')(x)

        # Output layer: Q-values for each possible bass note action
        q_values = keras.layers.Dense(self.num_actions, activation='linear', name="q_values")(x)

        model = keras.Model(inputs=[melody_input, chord_input, bass_input, context_input], outputs=q_values)
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def choose_action(self, state):
        """Choose action using epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions)  # Explore
        
        # Exploit: Predict Q-values and choose the best action
        act_values = self.model.predict([state["melody"], state["chord"], state["bass"], state["context"]], verbose=0)
        return np.argmax(act_values[0])  # Returns action index

    def load(self, name):
        try:
            self.model.load_weights(name)
            return True
        except:
            return False


class MidiHandler:
    """
    Handles MIDI input and output operations.
    """
    def __init__(self):
        self.midi_in = rtmidi.MidiIn()
        self.midi_out = rtmidi.MidiOut()
        self.available_inputs = self.midi_in.get_ports()
        self.available_outputs = self.midi_out.get_ports()
        self.notes_queue = queue.Queue()
        self.is_listening = False
        self.listen_thread = None
        
    def refresh_devices(self):
        """Refresh available MIDI devices."""
        self.midi_in = rtmidi.MidiIn()
        self.midi_out = rtmidi.MidiOut()
        self.available_inputs = self.midi_in.get_ports()
        self.available_outputs = self.midi_out.get_ports()
        return self.available_inputs, self.available_outputs
        
    def open_input(self, port_name):
        """Open a MIDI input port by name."""
        if port_name in self.available_inputs:
            port_idx = self.available_inputs.index(port_name)
            try:
                self.midi_in.close_port()
                self.midi_in.open_port(port_idx)
                self.midi_in.set_callback(self._midi_callback)
                return True
            except:
                return False
        return False
        
    def open_output(self, port_name):
        """Open a MIDI output port by name."""
        if port_name in self.available_outputs:
            port_idx = self.available_outputs.index(port_name)
            try:
                self.midi_out.close_port()
                self.midi_out.open_port(port_idx)
                return True
            except:
                return False
        return False
        
    def _midi_callback(self, message, time_stamp):
        """Called when a MIDI message is received."""
        message_type = message[0][0] & 0xF0
        # Note On message
        if message_type == 0x90 and message[0][2] > 0:  # Note On with velocity > 0
            note_value = message[0][1]
            self.notes_queue.put(note_value)
            
    def start_listening(self):
        """Start listening for incoming MIDI notes."""
        if not self.is_listening:
            self.is_listening = True
            self.notes_queue = queue.Queue()
            self.listen_thread = threading.Thread(target=self._listen_loop)
            self.listen_thread.daemon = True
            self.listen_thread.start()
            
    def stop_listening(self):
        """Stop listening for incoming MIDI notes."""
        self.is_listening = False
        if self.listen_thread:
            self.listen_thread.join(1.0)
            
    def _listen_loop(self):
        """Background thread that listens for incoming MIDI notes."""
        while self.is_listening:
            time.sleep(0.01)  # Small sleep to prevent CPU hogging
            
    def get_received_notes(self, max_notes=None):
        """Get received notes from the queue."""
        notes = []
        try:
            while not self.notes_queue.empty() and (max_notes is None or len(notes) < max_notes):
                notes.append(self.notes_queue.get_nowait())
        except queue.Empty:
            pass
        return notes
        
    def send_note(self, note_value, velocity=80, duration=0.25):
        """Send a MIDI note through the output port."""
        if self.midi_out and self.midi_out.is_port_open():
            # Note On
            self.midi_out.send_message([0x90, note_value, velocity])
            time.sleep(duration)
            # Note Off
            self.midi_out.send_message([0x80, note_value, 0])
            return True
        return False
        
    def close(self):
        """Close all MIDI ports."""
        self.stop_listening()
        if self.midi_in and self.midi_in.is_port_open():
            self.midi_in.close_port()
        if self.midi_out and self.midi_out.is_port_open():
            self.midi_out.close_port()


class JazzBassImprovisationApp:
    """
    Main application class for the Jazz Bass Improvisation UI.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Jazz Bass Improvisation Plugin")
        self.root.geometry("1200x800")
        self.root.configure(bg="#2D2D2D")
        
        # Set dark theme colors
        self.bg_color = "#2D2D2D"
        self.label_color = "#CCCCCC"
        self.button_color = "#444444"
        self.text_color = "#EEEEEE"
        self.header_color = "#555555"
        self.status_color = "#222222"
        
        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background=self.bg_color)
        self.style.configure('TLabel', background=self.bg_color, foreground=self.label_color)
        self.style.configure('TButton', background=self.button_color, foreground=self.text_color)
        self.style.map('TButton', background=[('active', '#666666')])
        self.style.configure('Header.TLabel', background=self.header_color, foreground=self.text_color, font=('Helvetica', 16, 'bold'))
        self.style.configure('TCombobox', fieldbackground=self.button_color, background=self.button_color, foreground=self.text_color)
        self.style.configure('TEntry', fieldbackground=self.button_color, foreground=self.text_color)
        
        # Initialize components
        self.midi_handler = MidiHandler()
        self.agent = None
        self.environment = None
        self.is_improvising = False
        self.improv_thread = None
        
        # Create UI elements
        self._create_title()
        self._create_midi_config()
        self._create_improv_settings()
        self._create_model_config()
        self._create_chord_prog()
        self._create_buttons()
        self._create_status_bar()
        self._create_log_area()
        
        # Refresh MIDI devices on startup
        self.refresh_midi_devices()
        
    def _create_title(self):
        """Create the main title."""
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        
        title_label = ttk.Label(title_frame, text="Jazz Bass Improvisation", style='Header.TLabel')
        title_label.pack(fill=tk.X)
        
    def _create_midi_config(self):
        """Create MIDI device configuration section."""
        section_frame = ttk.Frame(self.root)
        section_frame.pack(fill=tk.X, padx=10, pady=5)
        
        section_label = ttk.Label(section_frame, text="MIDI Configuration", font=('Helvetica', 10, 'bold'))
        section_label.pack(fill=tk.X, pady=(0, 5))
        
        # Input device
        input_frame = ttk.Frame(section_frame)
        input_frame.pack(fill=tk.X, pady=2)
        
        input_label = ttk.Label(input_frame, text="Input Device:", width=14, anchor=tk.W)
        input_label.pack(side=tk.LEFT, padx=5)
        
        self.input_combobox = ttk.Combobox(input_frame, state="readonly", width=40)
        self.input_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        refresh_button = ttk.Button(input_frame, text="Refresh", command=self.refresh_midi_devices)
        refresh_button.pack(side=tk.RIGHT, padx=5)
        
        # Output device
        output_frame = ttk.Frame(section_frame)
        output_frame.pack(fill=tk.X, pady=2)
        
        output_label = ttk.Label(output_frame, text="Output Device:", width=14, anchor=tk.W)
        output_label.pack(side=tk.LEFT, padx=5)
        
        self.output_combobox = ttk.Combobox(output_frame, state="readonly", width=40)
        self.output_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
    def _create_improv_settings(self):
        """Create improvisation settings section."""
        section_frame = ttk.Frame(self.root)
        section_frame.pack(fill=tk.X, padx=10, pady=5)
        
        section_label = ttk.Label(section_frame, text="Improvisation Settings", font=('Helvetica', 10, 'bold'))
        section_label.pack(fill=tk.X, pady=(0, 5))
        
        # Genre
        genre_frame = ttk.Frame(section_frame)
        genre_frame.pack(fill=tk.X, pady=2)
        
        genre_label = ttk.Label(genre_frame, text="Genre:", width=14, anchor=tk.W)
        genre_label.pack(side=tk.LEFT, padx=5)
        
        self.genre_combobox = ttk.Combobox(genre_frame, state="readonly", values=["Jazz", "Funk", "Latin", "Blues"])
        self.genre_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.genre_combobox.current(0)
        
        # Mode
        mode_frame = ttk.Frame(section_frame)
        mode_frame.pack(fill=tk.X, pady=2)
        
        mode_label = ttk.Label(mode_frame, text="Mode:", width=14, anchor=tk.W)
        mode_label.pack(side=tk.LEFT, padx=5)
        
        self.mode_combobox = ttk.Combobox(mode_frame, state="readonly", 
                                          values=["Automatic", "Responsive", "Training"])
        self.mode_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.mode_combobox.current(0)
        
        # Tempo
        tempo_frame = ttk.Frame(section_frame)
        tempo_frame.pack(fill=tk.X, pady=2)
        
        tempo_label = ttk.Label(tempo_frame, text="Tempo (BPM):", width=14, anchor=tk.W)
        tempo_label.pack(side=tk.LEFT, padx=5)
        
        self.tempo_var = tk.StringVar(value="120")
        tempo_entry = ttk.Entry(tempo_frame, textvariable=self.tempo_var, width=5)
        tempo_entry.pack(side=tk.LEFT, padx=5)
        
        tempo_scale = ttk.Scale(tempo_frame, from_=60, to=200, orient=tk.HORIZONTAL, 
                                command=lambda v: self.tempo_var.set(str(int(float(v)))))
        tempo_scale.set(120)
        tempo_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
    def _create_model_config(self):
        """Create model configuration section."""
        section_frame = ttk.Frame(self.root)
        section_frame.pack(fill=tk.X, padx=10, pady=5)
        
        section_label = ttk.Label(section_frame, text="Model Configuration", font=('Helvetica', 10, 'bold'))
        section_label.pack(fill=tk.X, pady=(0, 5))
        
        # Model loading
        model_frame = ttk.Frame(section_frame)
        model_frame.pack(fill=tk.X, pady=2)
        
        load_button = ttk.Button(model_frame, text="Load Model", command=self.load_model)
        load_button.pack(side=tk.LEFT, padx=5)
        
        self.model_status_var = tk.StringVar(value="No model loaded")
        model_status = ttk.Label(model_frame, textvariable=self.model_status_var, anchor=tk.W, 
                                background="#333333", foreground=self.text_color)
        model_status.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
    def _create_chord_prog(self):
        """Create chord progression section."""
        section_frame = ttk.Frame(self.root)
        section_frame.pack(fill=tk.X, padx=10, pady=5)
        
        section_label = ttk.Label(section_frame, text="Chord Progression", font=('Helvetica', 10, 'bold'))
        section_label.pack(fill=tk.X, pady=(0, 5))
        
        # Chord progression entry
        chord_label = ttk.Label(section_frame, text="Enter chord progression (comma separated):")
        chord_label.pack(fill=tk.X, pady=(0, 5))
        
        self.chord_var = tk.StringVar(value="Cmaj7, Dm7, G7, Cmaj7")
        chord_entry = ttk.Entry(section_frame, textvariable=self.chord_var)
        chord_entry.pack(fill=tk.X, padx=5, pady=2)
        
        # Presets
        preset_frame = ttk.Frame(section_frame)
        preset_frame.pack(fill=tk.X, pady=2)
        
        preset_label = ttk.Label(preset_frame, text="Presets:", width=8, anchor=tk.W)
        preset_label.pack(side=tk.LEFT, padx=5)
        
        self.preset_combobox = ttk.Combobox(preset_frame, state="readonly", 
                                        values=["2-5-1 in C", "Blues in F", "Autumn Leaves", "So What", "Basic 2-5-1"])
        self.preset_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.preset_combobox.bind("<<ComboboxSelected>>", self.load_preset)
        
    def _create_buttons(self):
        """Create main action buttons."""
        buttons_frame = ttk.Frame(self.root)
        buttons_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.start_button = ttk.Button(buttons_frame, text="Start Improvisation", command=self.toggle_improvisation)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        save_button = ttk.Button(buttons_frame, text="Save Settings", command=self.save_settings)
        save_button.pack(side=tk.LEFT, padx=5)
        
    def _create_status_bar(self):
        """Create status bar."""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        status_label = ttk.Label(status_frame, text="Status:", width=8, anchor=tk.W)
        status_label.pack(side=tk.LEFT, padx=5)
        
        self.status_var = tk.StringVar(value="Ready")
        status_value = ttk.Label(status_frame, textvariable=self.status_var, 
                                background="#1A5E1A", foreground="#FFFFFF")
        status_value.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
    def _create_log_area(self):
        """Create log area."""
        log_frame = ttk.Frame(self.root)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        log_label = ttk.Label(log_frame, text="Log")
        log_label.pack(fill=tk.X, pady=(0, 5))
        
        self.log_text = tk.Text(log_frame, bg="#333333", fg="#CCCCCC", height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.insert(tk.END, "[22:04:03] MIDI devices refreshed\n")
        self.log_text.config(state=tk.DISABLED)
        
    def refresh_midi_devices(self):
        """Refresh available MIDI devices."""
        inputs, outputs = self.midi_handler.refresh_devices()
        
        self.input_combobox['values'] = inputs if inputs else ["No MIDI inputs found"]
        if inputs:
            self.input_combobox.current(0)
            
        self.output_combobox['values'] = outputs if outputs else ["No MIDI outputs found"]
        if outputs:
            self.output_combobox.current(0)
            
        self.log(f"MIDI devices refreshed")
        
    
    def load_model(self):
        """Load a trained model."""
        file_path = filedialog.askopenfilename(
            title="Select model weights file",
            filetypes=[("H5 files", "*.h5"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            # Initialize state shapes for agent
            state_shapes = {
                "melody": (1, *STATE_SHAPE_MELODY),
                "chord": (1, *STATE_SHAPE_CHORD),
                "bass": (1, *STATE_SHAPE_BASS),
                "context": (1, *STATE_SHAPE_CONTEXT)
            }
            
            # Create or update agent
            self.agent = RLJazzAgent(state_shapes=state_shapes, num_actions=NUM_BASS_NOTES)
            if self.agent.load(file_path):
                self.model_status_var.set(f"Model loaded: {os.path.basename(file_path)}")
                self.log(f"Model loaded from {file_path}")
            else:
                raise Exception("Failed to load model weights")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.model_status_var.set("No model loaded")
            self.agent = None
            self.log(f"Error loading model: {str(e)}")

    def load_preset(self, event=None):
        """Load a chord progression preset."""
        preset = self.preset_combobox.get()
        chord_progs = {
            "2-5-1 in C": "Dm7, G7, Cmaj7, Cmaj7",
            "Blues in F": "F7, Bb7, F7, C7, Bb7, F7",
            "Autumn Leaves": "Cm7, F7, Bbmaj7, Eb7, Am7b5, D7, Gm7",
            "So What": "Dm7, Ebm7",
            "Basic 2-5-1": "Dm7, G7, Cmaj7"
        }
        if preset in chord_progs:
            self.chord_var.set(chord_progs[preset])
            self.log(f"Loaded preset: {preset}")

    def save_settings(self):
        """Save current settings to a file."""
        settings = {
            "input_device": self.input_combobox.get(),
            "output_device": self.output_combobox.get(),
            "genre": self.genre_combobox.get(),
            "mode": self.mode_combobox.get(),
            "tempo": self.tempo_var.get(),
            "chord_progression": self.chord_var.get()
        }
        file_path = filedialog.asksaveasfilename(
            title="Save settings",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            try:
                import json
                with open(file_path, 'w') as f:
                    json.dump(settings, f, indent=4)
                self.log(f"Settings saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save settings: {str(e)}")
                self.log(f"Error saving settings: {str(e)}")

    def log(self, message):
        """Add a message to the log area."""
        self.log_text.config(state=tk.NORMAL)
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def toggle_improvisation(self):
        """Start or stop the improvisation process."""
        if self.is_improvising:
            self.stop_improvisation()
        else:
            self.start_improvisation()

    def start_improvisation(self):
        """Start the improvisation process."""
        if not self.agent:
            messagebox.showerror("Error", "No model loaded. Please load a model first.")
            return

        try:
            # Get settings
            tempo = int(self.tempo_var.get())
            chord_str = self.chord_var.get()
            mode = self.mode_combobox.get()
            input_device = self.input_combobox.get()
            output_device = self.output_combobox.get()

            # Validate chord progression
            chord_progression = [c.strip() for c in chord_str.split(',')]
            if not chord_progression or not all(chord_progression):
                raise ValueError("Invalid chord progression")

            # Initialize MIDI
            if not self.midi_handler.open_input(input_device):
                raise ValueError(f"Could not open MIDI input: {input_device}")
            if not self.midi_handler.open_output(output_device):
                raise ValueError(f"Could not open MIDI output: {output_device}")

            # Clear any previous received notes
            self.received_melody_notes = []
                
            # Handle different modes
            if mode == "Responsive":
                self.midi_handler.start_listening()
                self.is_waiting_for_input = True
                self.status_var.set("Waiting for MIDI input...")
                self.log("Waiting for MIDI input. Play something on your keyboard...")
                
                # Start a wait thread
                self.is_improvising = True
                self.start_button.configure(text="Stop Improvisation")
                self.improv_thread = threading.Thread(target=self.wait_for_input_loop, 
                                                    args=(chord_progression, tempo))
                self.improv_thread.daemon = True
                self.improv_thread.start()
            else:
                # Automatic mode - use default melody
                melody_sequence = [60] * len(chord_progression) * 4  # Placeholder: Middle C
                
                # Initialize environment and start improvisation
                self.environment = JazzEnvironment(
                    chord_progression=chord_progression,
                    melody_sequence=melody_sequence,
                    tempo=tempo
                )
                
                self.is_improvising = True
                self.start_button.configure(text="Stop Improvisation")
                self.status_var.set("Improvising...")
                self.improv_thread = threading.Thread(target=self.improv_loop)
                self.improv_thread.daemon = True
                self.improv_thread.start()
                self.log("Automatic improvisation started")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start improvisation: {str(e)}")
            self.log(f"Error starting improvisation: {str(e)}")
            self.stop_improvisation()

    def wait_for_input_loop(self, chord_progression, tempo):
        """Wait for MIDI input before starting actual improvisation."""
        input_wait_timeout = 30  # Maximum seconds to wait for input
        start_time = time.time()
        min_notes_required = 2  # Minimum notes required before starting
        
        while self.is_improvising and self.is_waiting_for_input:
            # Check if we've timed out
            elapsed = time.time() - start_time
            if elapsed > input_wait_timeout:
                self.log("Input wait timeout. Starting with default melody.")
                # Create default melody if no input received
                melody_sequence = [60] * len(chord_progression) * 4
                self.is_waiting_for_input = False
                break
                
            # Check for new notes
            new_notes = self.midi_handler.get_received_notes()
            if new_notes:
                self.received_melody_notes.extend(new_notes)
                self.log(f"Received MIDI notes: {new_notes}")
                
                # Flash status to provide feedback
                self.status_var.set(f"Received {len(self.received_melody_notes)} notes...")
                
                # If we have enough notes, start improvisation
                if len(self.received_melody_notes) >= min_notes_required:
                    # Give a little buffer time for additional input
                    if len(new_notes) > 0:
                        self.log("Input received. Starting in 1 second...")
                        time.sleep(1)  # Brief pause to allow for more input
                        # Get any remaining notes
                        additional_notes = self.midi_handler.get_received_notes()
                        if additional_notes:
                            self.received_melody_notes.extend(additional_notes)
                    self.is_waiting_for_input = False
                    break
                    
            time.sleep(0.1)  # Small sleep to prevent CPU hogging
        
        # If we're still improvising but no longer waiting for input,
        # start the actual improvisation
        if self.is_improvising and not self.is_waiting_for_input:
            # Use the received notes or default if none
            melody_sequence = self.received_melody_notes if self.received_melody_notes else [60] * len(chord_progression) * 4
            
            # Initialize environment with the melody
            self.environment = JazzEnvironment(
                chord_progression=chord_progression,
                melody_sequence=melody_sequence,
                tempo=tempo
            )
            
            self.status_var.set("Improvising...")
            self.log(f"Starting improvisation with {len(melody_sequence)} melody notes")
            
            # Start the improvisation loop
            self.improv_loop()
    

    def stop_improvisation(self):
        """Stop the improvisation process."""
        self.is_improvising = False
        self.is_waiting_for_input = False  # Make sure to reset this flag
        self.midi_handler.stop_listening()
        if self.improv_thread:
            self.improv_thread.join(1.0)
        self.start_button.configure(text="Start Improvisation")
        self.status_var.set("Ready")
        self.log("Improvisation stopped")

    def improv_loop(self):
        """Main improvisation loop running in a separate thread."""
        state = self.environment.reset()
        step_duration = 60.0 / self.environment.tempo  # Seconds per beat
        mode = self.mode_combobox.get()
        
        # For responsive mode, keep a buffer of upcoming notes
        note_buffer = []
        response_delay = 1  # Number of notes to wait before responding
        
        while self.is_improvising:
            try:
                # Check for new input in responsive mode
                if mode == "Responsive":
                    new_notes = self.midi_handler.get_received_notes()
                    if new_notes:
                        note_buffer.extend(new_notes)
                        self.log(f"Added {len(new_notes)} notes to buffer")
                        
                    # Update melody if buffer has notes
                    if note_buffer:
                        # Add next note from buffer to environment
                        next_note = note_buffer.pop(0)
                        self.environment.melody_sequence.append(next_note)
                        self.environment.max_steps = len(self.environment.melody_sequence)
                        self.log(f"Playing in response to melody note: {next_note}")
                    else:
                        # If no new notes, briefly pause
                        time.sleep(0.1)
                        continue
                        
                # Get action from agent
                action = self.agent.choose_action(state)
                bass_note_midi = action + BASS_NOTE_MIN

                # Play the bass note
                self.midi_handler.send_note(
                    note_value=bass_note_midi,
                    velocity=80,
                    duration=step_duration * 0.8  # Slightly shorter than full beat
                )

                # Step environment
                next_state, reward, done, _ = self.environment.step(action)
                self.log(f"Played bass note: {bass_note_midi}, Reward: {reward:.2f}")

                state = next_state

                if done:
                    # In automatic mode, reset environment
                    if mode != "Responsive":
                        state = self.environment.reset()
                        self.log("Sequence completed, resetting environment")
                    else:
                        # In responsive mode, wait for more input
                        if len(note_buffer) == 0:
                            self.log("Waiting for more melody input...")
                            time.sleep(step_duration)

                time.sleep(step_duration * 0.2)  # Small pause between notes

            except Exception as e:
                self.log(f"Error in improvisation loop: {str(e)}")
                self.root.after(0, lambda: messagebox.showerror("Error", f"Improvisation error: {str(e)}"))
                break

        self.root.after(0, self.stop_improvisation)

    def run(self):
        """Start the Tkinter main loop."""
        self.root.mainloop()

    def cleanup(self):
        """Cleanup resources when closing the application."""
        self.stop_improvisation()
        self.midi_handler.close()
        self.root.destroy()


def main():
    """Main entry point for the application."""
    root = tk.Tk()
    app = JazzBassImprovisationApp(root)
    try:
        app.run()
    except KeyboardInterrupt:
        app.cleanup()


if __name__ == "__main__":
    main()