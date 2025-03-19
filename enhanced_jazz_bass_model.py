import numpy as np
import tensorflow as tf
from tensorflow import keras
from music21 import chord, note, stream, scale, interval
import time
import matplotlib.pyplot as plt
import pygame
import os
import random
from sklearn.model_selection import train_test_split
import sys
import glob
import pandas as pd
from collections import deque

class EnhancedJazzBassModel:
    def __init__(self, model_path=None):
        """Initialize the enhanced jazz bass generation model with improved features"""
        self.sequence_length = 64  # Increased for better musical context
        self.is_playing = False
        self.notes_buffer = []
        self.chords_buffer = []
        self.tempo = 120  # Default tempo in BPM
        self.last_note_time = 0  # To track timing between notes
        self.prev_notes = deque(maxlen=8)  # Keep track of previous notes to avoid repetition
        self.beat_position = 0.0  # Track position within the measure
        self.measures_played = 0
        self.style = "walking"  # Default style: walking, latin, modal, etc.
        
        # Enhanced midi representation with musical meaning
        self.use_relative_pitch = True
        self.use_functional_encoding = True
        
        # Define key centers and their corresponding scales with added modes
        self.key_centers = {
            'C': {
                'major': scale.MajorScale('C'),
                'minor': scale.MinorScale('C'),
                'dorian': scale.DorianScale('C'),
                'mixolydian': scale.MixolydianScale('C'),
                'lydian': scale.LydianScale('C')
            },
            'F': {
                'major': scale.MajorScale('F'),
                'minor': scale.MinorScale('F'),
                'dorian': scale.DorianScale('F'),
                'mixolydian': scale.MixolydianScale('F'),
                'lydian': scale.LydianScale('F')
            },
            'Bb': {
                'major': scale.MajorScale('B-'),
                'minor': scale.MinorScale('B-'),
                'dorian': scale.DorianScale('B-'),
                'mixolydian': scale.MixolydianScale('B-'),
                'lydian': scale.LydianScale('B-')
            },
            'Eb': {
                'major': scale.MajorScale('E-'),
                'minor': scale.MinorScale('E-'),
                'dorian': scale.DorianScale('E-'),
                'mixolydian': scale.MixolydianScale('E-'),
                'lydian': scale.LydianScale('E-')
            },
            'G': {
                'major': scale.MajorScale('G'),
                'minor': scale.MinorScale('G'),
                'dorian': scale.DorianScale('G'),
                'mixolydian': scale.MixolydianScale('G'),
                'lydian': scale.LydianScale('G')
            },
            'D': {
                'major': scale.MajorScale('D'),
                'minor': scale.MinorScale('D'),
                'dorian': scale.DorianScale('D'),
                'mixolydian': scale.MixolydianScale('D'),
                'lydian': scale.LydianScale('D')
            },
            'A': {
                'major': scale.MajorScale('A'),
                'minor': scale.MinorScale('A'),
                'dorian': scale.DorianScale('A'),
                'mixolydian': scale.MixolydianScale('A'),
                'lydian': scale.LydianScale('A')
            }
        }
        
        # Enhanced jazz chord progressions with extensions and substitutions
        self.jazz_progressions = {
            'basic': [
                ['Cmaj7', 'Dm7', 'G7', 'Cmaj7'],  # I-ii-V-I in C
                ['Dm7', 'G7', 'Cmaj7', 'Cmaj7'],  # ii-V-I-I in C
                ['Dm7', 'G7', 'Em7', 'A7'],       # ii-V-iii-VI in C
                ['Fmaj7', 'Bb7', 'Cmaj7', 'G7'],  # IV-bVII-I-V in C
                ['Cmaj7', 'A7', 'Dm7', 'G7'],     # I-VI-ii-V in C
                ['Dm7', 'G7', 'Cmaj7', 'A7', 'Dm7', 'G7'] # ii-V-I-VI-ii-V in C
            ],
            'modal': [
                ['Dm7', 'Em7', 'Fmaj7', 'G7'],    # Dorian vamp
                ['Cmaj7', 'D7sus4', 'Ebmaj7', 'D7sus4'], # Lydian vamp
                ['G7sus4', 'G7', 'G7sus4', 'G7'],  # Mixolydian vamp
                ['Am7', 'D7sus4', 'Gmaj7', 'Cmaj7'], # Modal interchange
            ],
            'substitutions': [
                ['Cmaj7', 'Eb7', 'Dm7', 'Db7'],   # With tritone subs
                ['Dm7', 'G7b9', 'Cmaj7', 'B7b5'],  # With altered dominants
                ['Dm7', 'G7#11', 'Cmaj7#11', 'Fmaj7/G'], # With extensions
                ['Dm9', 'G13', 'Cmaj9', 'Ebmaj7#11'] # Rich harmonies
            ],
            'latin': [
                ['Cm6', 'Fm7', 'Dm7b5', 'G7b9'],  # Minor bossa
                ['Fmaj7', 'G7', 'Em7b5', 'A7b9'],  # Latin turnaround
                ['Dm7', 'G9', 'Cmaj7', 'Bbmaj7']   # Samba progression
            ]
        }
        
        # Enhanced bass line patterns with more variety and musical intention
        self.bass_patterns = {
            'walking': [
                [0, 7, 3, 5],      # Root, 5th, 3rd, approach
                [0, 3, 5, 7],      # Root, 3rd, 5th, 7th
                [0, -2, 3, 5],     # Root, 7th below, 3rd, 5th
                [0, 4, 7, 10],     # Root, chord tones
                [0, 2, 4, 5],      # Root, scale walk up
                [0, -1, -2, -3]    # Root, chromatic walk down
            ],
            'latin': [
                [0, 5, 0, 5],      # Tumbao pattern 1
                [0, 0, 5, 7],      # Tumbao pattern 2
                [0, 0, 7, 10],     # Montuno pattern
                [0, 5, 7, 0],      # Bossa pattern
                [-3, 0, 0, 0],     # Anticipation pattern
            ],
            'modal': [
                [0, 0, 2, 4],      # Modal pattern 1
                [0, 2, 4, 7],      # Modal pattern 2
                [0, 0, 0, 7],      # Pedal with 5th
                [0, 7, 0, 7],      # Ostinato pattern
                [0, 4, 0, 4],      # Quartal pattern
            ],
            'funk': [
                [0, 0, 3, 0],      # Funk pattern 1
                [0, 10, 0, 7],     # Funk pattern 2
                [0, -5, 0, 0],     # Root-fifth pattern
                [0, 0, -7, 0],     # Syncopated pattern
            ]
        }
        
        # Voice leading and contour patterns based on music theory
        self.voice_leading_rules = {
            'strong_root': {
                'beats': [0.0],  # First beat of measure
                'preference': ['root', 'fifth']
            },
            'weak_beats': {
                'beats': [0.5, 1.5, 2.5, 3.5],  # Upbeats
                'preference': ['third', 'seventh', 'tension', 'approach']
            },
            'cadence': {
                'chords': ['V7', 'viidim7'],
                'preference': ['leading_tone', 'seventh', 'approach']
            },
            'resolution': {
                'chords': ['Imaj7', 'vi7'],
                'preference': ['root', 'third']
            }
        }
        
        # Rhythmic patterns for different styles
        self.rhythm_patterns = {
            'walking': [1, 1, 1, 1],  # Quarter notes
            'latin': [1.5, 0.5, 1, 1],  # Dotted eighth + sixteenth
            'funk': [2, 0, 1.5, 0.5],  # Half note + dotted eighth + sixteenth
            'broken': [1, 0, 1, 0, 1, 0, 1],  # Broken feel
        }
        
        if model_path and os.path.exists(model_path):
            try:
                print(f"Loading model from {model_path}")
                keras.config.enable_unsafe_deserialization()
                self.model = keras.models.load_model(model_path)
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.build_model()
        else:
            print("Building new model")
            self.build_model()
        
        self.current_key = 'C'
        self.current_mode = 'major'
        self.current_chord_root = 0  # C
        self.current_chord_type = 'maj7'
            
    def build_model(self):
        """Build an enhanced deep learning model for jazz bass generation"""
        print("Building enhanced model architecture...")
        
        # Define the dimensions for our enhanced input representations
        pitch_dim = 128  # Standard MIDI range
        chord_features_dim = 36  # 12 roots × 3 chord types
        musical_context_dim = 24  # Additional musical features (key, mode, position, etc.)
        
        # Input layers with named inputs for clarity
        melody_input = keras.layers.Input(shape=(self.sequence_length, pitch_dim), name="melody_input")
        chord_input = keras.layers.Input(shape=(self.sequence_length, chord_features_dim), name="chord_input")
        tempo_input = keras.layers.Input(shape=(1,), name="tempo_input")
        style_input = keras.layers.Input(shape=(4,), name="style_input")  # One-hot encoded style
        context_input = keras.layers.Input(shape=(self.sequence_length, musical_context_dim), name="context_input")
        
        # Process melody with bi-directional LSTM and attention
        x1 = keras.layers.LayerNormalization()(melody_input)
        x1 = keras.layers.Bidirectional(
            keras.layers.LSTM(128, return_sequences=True)
        )(x1)
        x1 = keras.layers.LayerNormalization()(x1)
        x1 = keras.layers.Dropout(0.3)(x1)
        x1 = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(x1)
        
        # Self-attention mechanism for melody using Keras layers instead of raw TF ops
        query = keras.layers.Dense(64)(x1)
        key = keras.layers.Dense(64)(x1)
        value = keras.layers.Dense(64)(x1)
        
        # Use MultiHeadAttention layer instead of raw TF operations
        melody_features = keras.layers.MultiHeadAttention(
            num_heads=1, key_dim=64
        )(query, key, value)
        
        # Global average pooling to reduce sequence dimension
        melody_features = keras.layers.GlobalAveragePooling1D()(melody_features)
        melody_features = keras.layers.LayerNormalization()(melody_features)
        
        # Process chord with bi-directional LSTM and attention
        x2 = keras.layers.LayerNormalization()(chord_input)
        x2 = keras.layers.Bidirectional(
            keras.layers.LSTM(64, return_sequences=True)
        )(x2)
        x2 = keras.layers.LayerNormalization()(x2)
        x2 = keras.layers.Dropout(0.3)(x2)
        
        # Self-attention for chords - using Keras layers
        chord_query = keras.layers.Dense(32)(x2)
        chord_key = keras.layers.Dense(32)(x2)
        chord_value = keras.layers.Dense(32)(x2)
        
        # Use MultiHeadAttention for chords as well
        chord_features = keras.layers.MultiHeadAttention(
            num_heads=1, key_dim=32
        )(chord_query, chord_key, chord_value)
        
        # Global average pooling for chords
        chord_features = keras.layers.GlobalAveragePooling1D()(chord_features)
        chord_features = keras.layers.LayerNormalization()(chord_features)
        
        # Process musical context
        context_features = keras.layers.Bidirectional(
            keras.layers.LSTM(32, return_sequences=False)
        )(context_input)
        context_features = keras.layers.LayerNormalization()(context_features)
        
        # Process tempo with normalization
        tempo_normalized = keras.layers.Lambda(lambda x: x / 300.0)(tempo_input)  # Normalize to 0-1 range
        tempo_features = keras.layers.Dense(16, activation='relu')(tempo_normalized)
        
        # Process style one-hot encoding
        style_features = keras.layers.Dense(16, activation='relu')(style_input)
        
        # Combine all features
        combined = keras.layers.Concatenate()([
            melody_features, 
            chord_features, 
            context_features,
            tempo_features,
            style_features
        ])
        
        # Dense layers with residual connections for better gradient flow
        x = keras.layers.Dense(256, activation='relu')(combined)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Dropout(0.3)(x)
        
        residual = x
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Add()([x, residual])  # Residual connection
        
        # Second residual block - FIXED to maintain consistent dimensions
        residual2 = x
        x = keras.layers.Dense(256, activation='relu')(x)  # Changed from 128 to 256 to match residual2
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(256, activation='relu')(x)  # Changed from 128 to 256 to match residual2
        x = keras.layers.Add()([x, residual2])  # Second residual connection
        
        # Final dimensionality reduction before output layer
        x = keras.layers.Dense(128, activation='relu')(x)
        
        # Output layer - bass note prediction in the common bass range (E1 to G3, MIDI 28-55)
        bass_range = 28  # Number of notes in typical bass range
        outputs = keras.layers.Dense(bass_range, activation='softmax', name='bass_note')(x)
        
        # Construct the complete model
        self.model = keras.Model(
            inputs=[melody_input, chord_input, tempo_input, style_input, context_input],
            outputs=outputs
        )
        
        # Use Adam optimizer with learning rate scheduling
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=10000,
            decay_rate=0.9
        )
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        
        # Compile with categorical crossentropy for multi-class classification
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model architecture built successfully!")
    
    def setup_midi_input(self, port_name="any"):
        """Set up MIDI input for real-time bass generation"""
        try:
            import pygame.midi
            # Initialize pygame midi
            pygame.midi.init()
            
            # Find available input devices
            input_devices = []
            for i in range(pygame.midi.get_count()):
                info = pygame.midi.get_device_info(i)
                if info[2] == 1:  # is input device
                    input_devices.append((i, info[1].decode('utf-8')))
            
            if not input_devices:
                print("No MIDI input devices found.")
                return False
            
            # Select a device
            selected_device = None
            if port_name.lower() == "any":
                selected_device = input_devices[0][0]
            else:
                for device_id, device_name in input_devices:
                    if port_name.lower() in device_name.lower():
                        selected_device = device_id
                        break
            
            if selected_device is None:
                print(f"No MIDI device matching '{port_name}' found.")
                print("Available devices:")
                for device_id, device_name in input_devices:
                    print(f"  {device_id}: {device_name}")
                return False
            
            # Open the device
            self.midi_input = pygame.midi.Input(selected_device)
            
            # Set up a callback handler
            self.setup_midi_callback()
            
            return True
        
        except ImportError:
            print("pygame.midi not available. Please install pygame.")
            return False
        except Exception as e:
            print(f"Error setting up MIDI: {str(e)}")
            return False
    
    def setup_midi_callback(self):
        """Set up a callback handler for MIDI input"""
        import threading
        
        def midi_callback_thread():
            current_chord = "Cmaj7"  # Default chord
            current_beat = 0.0
            last_note_time = 0
            
            print("MIDI callback active. Play notes to generate bass responses.")
            
            while self.midi_input:
                if self.midi_input.poll():
                    midi_events = self.midi_input.read(10)
                    midi_notes = []
                    
                    for event in midi_events:
                        # Extract note data
                        data = event[0]
                        status = data[0]
                        
                        # Note on events
                        if 144 <= status <= 159:  # Note on events (0x90 - 0x9F)
                            note = data[1]
                            velocity = data[2]
                            if velocity > 0:  # Note on with velocity > 0
                                midi_notes.append(note)
                                
                                # Determine chord from note
                                chord = self.determine_chord_from_note(note)
                                
                                # Only respond after a small delay to avoid too many notes
                                current_time = time.time()
                                if current_time - last_note_time > 0.5:
                                    # Generate bass note
                                    bass_note = self.predict_next_bass_note(
                                        [note], [chord], [current_beat]
                                    )
                                    
                                    # Output result
                                    print(f"Input: {note}, Chord: {chord}, Bass: {bass_note}")
                                    
                                    # Update time
                                    last_note_time = current_time
                                    
                                    # Update beat
                                    current_beat = (current_beat + 1) % 4
                
                time.sleep(0.01)  # Small delay to prevent CPU hogging
        
        # Start the callback thread
        self.midi_thread = threading.Thread(target=midi_callback_thread)
        self.midi_thread.daemon = True
        self.midi_thread.start()

    def set_style(self, style):
        """Set the playing style for bass generation"""
        valid_styles = ["walking", "latin", "modal", "funk"]
        if style in valid_styles:
            self.style = style
            print(f"Style set to {style}")
        else:
            print(f"Unknown style '{style}'. Using default style 'walking'")
            self.style = "walking"
    
    def determine_chord_from_note(self, note):
        """Determine a likely chord based on a MIDI note"""
        # This is a simplified method that could be expanded
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note_name = note_names[note % 12]
        
        # For demo purposes, return a major chord
        return f"{note_name}maj7"

    def generate_bass_note_for_chord(self, chord):
        """Generate a bass note for a given chord"""
        # Simple implementation - return the root note
        return self.extract_root_from_chord(chord)

    def set_tempo(self, bpm):
        """Set tempo for playback and generation"""
        if 40 <= bpm <= 300:
            self.tempo = bpm
            print(f"Tempo set to {bpm} BPM")
        else:
            print(f"Tempo {bpm} out of range (40-300 BPM). Using default 120 BPM")
            self.tempo = 120
    
    def set_key(self, key, mode='major'):
        """Set the musical key for better generation context"""
        valid_modes = ['major', 'minor', 'dorian', 'mixolydian', 'lydian']
        
        if key in self.key_centers and mode in valid_modes:
            self.current_key = key
            self.current_mode = mode
            print(f"Musical key set to {key} {mode}")
        else:
            print(f"Invalid key or mode. Using default C major")
            self.current_key = 'C'
            self.current_mode = 'major'
    
    def encode_chord(self, chord_name):
        """Enhanced chord encoding with music theory awareness"""
        # Parse chord name into root and quality
        if len(chord_name) >= 1:
            # Extract root note
            if len(chord_name) >= 2 and chord_name[1] in ['b', '#', '-']:
                root = chord_name[:2]
                quality = chord_name[2:]
            else:
                root = chord_name[0]
                quality = chord_name[1:]
            
            # Convert root to MIDI note number (C=0, C#=1, etc.)
            root_map = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4,
                       'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9,
                       'A#': 10, 'Bb': 10, 'B': 11}
            
            if root in root_map:
                root_num = root_map[root]
            else:
                print(f"Unknown root note in chord: {chord_name}")
                root_num = 0  # Default to C
                
            # Create one-hot encoding for root (12 dimensions)
            root_vector = np.zeros(12)
            root_vector[root_num] = 1
            
            # Create encoding for chord quality (24 dimensions total: 12 for root, 12 for quality)
            quality_vector = np.zeros(24)
            
            # Copy root information to first 12 dimensions
            quality_vector[:12] = root_vector
            
            # Set chord quality in remaining dimensions
            if 'maj7' in quality:
                quality_vector[12] = 1  # Major 7th
            elif 'maj9' in quality:
                quality_vector[12] = 1  # Major 7th (base)
                quality_vector[13] = 1  # With 9th
            elif 'm7' in quality:
                quality_vector[14] = 1  # Minor 7th
            elif 'm9' in quality:
                quality_vector[14] = 1  # Minor 7th (base)
                quality_vector[15] = 1  # With 9th
            elif '7b9' in quality:
                quality_vector[16] = 1  # Dominant 7th (base)
                quality_vector[17] = 1  # With flat 9th
            elif '7#11' in quality:
                quality_vector[16] = 1  # Dominant 7th (base)
                quality_vector[18] = 1  # With sharp 11th
            elif '7' in quality:
                quality_vector[16] = 1  # Dominant 7th
            elif 'dim7' in quality:
                quality_vector[19] = 1  # Diminished 7th
            elif 'm7b5' in quality:
                quality_vector[20] = 1  # Half-diminished 7th
            elif 'sus4' in quality:
                quality_vector[21] = 1  # Suspended 4th
            elif 'm' in quality:
                quality_vector[22] = 1  # Minor triad
            else:
                quality_vector[23] = 1  # Major triad (default)
            
            # Create extended vector with additional functional harmony information (36 dimensions total)
            extended_vector = np.zeros(36)
            extended_vector[:24] = quality_vector
            
            # Determine chord function within the current key
            if self.current_key in self.key_centers:
                key_obj = self.key_centers[self.current_key][self.current_mode]
                key_root = note.Note(self.current_key).pitch.midi % 12
                
                # Calculate relative position in the key (0=tonic, 7=dominant, etc.)
                degree = (root_num - key_root) % 12
                
                # Mark common functional positions 
                if degree == 0:
                    extended_vector[24] = 1  # Tonic
                elif degree == 7:
                    extended_vector[25] = 1  # Dominant
                elif degree == 2:
                    extended_vector[26] = 1  # Supertonic
                elif degree == 5:
                    extended_vector[27] = 1  # Subdominant
                elif degree == 9:
                    extended_vector[28] = 1  # Submediant
                
                # Mark modal interchange chords
                if self.current_mode == 'major' and 'm' in quality and degree != 2 and degree != 9:
                    extended_vector[29] = 1  # Modal interchange - minor chord in major key
                elif self.current_mode == 'minor' and 'm' not in quality and degree != 3 and degree != 10:
                    extended_vector[30] = 1  # Modal interchange - major chord in minor key
                
                # Mark secondary dominants
                if '7' in quality and degree != 7:
                    extended_vector[31] = 1  # Secondary dominant
            
            return extended_vector
        else:
            # Return neutral chord encoding if invalid chord name
            result = np.zeros(36)
            result[23] = 1  # Default to C major
            return result
                
    def preprocess_midi_input(self, midi_input):
        """Preprocess MIDI notes for model input with enhanced musicality"""
        # Create a sequence of one-hot encoded vectors for each note in the input
        note_sequence = np.zeros((1, self.sequence_length, 128))
        
        # Fill the sequence with the most recent notes (up to sequence_length)
        for i in range(min(len(midi_input), self.sequence_length)):
            idx = self.sequence_length - min(len(midi_input), self.sequence_length) + i
            note_val = midi_input[i]
            if 0 <= note_val < 128:  # Valid MIDI note range
                note_sequence[0, idx, note_val] = 1
        
        return note_sequence
    
    def preprocess_chord_input(self, chord_input):
        """Preprocess chord names for model input"""
        # Create a sequence of encoded vectors for each chord
        chord_sequence = np.zeros((1, self.sequence_length, 36))
        
        # Fill the sequence with encoded chords (up to sequence_length)
        for i in range(min(len(chord_input), self.sequence_length)):
            idx = self.sequence_length - min(len(chord_input), self.sequence_length) + i
            chord_sequence[0, idx] = self.encode_chord(chord_input[i])
        
        return chord_sequence
    
    def create_context_vector(self, beat_positions):
        """Generate musical context vector with enhanced features"""
        context_sequence = np.zeros((1, self.sequence_length, 24))
        
        # Fill the sequence with context information for each position
        for i in range(min(len(beat_positions), self.sequence_length)):
            idx = self.sequence_length - min(len(beat_positions), self.sequence_length) + i
            beat_pos = beat_positions[i]
            
            # Encode beat position (first 4 dimensions)
            # One-hot encode which beat of the measure we're on
            beat_idx = int(beat_pos) % 4
            context_sequence[0, idx, beat_idx] = 1
            
            # Encode fraction within beat (next 4 dimensions)
            beat_fraction = beat_pos % 1.0
            if beat_fraction < 0.25:
                context_sequence[0, idx, 4] = 1
            elif beat_fraction < 0.5:
                context_sequence[0, idx, 5] = 1
            elif beat_fraction < 0.75:
                context_sequence[0, idx, 6] = 1
            else:
                context_sequence[0, idx, 7] = 1
            
            # Encode measure position (next 4 dimensions)
            measure_idx = int(beat_pos / 4) % 4
            context_sequence[0, idx, 8 + measure_idx] = 1
            
            # Encode key information (next 12 dimensions)
            key_root = note.Note(self.current_key).pitch.midi % 12
            context_sequence[0, idx, 12 + key_root] = 1
            
            # Encode mode (remaining dimensions)
            mode_map = {'major': 0, 'minor': 1, 'dorian': 2, 'mixolydian': 3, 'lydian': 4}
            if self.current_mode in mode_map:
                context_sequence[0, idx, 12 + 12 + mode_map[self.current_mode]] = 1
        
        return context_sequence
    
    def create_style_vector(self):
        """Create one-hot encoded style vector"""
        style_vector = np.zeros((1, 4))
        style_map = {'walking': 0, 'latin': 1, 'modal': 2, 'funk': 3}
        if self.style in style_map:
            style_vector[0, style_map[self.style]] = 1
        else:
            style_vector[0, 0] = 1  # Default to walking bass
        
        return style_vector
    
    def predict_next_bass_note(self, melody_notes, chord_names, beat_positions):
        """Predict the next bass note with enhanced musical awareness"""
        # Preprocess inputs
        melody_data = self.preprocess_midi_input(melody_notes)
        chord_data = self.preprocess_chord_input(chord_names)
        context_data = self.create_context_vector(beat_positions)
        style_data = self.create_style_vector()
        tempo_data = np.array([[self.tempo]])
        
        # Make prediction
        prediction = self.model.predict(
            [melody_data, chord_data, tempo_data, style_data, context_data],
            verbose=0
        )[0]
        
        # Determine whether to use deterministic (argmax) or probabilistic sampling
        if random.random() < 0.8:  # 80% deterministic for stability
            bass_note_rel = np.argmax(prediction)
        else:  # 20% probabilistic for creativity
            bass_note_rel = np.random.choice(len(prediction), p=prediction)
        
        # Convert to absolute MIDI note (bass range starts at E1 = MIDI 28)
        bass_note = bass_note_rel + 28
        
        # Apply musical constraints:
        chord_name = chord_names[-1] if chord_names else "Cmaj7"
        current_chord = self.parse_chord(chord_name)
        current_beat = beat_positions[-1] % 4 if beat_positions else 0.0
        
        # Avoid excessive repetition
        if len(self.prev_notes) >= 3 and all(note == bass_note for note in self.prev_notes):
            # Force different note if we've had the same note 3+ times
            chord_tones = self.get_chord_tones(current_chord, bass_range=True)
            if chord_tones and len(chord_tones) > 1:
                alternatives = [note for note in chord_tones if note != bass_note]
                if alternatives:
                    bass_note = random.choice(alternatives)
        
        # Apply voice leading rules based on beat position
        for rule_name, rule in self.voice_leading_rules.items():
            if any(abs(beat - current_beat) < 0.1 for beat in rule['beats']):
                # Apply the rule for this beat position
                bass_note = self.apply_voice_leading(bass_note, current_chord, rule)
                break
        
        # Update history
        self.prev_notes.append(bass_note)
        
        return bass_note
    
    def parse_chord(self, chord_name):
        """Parse chord name into root and type"""
        result = {}
        
        # Extract root note
        if len(chord_name) >= 2 and chord_name[1] in ['b', '#', '-']:
            result['root'] = chord_name[:2]
            quality = chord_name[2:]
        else:
            result['root'] = chord_name[0]
            quality = chord_name[1:]
        
        # Set chord quality
        if 'maj7' in quality:
            result['type'] = 'maj7'
        elif 'm7b5' in quality:
            result['type'] = 'm7b5'
        elif 'm7' in quality:
            result['type'] = 'm7'
        elif '7' in quality:
            result['type'] = '7'
        elif 'dim7' in quality:
            result['type'] = 'dim7'
        elif 'm' in quality:
            result['type'] = 'm'
        else:
            result['type'] = 'maj'
        
        # Add any extensions or alterations
        if '9' in quality:
            result['extensions'] = ['9']
        if '11' in quality:
            result['extensions'] = result.get('extensions', []) + ['11']
        if '13' in quality:
            result['extensions'] = result.get('extensions', []) + ['13']
        if '#11' in quality:
            result['alterations'] = ['#11']
        if 'b9' in quality:
            result['alterations'] = result.get('alterations', []) + ['b9']
        
        return result
    
    def get_chord_tones(self, chord_dict, bass_range=True):
        """Get MIDI note numbers for chord tones, optionally in bass range"""
        if not chord_dict:
            return []
            
        root_map = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4,
                   'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9,
                   'A#': 10, 'Bb': 10, 'B': 11}
        
        if chord_dict['root'] not in root_map:
            return []
            
        root = root_map[chord_dict['root']]
        chord_type = chord_dict['type']
        
        # Define intervals based on chord type
        intervals = []
        if chord_type == 'maj7':
            intervals = [0, 4, 7, 11]  # root, major 3rd, perfect 5th, major 7th
        elif chord_type == 'm7':
            intervals = [0, 3, 7, 10]  # root, minor 3rd, perfect 5th, minor 7th
        elif chord_type == '7':
            intervals = [0, 4, 7, 10]  # root, major 3rd, perfect 5th, minor 7th
        elif chord_type == 'm7b5':
            intervals = [0, 3, 6, 10]  # root, minor 3rd, diminished 5th, minor 7th
        elif chord_type == 'dim7':
            intervals = [0, 3, 6, 9]   # root, minor 3rd, diminished 5th, diminished 7th
        elif chord_type == 'm':
            intervals = [0, 3, 7]      # root, minor 3rd, perfect 5th
        elif chord_type == 'maj':
            intervals = [0, 4, 7]      # root, major 3rd, perfect 5th
        
        # Add extensions if present
        if 'extensions' in chord_dict:
            if '9' in chord_dict['extensions']:
                intervals.append(14)  # 9th
            if '11' in chord_dict['extensions']:
                intervals.append(17)  # 11th
            if '13' in chord_dict['extensions']:
                intervals.append(21)  # 13th
        
        # Generate actual MIDI notes
        notes = []
        for octave in range(2, 5):  # E1 to G3 range for bass
            base = octave * 12
            for interval in intervals:
                note_value = base + root + interval
                if not bass_range or (28 <= note_value <= 55):  # E1 to G3
                    notes.append(note_value)
        
        return notes
    
    def apply_voice_leading(self, note_value, chord_dict, rule):
        """Apply voice leading rules to adjust the bass note if needed"""
        preference = rule.get('preference', [])
        chord_tones = self.get_chord_tones(chord_dict, bass_range=True)
        
        if not chord_tones:
            return note_value
            
        # Find the root note in bass range
        root_map = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4,
                   'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9,
                   'A#': 10, 'Bb': 10, 'B': 11}
        
        if chord_dict['root'] in root_map:
            root = root_map[chord_dict['root']]
            root_notes = [note for note in chord_tones if note % 12 == root]
            
            if 'root' in preference and root_notes:
                # Prefer root on strong beats
                return min(root_notes, key=lambda x: abs(x - 40))  # Around E2-F2
                
            if 'fifth' in preference:
                # Fifth above root
                fifth = (root + 7) % 12
                fifth_notes = [note for note in chord_tones if note % 12 == fifth]
                if fifth_notes:
                    return min(fifth_notes, key=lambda x: abs(x - 40))
            
            if 'third' in preference:
                # Third of the chord
                third = (root + 4) % 12 if chord_dict['type'] in ['maj', 'maj7', '7'] else (root + 3) % 12
                third_notes = [note for note in chord_tones if note % 12 == third]
                if third_notes:
                    return min(third_notes, key=lambda x: abs(x - 40))
            
            if 'seventh' in preference:
                # Seventh of the chord
                seventh = (root + 11) % 12 if chord_dict['type'] == 'maj7' else (root + 10) % 12
                seventh_notes = [note for note in chord_tones if note % 12 == seventh]
                if seventh_notes:
                    return min(seventh_notes, key=lambda x: abs(x - 40))
            
            if 'leading_tone' in preference and chord_dict['type'] == '7':
                # Leading tone (major 7th of the target chord, which is a half-step below root of next chord)
                # This is just a heuristic - would need the next chord to be fully accurate
                leading = (root + 11) % 12
                scale_notes = self.get_scale_notes(bass_range=True)
                leading_notes = [note for note in scale_notes if note % 12 == leading]
                if leading_notes:
                    return min(leading_notes, key=lambda x: abs(x - 40))
            
            if 'approach' in preference:
                # Chromatic approach - half step below or above next likely chord tone
                # For our purposes let's approach the root of current chord
                approach_below = max([note for note in range(28, 56) if note % 12 == (root - 1) % 12], default=note_value)
                approach_above = min([note for note in range(28, 56) if note % 12 == (root + 1) % 12], default=note_value)
                
                if abs(approach_below - note_value) < abs(approach_above - note_value):
                    return approach_below
                else:
                    return approach_above
        
        return note_value
    
    def get_scale_notes(self, bass_range=True):
        """Get the notes of the current scale, optionally in bass range"""
        if self.current_key in self.key_centers and self.current_mode in self.key_centers[self.current_key]:
            scale_obj = self.key_centers[self.current_key][self.current_mode]
            scale_degrees = [p.midi % 12 for p in scale_obj.getPitches()]
            
            # Generate actual MIDI notes across the bass range
            notes = []
            for octave in range(2, 5):  # E1 to G3 range for bass
                base = octave * 12
                for degree in scale_degrees:
                    note_value = base + degree
                    if not bass_range or (28 <= note_value <= 55):  # E1 to G3
                        notes.append(note_value)
            
            return notes
        else:
            # Default to C major if key or mode not found
            return [28, 30, 32, 33, 35, 37, 39, 40, 42, 44, 45, 47, 49, 51, 52, 54]
    
    def get_bass_pattern(self, chord_name, style=None):
        """Get appropriate bass pattern for the given chord and style"""
        if style is None:
            style = self.style
            
        if style not in self.bass_patterns:
            style = "walking"  # Default to walking bass
            
        # Choose a pattern for the style
        pattern = random.choice(self.bass_patterns[style])
        
        # Parse the chord
        chord_dict = self.parse_chord(chord_name)
        root_map = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4,
                   'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9,
                   'A#': 10, 'Bb': 10, 'B': 11}
                   
        if chord_dict['root'] not in root_map:
            return []
            
        root = root_map[chord_dict['root']]
        
        # Determine chord tones
        chord_type = chord_dict['type']
        if chord_type == 'maj7':
            third = (root + 4) % 12
            fifth = (root + 7) % 12
            seventh = (root + 11) % 12
        elif chord_type == 'm7':
            third = (root + 3) % 12
            fifth = (root + 7) % 12
            seventh = (root + 10) % 12
        elif chord_type == '7':
            third = (root + 4) % 12
            fifth = (root + 7) % 12
            seventh = (root + 10) % 12
        elif chord_type == 'm7b5':
            third = (root + 3) % 12
            fifth = (root + 6) % 12
            seventh = (root + 10) % 12
        else:
            # Default to major triad
            third = (root + 4) % 12
            fifth = (root + 7) % 12
            seventh = (root + 10) % 12  # Default to dominant seventh
        
        # Convert relative pattern to actual notes
        actual_notes = []
        for interval in pattern:
            if interval == 0:
                # Root
                actual_notes.append(root)
            elif interval == 3 or interval == 4:
                # Third (could be major or minor)
                actual_notes.append(third)
            elif interval == 5:
                # Fifth
                actual_notes.append(fifth)
            elif interval == 7:
                # Perfect fifth (always)
                actual_notes.append((root + 7) % 12)
            elif interval == 10 or interval == 11:
                # Seventh (could be dominant or major)
                actual_notes.append(seventh)
            elif interval == -1:
                # Half-step below root
                actual_notes.append((root - 1) % 12)
            elif interval == -2:
                # Whole-step below root
                actual_notes.append((root - 2) % 12)
            elif interval == -3:
                # Minor third below root
                actual_notes.append((root - 3) % 12)
            elif interval == -5:
                # Perfect fourth below root (fifth below octave)
                actual_notes.append((root - 5) % 12)
            elif interval == 2:
                # Major second above root
                actual_notes.append((root + 2) % 12)
            elif interval == 4:
                # Major third above root
                actual_notes.append((root + 4) % 12)
            else:
                # Default to root if unknown interval
                actual_notes.append(root)
        
        # Adjust to appropriate octave in bass range
        bass_notes = []
        base_octave = 3  # E2 is MIDI 40
        base_note = (base_octave * 12) + root
        
        # Ensure base note is in bass range
        while base_note < 28:  # E1
            base_note += 12
        while base_note > 43:  # G2
            base_note -= 12
            
        # Map the relative notes to actual MIDI notes
        for note_value in actual_notes:
            # Find the best octave for this note
            relative_to_root = (note_value - root) % 12
            absolute_note = base_note + relative_to_root
            
            # Ensure it's in bass range
            while absolute_note < 28:  # E1
                absolute_note += 12
            while absolute_note > 55:  # G3
                absolute_note -= 12
                
            bass_notes.append(absolute_note)
        
        return bass_notes
    
    def generate_rhythmic_bass_line(self, chord_progression, measures=1, beats_per_measure=4):
        """Generate a rhythmic bass line for the given chord progression"""
        bass_line = []
        
        # Distribute chords across measures
        chords_per_measure = max(1, len(chord_progression) // measures)
        expanded_progression = []
        for chord in chord_progression:
            # Repeat each chord for its duration
            expanded_progression.extend([chord] * chords_per_measure)
        
        # Generate beat positions for the entire progression
        beat_positions = []
        for measure in range(measures):
            for beat in range(beats_per_measure):
                beat_positions.append(measure * beats_per_measure + beat)
        
        # For each beat, predict a bass note
        for i, beat_pos in enumerate(beat_positions):
            current_chord_idx = min(i // beats_per_measure * chords_per_measure, len(expanded_progression) - 1)
            current_chord = expanded_progression[current_chord_idx]
            
            # Use different patterns based on position in measure
            if self.style == 'walking':
                pattern = self.get_bass_pattern(current_chord, self.style)
                pattern_idx = beat_pos % len(pattern)
                note = pattern[pattern_idx]
                bass_line.append(note)
            else:
                # If no specific pattern, use the predict_next_bass_note method
                # Provide dummy melody for now (can be enhanced later)
                dummy_melody = [60]  # Middle C
                chord_context = [current_chord]
                beat_context = [beat_pos]
                
                note = self.predict_next_bass_note(dummy_melody, chord_context, beat_context)
                bass_line.append(note)
        
        return bass_line

    def extract_root_from_chord(self, chord_name):
        """Extract the root note from a chord name and return as MIDI note number"""
        # Parse the chord name to get the root
        if len(chord_name) >= 2 and chord_name[1] in ['b', '#']:
            root = chord_name[:2]
        else:
            root = chord_name[0]
            
        # Map the root to a MIDI note number
        root_map = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4,
                'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9,
                'A#': 10, 'Bb': 10, 'B': 11}
        
        if root in root_map:
            # Return the MIDI note number for the root in the bass range (E1-G3)
            base_note = root_map[root] + 36  # C2
            
            # Ensure it's in the bass range
            while base_note < 28:  # E1
                base_note += 12
            while base_note > 55:  # G3
                base_note -= 12
                
            return base_note
        else:
            # Default to C if unknown root
            return 36  # C2
def main():
    """Main function to demonstrate the EnhancedJazzBassModel"""
    print("Enhanced Jazz Bass Model Demo")
    
    # Create model instance
    model = EnhancedJazzBassModel()
    
    # Set musical parameters
    model.set_key('F', 'dorian')
    model.set_tempo(120)
    model.set_style('walking')
    
    # Generate a bass line for a sample chord progression
    chord_progression = ['Dm7', 'G7', 'Cmaj7', 'A7']
    print(f"Generating bass line for chord progression: {chord_progression}")
    
    # Generate and print the bass line
    bass_line = model.generate_rhythmic_bass_line(chord_progression, measures=2)
    print("Generated bass notes:", bass_line)
    
    # Demonstrate real-time capabilities
    print("\nWould you like to setup MIDI input? (y/n)")
    choice = input().lower()
    if choice == 'y':
        port_success = model.setup_midi_input("any")
        if port_success:
            print("MIDI input setup successful! Ready for real-time generation.")
            print("Playing notes on your MIDI keyboard will generate bass responses.")
            print("Press Ctrl+C to exit.")
            try:
                # Keep the program running to receive MIDI input
                while True:
                    time.sleep(0.1)  # Small delay to prevent CPU hogging
            except KeyboardInterrupt:
                print("\nExiting...")
    else:
        print("MIDI setup failed. Demo concluded.")


if __name__ == "__main__":
    main()