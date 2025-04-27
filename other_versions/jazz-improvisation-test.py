import numpy as np
import tensorflow as tf
from tensorflow import keras
from music21 import chord, note, stream, scale
import time
import matplotlib.pyplot as plt
import pygame
import os
import random
from sklearn.model_selection import train_test_split
import sys

class JazzImprovisationModel:
    def __init__(self, model_path=None):
        """Initialize the jazz improvisation model"""
        self.sequence_length = 32  # Number of notes to consider for prediction
        self.is_playing = False
        self.notes_buffer = []
        self.chords_buffer = []
        self.tempo = 120  # Default tempo in BPM
        self.last_note_time = 0  # To track timing between notes

        if model_path:
            keras.config.enable_unsafe_deserialization()
            self.model = keras.models.load_model(model_path)
        else:
            self.build_model()
        
        # Key centers and their corresponding scales for more musical data generation
        self.key_centers = {
            'C': scale.MajorScale('C'),
            'F': scale.MajorScale('F'),
            'Bb': scale.MajorScale('B-'),
            'Eb': scale.MajorScale('E-'),
            'G': scale.MajorScale('G'),
            'D': scale.MajorScale('D'),
            'A': scale.MajorScale('A'),
        }
        
        # Common jazz chord progressions for structured training data
        self.jazz_progressions = [
            ['Cmaj7', 'Dm7', 'G7', 'Cmaj7'],  # I-ii-V-I in C
            ['Dm7', 'G7', 'Cmaj7', 'Cmaj7'],  # ii-V-I-I in C
            ['Dm7', 'G7', 'Em7', 'A7'],       # ii-V-iii-VI in C
            ['Fmaj7', 'Bb7', 'Cmaj7', 'G7'],  # IV-bVII-I-V in C
            ['Cmaj7', 'A7', 'Dm7', 'G7'],     # I-VI-ii-V in C
            ['Dm7', 'G7', 'Cmaj7', 'A7', 'Dm7', 'G7'] # ii-V-I-VI-ii-V in C
        ]
        
        # Bass line patterns based on common jazz walking bass approaches
        self.bass_patterns = [
            [0, 7, 3, 5],      # Root, 5th, 3rd, approach
            [0, 3, 5, 7],      # Root, 3rd, 5th, 7th
            [0, -2, 3, 5],     # Root, 7th below, 3rd, 5th
            [0, 4, 7, 10],     # Root, chord tones
            [0, 2, 4, 5],      # Root, scale walk up
            [0, -1, -2, -3]    # Root, chromatic walk down
        ]
        
        if model_path:
            self.model = keras.models.load_model(model_path)
        else:
            self.build_model()
    
    def build_model(self):
        """Build a deep learning model for jazz improvisation with improved architecture"""
        # Input layers
        melody_input = keras.layers.Input(shape=(self.sequence_length, 128))
        chord_input = keras.layers.Input(shape=(self.sequence_length, 24))
        tempo_input = keras.layers.Input(shape=(1,))
        # Process melody with LSTM
        x1 = keras.layers.Bidirectional(
            keras.layers.LSTM(128, return_sequences=True)
        )(melody_input)
        x1 = keras.layers.LayerNormalization()(x1)
        x1 = keras.layers.Dropout(0.3)(x1)
        x1 = keras.layers.Bidirectional(keras.layers.LSTM(64))(x1)
        x1 = keras.layers.LayerNormalization()(x1)
        
        # Process chord with LSTM
        x2 = keras.layers.Bidirectional(
            keras.layers.LSTM(64, return_sequences=True)
        )(chord_input)
        x2 = keras.layers.LayerNormalization()(x2)
        x2 = keras.layers.Dropout(0.3)(x2)
        x2 = keras.layers.Bidirectional(keras.layers.LSTM(128))(x2)
        x2 = keras.layers.LayerNormalization()(x1)

        tempo_normalized = keras.layers.Lambda(lambda x: x / 240.0)(tempo_input)  # Normalize to 0-1 range
        tempo_features = keras.layers.Dense(16, activation='relu')(tempo_normalized)
        
        # Combine melody and chord features
        combined = keras.layers.Concatenate()([x1, x2, tempo_features])
        
        # Dense layers with residual connections for better gradient flow
        x = keras.layers.Dense(256, activation='relu')(combined)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Dropout(0.3)(x)
        
        residual = x
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Add()([x, residual])  # Residual connection
        
        # Output layer focusing on bass range (32-60)
        # Smaller output range makes the prediction task easier
        bass_range_size = 29  # MIDI notes 32-60
        output = keras.layers.Dense(bass_range_size, activation='softmax')(x)
        
        # Create model
        self.model = keras.Model(inputs=[melody_input, chord_input, tempo_input], 
                                 outputs=output)
        
        # Use a fixed learning rate instead of a schedule to avoid the error
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
    
    def set_tempo(self,bpm):
        self.tempo = max(40,min(300,bpm))
        return self.tempo
    
    def setup_midi_input(self, port_name="loopMIDI Port"):
        """Set up MIDI input to receive notes from a virtual keyboard"""
        try:
            import pygame.midi
            pygame.midi.init()
            
            # List available MIDI devices
            print("Available MIDI input devices:")
            for i in range(pygame.midi.get_count()):
                try:
                    info = pygame.midi.get_device_info(i)
                    # Handle different pygame versions which might return different formats
                    if isinstance(info, tuple) and len(info) >= 2:
                        # Older pygame versions
                        device_name = str(info[1] if isinstance(info[1], str) else info[1].decode('utf-8', 'ignore'))
                        is_input = bool(info[2]) if len(info) > 2 else False
                        print(f"  Device {i}: {device_name} (Input: {is_input})")
                    else:
                        # Newer pygame versions or different format
                        print(f"  Device {i}: {info}")
                except Exception as e:
                    print(f"  Device {i}: Could not get info - {e}")
            
            # Find the specified input port or any available input port
            input_id = None
            for i in range(pygame.midi.get_count()):
                try:
                    info = pygame.midi.get_device_info(i)
                    device_name = ""
                    is_input = False
                    
                    # Handle different pygame versions
                    if isinstance(info, tuple):
                        if len(info) >= 2:
                            device_name = str(info[1] if isinstance(info[1], str) else info[1].decode('utf-8', 'ignore'))
                        if len(info) > 2:
                            is_input = bool(info[2])
                    
                    if is_input and (port_name in device_name or port_name == "any"):
                        input_id = i
                        print(f"Found MIDI input device: {device_name}")
                        break
                except Exception as e:
                    continue
                    
            if input_id is None:
                print(f"Could not find MIDI input port: {port_name}")
                print("Please choose from available devices listed above")
                print("Trying to use any available input device...")
                
                # Try to find any input device as fallback
                for i in range(pygame.midi.get_count()):
                    try:
                        info = pygame.midi.get_device_info(i)
                        if isinstance(info, tuple) and len(info) > 2 and bool(info[2]):
                            input_id = i
                            print(f"Using alternative MIDI input device: {i}")
                            break
                    except:
                        continue
                        
            if input_id is None:
                print("No MIDI input devices available")
                return False
                    
            # Open the input port
            self.midi_input = pygame.midi.Input(input_id)
            print(f"MIDI input initialized on device ID: {input_id}")
            return True
        except Exception as e:
            print(f"Error setting up MIDI input: {e}")
            return False

    def process_midi_input(self):
        """Process incoming MIDI input and update the model's buffers"""
        if not hasattr(self, 'midi_input'):
            return None
            
        # Check if there's data available
        if self.midi_input.poll():
            midi_events = self.midi_input.read(10)  # Read up to 10 events
            
            for event in midi_events:
                data = event[0]
                # Check if it's a note on event
                if data[0] & 0xF0 == 0x90 and data[2] > 0:  # Note on with velocity > 0
                    note_value = data[1]
                    self.notes_buffer.append(note_value)
                    
                    # Keep buffer at appropriate size
                    if len(self.notes_buffer) > self.sequence_length:
                        self.notes_buffer = self.notes_buffer[-self.sequence_length:]
                        
                    # Use a default chord for now
                    current_chord = 'Cmaj7'  # Could be improved with chord detection
                    self.chords_buffer.append(current_chord)
                    
                    # Keep chord buffer at appropriate size
                    if len(self.chords_buffer) > self.sequence_length:
                        self.chords_buffer = self.chords_buffer[-self.sequence_length:]
                        
                    # Generate bass note if we have enough data
                    if len(self.notes_buffer) >= 8 and len(self.chords_buffer) >= 8:
                        return self.predict_next_note(self.notes_buffer, self.chords_buffer)
        
        return None

    def interactive_test(self):
        """Run an interactive test with MIDI input and real-time output"""
        midi_already_setup = hasattr(self, 'midi_input') and self.midi_input is not None
        
        if not midi_already_setup and not self.setup_midi_input():
            print("Failed to set up MIDI input. Using test mode instead.")
            self.test_without_midi(audio=True)
            return
                
        print("Interactive test mode active. Play notes on your MIDI keyboard.")
        print("Press Ctrl+C to exit.")
        
        self.is_playing = True
        self.last_note_time = time.time()
        
        try:
            # Initialize audio
            pygame.mixer.init(44100, -16, 2, 512)
            
            # Sound cache
            sound_cache = {}
            
            # Function to create a sound
            def midi_to_freq(midi_note):
                return 440 * (2 ** ((midi_note - 69) / 12))
                    
            def create_sound(note_value):
                freq = midi_to_freq(note_value)
                sample_rate = 44100
                duration = 0.25
                t = np.linspace(0, duration, int(sample_rate * duration), False)
                
                # Bass-like tone
                wave = 0.7 * np.sin(freq * 2 * np.pi * t)
                wave += 0.2 * np.sin(2 * freq * 2 * np.pi * t)
                wave += 0.1 * np.sin(3 * freq * 2 * np.pi * t)
                
                # Envelope
                envelope = np.ones_like(wave)
                attack = int(0.01 * sample_rate)
                release = int(0.15 * sample_rate)
                envelope[:attack] = np.linspace(0, 1, attack)
                envelope[-release:] = np.linspace(1, 0, release)
                wave = wave * envelope
                
                stereo_wave = np.column_stack((wave, wave))
                stereo_wave = (stereo_wave * 32767).astype(np.int16)
                return pygame.sndarray.make_sound(stereo_wave)
            
            # Main interactive loop
            while self.is_playing:
                bass_note = self.process_midi_input()
                
                if bass_note is not None:
                    print(f"Playing bass note: {bass_note} ({note.Note(midi=bass_note).nameWithOctave})")
                    
                    # Create and play sound
                    if bass_note not in sound_cache:
                        sound_cache[bass_note] = create_sound(bass_note)
                    
                    sound_cache[bass_note].play()
                
                # Sleep a bit to prevent CPU overload
                time.sleep(0.01)
                    
        except KeyboardInterrupt:
            print("\nInteractive test stopped by user.")
        except Exception as e:
            print(f"Error in interactive test: {e}")
        finally:
            # Clean up
            if hasattr(self, 'midi_input'):
                self.midi_input.close()
            pygame.mixer.quit()
            pygame.midi.quit()

    def train(self, melody_data, tempo_data, chord_data, note_targets, epochs=100, batch_size=64):
        """Train the model with jazz music data and improved training process"""
        # Create callbacks for better training
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            ),
            keras.callbacks.ModelCheckpoint(
                filepath='jazz_model_checkpoint.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Train with callbacks
        history = self.model.fit(
            [melody_data, chord_data, tempo_data], 
            note_targets,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks
        )
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.tight_layout()
        plt.show()
        
        return history
    
    def predict_next_note(self, melody_input, chord_input):
        """Predict the next bass note based on melody and chord context"""
        if not melody_input or not chord_input:
            # Return a default note if inputs are empty
            return 40  # Default bass E note
            
        melody_processed = self.preprocess_midi_input(melody_input)
        chord_processed = self.preprocess_chord_input(chord_input)
        tempo_processed = np.array([[self.tempo]])
        
        # Get model prediction (adjusted for bass range 32-60)
        prediction = self.model.predict([melody_processed, chord_processed, tempo_processed], verbose=0)[0]
        
        # Calculate timing based on tempo
        current_time = time.time()
        time_since_last_note = current_time - self.last_note_time
        expected_time_between_notes = 60.0 / self.tempo  # seconds per beat

        # Update last note time
        self.last_note_time = current_time

        # FOR MORE CONSISTENT RESULTS: Take the most probable note instead of random sampling
        predicted_note = np.argmax(prediction) + 32  # Convert from 0-28 range to 32-60 range
        
        # Add musical constraints - ensure note fits with current chord
        current_chord = chord_input[-1] if chord_input else 'Cmaj7'
        try:
            c = chord.Chord(current_chord)
            chord_pitches = [p.midi % 12 for p in c.pitches]
            
            if predicted_note % 12 not in chord_pitches:
                # Note not in chord, try to adjust to closest chord tone
                closest_pitch = min(chord_pitches, key=lambda p: abs((predicted_note % 12) - p))
                octave = predicted_note // 12
                predicted_note = (octave * 12) + closest_pitch
        except:
            pass  # If chord parsing fails, keep the original prediction
        
        return predicted_note

    def _get_temperature_for_tempo(self, tempo):
        """Adjust temperature based on tempo - faster tempo, lower temperature for precision"""
        if tempo < 80:
            return 1.2  # More random at slower tempos
        elif tempo < 120:
            return 1.0  # Balanced at medium tempos
        elif tempo < 180:
            return 0.8  # More precise at faster tempos
        else:
            return 0.7  # Very precise at very fast tempos

    def preprocess_midi_input(self, midi_notes):
        """Convert MIDI notes to a format suitable for the model"""
        # One-hot encode MIDI notes
        encoded_notes = np.zeros((len(midi_notes), 128))
        for i, note_num in enumerate(midi_notes):
            if 0 <= note_num < 128:
                encoded_notes[i, note_num] = 1
        
        # Ensure we have the right sequence length
        if len(encoded_notes) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(encoded_notes), 128))
            encoded_notes = np.vstack([padding, encoded_notes])
        elif len(encoded_notes) > self.sequence_length:
            encoded_notes = encoded_notes[-self.sequence_length:]
        
        return encoded_notes.reshape(1, self.sequence_length, 128)
    
    def preprocess_chord_input(self, chord_names):
        """Convert chord names to a format suitable for the model"""
        # Encode chords to a simpler representation (root + type)
        encoded_chords = np.zeros((len(chord_names), 24))  # 12 roots + 12 chord types
        
        for i, chord_name in enumerate(chord_names):
            if chord_name:
                try:
                    c = chord.Chord(chord_name)
                    root = c.root().pitchClass
                    # Encode root
                    encoded_chords[i, root] = 1
                    
                    # Encode chord type based on intervals
                    chord_type = 12
                    if c.isDominantSeventh():
                        chord_type = 12
                    elif c.isMinorTriad():
                        chord_type = 13
                    elif c.isMajorTriad():
                        chord_type = 14
                    elif c.isMinorSeventh():
                        chord_type = 15
                    elif c.isMajorSeventh():
                        chord_type = 16
                    # Add more chord types as needed
                    
                    encoded_chords[i, chord_type] = 1
                except:
                    # Default to C major if parsing fails
                    encoded_chords[i, 0] = 1
                    encoded_chords[i, 14] = 1
        
        # Ensure we have the right sequence length
        if len(encoded_chords) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(encoded_chords), 24))
            encoded_chords = np.vstack([padding, encoded_chords])
        elif len(encoded_chords) > self.sequence_length:
            encoded_chords = encoded_chords[-self.sequence_length:]
        
        return encoded_chords.reshape(1, self.sequence_length, 24)
    
    def interactive_collection_mode(self):
        """Collect notes from MIDI keyboard first, then generate a complete bass line"""
        midi_already_setup = hasattr(self, 'midi_input') and self.midi_input is not None
        
        if not midi_already_setup and not self.setup_midi_input():
            print("Failed to set up MIDI input. Using test mode instead.")
            self.test_without_midi(audio=True)
            return
        
        print("\n=== MIDI Collection Mode ===")
        print("Play a sequence of notes on your MIDI keyboard.")
        print("The notes will be collected to form a melody.")
        print("Press Ctrl+C when you're done to generate the bass line.")
        
        # Initialize collection buffer
        collected_notes = []
        collected_times = []
        start_time = time.time()
        
        try:
            print("\nRecording your notes... (Press Ctrl+C when finished)")
            
            while True:
                # Check if there's MIDI input
                if self.midi_input.poll():
                    midi_events = self.midi_input.read(10)
                    
                    for event in midi_events:
                        data = event[0]
                        # Check if it's a note on event
                        if data[0] & 0xF0 == 0x90 and data[2] > 0:  # Note on with velocity > 0
                            note_value = data[1]
                            velocity = data[2]
                            current_time = time.time() - start_time
                            
                            collected_notes.append(note_value)
                            collected_times.append(current_time)
                            
                            print(f"Recorded note: {note_value} ({note.Note(midi=note_value).nameWithOctave}) at time {current_time:.2f}s")
                
                time.sleep(0.01)  # Small delay to prevent CPU overload
                
        except KeyboardInterrupt:
            print("\nNote collection finished. Processing...")
        
        if not collected_notes:
            print("No notes were collected. Using default sequence.")
            collected_notes = [60, 64, 67, 72, 60, 64, 67, 72]  # C major arpeggio
        
        # Calculate note durations based on timing
        if len(collected_notes) > 1:
            avg_duration = collected_times[-1] / len(collected_notes)
            self.tempo = int(60 / avg_duration)
            print(f"Estimated tempo: {self.tempo} BPM")
        else:
            self.tempo = 120
            print(f"Using default tempo: {self.tempo} BPM")
        
        # Use a standard chord progression
        chord_progression = ['Cmaj7', 'Dm7', 'G7', 'Cmaj7']
        print(f"Using chord progression: {' | '.join(chord_progression)}")
        
        # Generate bass line
        generated_bass = []
        
        # Initialize buffers with collected notes
        self.notes_buffer = collected_notes.copy()
        
        # Generate chord data based on progression
        self.chords_buffer = []
        notes_per_chord = max(1, len(collected_notes) // len(chord_progression))
        
        for chord in chord_progression:
            self.chords_buffer.extend([chord] * notes_per_chord)
        
        # Adjust lengths to ensure they match
        while len(self.chords_buffer) < len(self.notes_buffer):
            self.chords_buffer.append(chord_progression[-1])
        self.chords_buffer = self.chords_buffer[:len(self.notes_buffer)]
        
        print("\nGenerating bass line...")
        # Generate bass notes non-randomly (using argmax)
        generated_bass = []

        # Generate one note per input note
        for i in range(len(self.notes_buffer)):
            end_idx = i + 1
            melody_slice = self.notes_buffer[:end_idx]
            chord_slice = self.chords_buffer[:end_idx]
            
            bass_note = self.predict_deterministic(melody_slice, chord_slice)
            generated_bass.append(bass_note)
            print(f"Generated bass note {i+1}: {bass_note} ({note.Note(midi=bass_note).nameWithOctave})")

        # Play the result
        print("\nPlaying the generated bass line...")
        self._play_audio_simulation(generated_bass)

        print("\nDone! Press Enter to exit.")
        input()
        

    def test_without_midi(self, visualization=True, audio=False, test_tempo=120):
        """Test the jazz improvisation model without MIDI hardware"""
        print(f"Testing Jazz Improvisation Model without MIDI hardware at {test_tempo} BPM")
        self.set_tempo(test_tempo)
        self.last_note_time = time.time()  # Initialize last_note_time
        
        # Define a realistic jazz progression (ii-V-I in C)
        simulated_chords = ['Dm7', 'G7', 'Cmaj7', 'Cmaj7']
        
        # Define corresponding melodies (based on chord tones and common jazz vocabulary)
        simulated_keyboard_input = [
            # Dm7 notes (D F A C + extensions)
            62, 65, 69, 72, 74, 62, 65, 69,
            # G7 notes (G B D F + extensions)
            67, 71, 74, 65, 66, 67, 71, 74,
            # Cmaj7 notes (C E G B + extensions)
            60, 64, 67, 71, 72, 74, 76, 72,
            # Cmaj7 again with a different melodic pattern
            60, 67, 64, 72, 71, 69, 67, 64
        ]
        
        # Initialize buffers
        self.notes_buffer = []
        self.chords_buffer = []
        
        # Generate a bass line
        generated_bass_line = []
        chord_markers = []
        note_index = 0
        
        # Simulate playing through the chord progression multiple times
        for cycle in range(3):  # 3 cycles through the progression
            print(f"Cycle {cycle+1} of the chord progression:")
            
            for i, chord_name in enumerate(simulated_chords):
                print(f"  Playing: {chord_name}")
                
                # Add 8 notes per chord (typically two bars in jazz)
                for j in range(8):
                    # Add current melody note to buffer
                    current_note = simulated_keyboard_input[note_index]
                    self.notes_buffer.append(current_note)
                    note_index = (note_index + 1) % len(simulated_keyboard_input)
                    
                    # Keep buffer at appropriate size
                    if len(self.notes_buffer) > self.sequence_length:
                        self.notes_buffer = self.notes_buffer[-self.sequence_length:]
                    
                    # Add chord to buffer
                    self.chords_buffer.append(chord_name)
                    
                    # Keep chord buffer at appropriate size
                    if len(self.chords_buffer) > self.sequence_length:
                        self.chords_buffer = self.chords_buffer[-self.sequence_length:]
                    
                    # If we have enough data to predict
                    if len(self.notes_buffer) >= 8 and len(self.chords_buffer) >= 8:
                        # Generate bass note
                        bass_note = self.predict_next_note(self.notes_buffer, self.chords_buffer)
                        
                        if bass_note is not None:  # Ensure we have a valid note
                            note_name = "Unknown"
                            try:
                                note_name = note.Note(midi=bass_note).nameWithOctave
                            except:
                                pass
                            print(f"    Generated bass note: {bass_note} ({note_name})")
                            generated_bass_line.append(bass_note)
                            
                            # Add marker for chord changes
                            if j == 0:
                                chord_markers.append(len(generated_bass_line) - 1)
                
                # Short delay to simulate real-time playing
                time.sleep(0.1)  # Reduced from 0.2 to make the test run faster
        
        print("\nGenerated bass line complete!")
        
        # Ensure we have bass notes for visualization and audio
        if not generated_bass_line:
            print("No bass notes were generated. Using default bass line.")
            generated_bass_line = [40, 43, 45, 47, 48, 50, 52, 55]  # E minor scale
        
        # Visualize the result
        if visualization:
            try:
                plt.figure(figsize=(12, 6))
                plt.plot(generated_bass_line, 'b-o', linewidth=2)
                plt.title('Generated Jazz Bass Line')
                plt.ylabel('MIDI Note Value')
                plt.xlabel('Time Step')
                plt.grid(True)
                
                # Mark chord changes (only if we have valid markers)
                if chord_markers:
                    for idx in chord_markers:
                        if 0 <= idx < len(generated_bass_line):  # Ensure index is valid
                            plt.axvline(x=idx, color='r', linestyle='--', alpha=0.5)
                    
                    # Annotate with chord names
                    for i, idx in enumerate(chord_markers):
                        if 0 <= idx < len(generated_bass_line):  # Ensure index is valid
                            chord_idx = i % len(simulated_chords)
                            plt.annotate(simulated_chords[chord_idx], 
                                    (idx, max(generated_bass_line) + 2),
                                    rotation=45)
                
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Visualization error: {e}")
        
        # Play audio if requested
        if audio:
            try:
                print(f"Playing bass line with {len(generated_bass_line)} notes")
                self._play_audio_simulation(generated_bass_line)
            except Exception as e:
                print(f"Could not play audio: {e}")
                print("Audio playback requires pygame to be installed.")
    
    def set_chord_progression(self, chord_names):
        """Set current chord progression for improvisation"""
        self.current_progression = chord_names
        self.progression_start_time = time.time()
        print(f"Set chord progression: {' | '.join(chord_names)}")
    
    # Generate bass notes non-randomly (using argmax)
    def predict_deterministic(self, melody, chords):
        melody_processed = self.preprocess_midi_input(melody)
        chord_processed = self.preprocess_chord_input(chords)
        tempo_processed = np.array([[self.tempo]])
        
        prediction = self.model.predict([melody_processed, chord_processed, tempo_processed], verbose=0)[0]
        
        # Use argmax for deterministic output
        return np.argmax(prediction) + 32

    def update_chord_based_on_time(self):
        """Update current chord based on timing"""
        if not hasattr(self, 'current_progression') or not self.current_progression:
            return 'Cmaj7'  # Default chord
            
        # Calculate which chord we should be on based on time
        beats_per_measure = 4
        seconds_per_beat = 60.0 / self.tempo
        seconds_per_chord = seconds_per_beat * beats_per_measure  # Assume each chord lasts one measure
        
        elapsed_time = time.time() - self.progression_start_time
        chord_index = int(elapsed_time / seconds_per_chord) % len(self.current_progression)
        
        return self.current_progression[chord_index]

    def _play_audio_simulation(self, bass_line):
        """Play the generated bass line using pygame"""
        try:
            # Reinitialize pygame mixer
            pygame.mixer.quit()
            pygame.mixer.init(44100, -16, 2, 512)
            pygame.init()

            quarter_note_duration = int(60000 / self.tempo)
            
            # Create simple sine wave sounds
            def create_sine_wave(frequency, duration=0.25):
                sample_rate = 44100
                t = np.linspace(0, duration, int(sample_rate * duration), False)
                
                # Create a more bass-like tone with harmonics
                wave = 0.7 * np.sin(frequency * 2 * np.pi * t)  # Fundamental
                wave += 0.2 * np.sin(2 * frequency * 2 * np.pi * t)  # 1st harmonic
                wave += 0.1 * np.sin(3 * frequency * 2 * np.pi * t)  # 2nd harmonic
                
                # Apply envelope
                envelope = np.ones_like(wave)
                attack = int(0.01 * sample_rate)
                release = int(0.15 * sample_rate)
                envelope[:attack] = np.linspace(0, 1, attack)
                envelope[-release:] = np.linspace(1, 0, release)
                wave = wave * envelope
                
                # Convert to 16-bit integers and make stereo
                stereo_wave = np.column_stack((wave, wave))
                stereo_wave = (stereo_wave * 32767).astype(np.int16)
                return pygame.sndarray.make_sound(stereo_wave)
            
            # MIDI note to frequency conversion
            def midi_to_freq(midi_note):
                return 440 * (2 ** ((midi_note - 69) / 12))
            
            # Create sound cache
            sound_cache = {}
            
            print(f"Playing bass line at {self.tempo} BPM...")
            for note_value in bass_line:
                if note_value is None:
                    print("Skipping None note")
                    continue
                    
                freq = midi_to_freq(note_value)
                print(f"Playing note {note_value} at frequency {freq:.2f}Hz")
                
                # Check if we already have this note in cache
                if note_value not in sound_cache:
                    sound_cache[note_value] = create_sine_wave(freq)
                
                # Play the note and wait
                sound_cache[note_value].play()
                pygame.time.wait(quarter_note_duration)
            
            # Wait for last note to finish
            pygame.time.wait(300)
        except Exception as e:
            print(f"Audio playback error: {e}")
        finally:
            # Make sure to clean up pygame
            try:
                pygame.mixer.quit()
                pygame.quit()
            except:
                pass




# Improved training data generator with musical structure
def generate_musical_training_data(num_samples=1000):
    """Generate musically structured training data for the model"""
    # Define common jazz keys, chord progressions, and scales
    keys = ['C', 'F', 'Bb', 'Eb', 'G', 'D', 'A']
    
    # Common jazz chord progressions in roman numerals
    progressions = [
        ['I', 'vi', 'ii', 'V'],      # I-vi-ii-V
        ['ii', 'V', 'I', 'I'],       # ii-V-I-I
        ['I', 'IV', 'iii', 'vi'],    # I-IV-iii-vi
        ['ii', 'V', 'iii', 'vi'],    # ii-V-iii-vi (changed from VI to vi)
        ['I', 'vi', 'IV', 'V']       # I-vi-IV-V
    ]
    
    # Chord types
    chord_types = {
        'I': ['maj7', 'maj9', '6', 'maj7#11'],
        'ii': ['m7', 'm9', 'm11'],
        'iii': ['m7', 'm9'],
        'IV': ['maj7', 'maj9', '6'],
        'V': ['7', '9', '13', '7b9'],
        'vi': ['m7', 'm9'],
        'VII': ['m7b5', 'dim7']
    }
    
    # Roman numeral to scale degree mapping
    roman_to_degree = {
        'I': 0, 'II': 2, 'III': 4, 'IV': 5, 'V': 7, 'VI': 9, 'VII': 11,
        'i': 0, 'ii': 2, 'iii': 4, 'iv': 5, 'v': 7, 'vi': 9, 'vii': 11
    }
    
    # Initialize arrays
    melody_data = np.zeros((num_samples, 32, 128))
    chord_data = np.zeros((num_samples, 32, 24))
    target_data = np.zeros((num_samples, 29))  # Bass range 32-60
    tempo_data = np.zeros((num_samples, 1))
    
    for i in range(num_samples):
        # Select a random key
        key = random.choice(keys)
        key_obj = scale.MajorScale(key)
        key_pitches = [p.midi % 12 for p in key_obj.getPitches()]
        
        # Select a random progression
        progression = random.choice(progressions)
        
        # Build chord sequence with variations
        chord_sequence = []
        for roman in progression:
            # Get the root note
            root_degree = roman_to_degree[roman]
            # Convert degree to actual note in the key
            root_note = (key_obj.tonic.midi % 12 + root_degree) % 12
            
            # Choose a chord type
            chord_type = random.choice(chord_types[roman])

            tempo = random.randint(60, 240)
            tempo_data[i, 0] = tempo
            
            # Build the chord
            if chord_type == 'maj7':
                chord_pitches = [root_note, (root_note + 4) % 12, (root_note + 7) % 12, (root_note + 11) % 12]
            elif chord_type == 'm7':
                chord_pitches = [root_note, (root_note + 3) % 12, (root_note + 7) % 12, (root_note + 10) % 12]
            elif chord_type == '7':
                chord_pitches = [root_note, (root_note + 4) % 12, (root_note + 7) % 12, (root_note + 10) % 12]
            else:
                # Default to triad if other types not specifically defined
                if 'm' in chord_type:
                    chord_pitches = [root_note, (root_note + 3) % 12, (root_note + 7) % 12]
                else:
                    chord_pitches = [root_note, (root_note + 4) % 12, (root_note + 7) % 12]
            
            # Add to sequence - repeat each chord 8 times (common in jazz, 8 eighth notes per bar)
            chord_sequence.extend([{'root': root_note, 'pitches': chord_pitches}] * 8)
        
        # Generate melody using chord tones and key scale with embellishments
        melody_sequence = []
        for chord_info in chord_sequence:
            # 80% chance of chord tone, 20% chance of scale tone
            if random.random() < 0.8:
                note_options = chord_info['pitches']
            else:
                note_options = key_pitches
            
            # Choose a note
            melody_note = random.choice(note_options)
            
            # Convert to actual MIDI note in an appropriate register (middle register)
            octave = random.choice([4, 5])  # Middle C is C4 (MIDI 60)
            melody_note = melody_note + (octave * 12)
            
            melody_sequence.append(melody_note)
        
        # Generate target bass notes (focus on roots and fifths with walking patterns)
        target_index = random.randint(0, len(chord_sequence) - 1)
        target_chord = chord_sequence[target_index]
        
        # Bass notes are typically lower
        bass_octave = random.choice([2, 3])  # E2 is MIDI 40
        
        # Choose from common bass patterns
        if random.random() < 0.5:
            # Root is most common
            target_note = target_chord['root'] + (bass_octave * 12)
        elif random.random() < 0.8:
            # Fifth is next most common
            target_note = (target_chord['root'] + 7) % 12 + (bass_octave * 12)
        else:
            # Sometimes third or seventh
            offset = random.choice([3, 4, 10, 11])  # Minor 3rd, Major 3rd, Minor 7th, Major 7th
            target_note = (target_chord['root'] + offset) % 12 + (bass_octave * 12)
        
        # Ensure the note is in our target range (32-60)
        while target_note < 32:
            target_note += 12
        while target_note > 60:
            target_note -= 12
        
        # Fill in the data arrays
        for j in range(min(32, len(melody_sequence))):
            melody_note = melody_sequence[j]
            if 0 <= melody_note < 128:
                melody_data[i, j, melody_note] = 1
            
            chord_info = chord_sequence[j]
            # Set root note
            chord_data[i, j, chord_info['root']] = 1
            
            # Set chord type (simplified)
            if len(chord_info['pitches']) >= 4:
                if (chord_info['root'] + 4) % 12 in chord_info['pitches']:  # Major third
                    if (chord_info['root'] + 10) % 12 in chord_info['pitches']:  # Minor seventh
                        chord_data[i, j, 12] = 1  # Dominant 7th
                    elif (chord_info['root'] + 11) % 12 in chord_info['pitches']:  # Major seventh
                        chord_data[i, j, 16] = 1  # Major 7th
                elif (chord_info['root'] + 3) % 12 in chord_info['pitches']:  # Minor third
                    if (chord_info['root'] + 10) % 12 in chord_info['pitches']:  # Minor seventh
                        chord_data[i, j, 15] = 1  # Minor 7th
            else:
                # Triads
                if (chord_info['root'] + 4) % 12 in chord_info['pitches']:  # Major third
                    chord_data[i, j, 14] = 1  # Major triad
                elif (chord_info['root'] + 3) % 12 in chord_info['pitches']:  # Minor third
                    chord_data[i, j, 13] = 1  # Minor triad
        
        # Set the target (adjusted to bass range 32-60)
        target_adjusted = target_note - 32  # Adjust to 0-indexed range
        if 0 <= target_adjusted < 29:
            target_data[i, target_adjusted] = 1
    
    # Split into training and test sets
    X_melody_train, X_melody_test, X_chord_train, X_chord_test, X_tempo_train, X_tempo_test, y_train, y_test = train_test_split(
        melody_data, chord_data, tempo_data, target_data, test_size=0.2, random_state=42
    )
    
    print(f"Generated {num_samples} training samples with musical structure")
    print(f"Training set: {len(X_melody_train)} samples")
    print(f"Test set: {len(X_melody_test)} samples")
    
    return X_melody_train, X_chord_train, X_tempo_train, y_train, X_melody_test, X_chord_test, X_tempo_test, y_test
    
    # Generate one note per input note
    for i in range(len(collected_notes)):
        end_idx = i + 1
        melody_slice = self.notes_buffer[:end_idx]
        chord_slice = self.chords_buffer[:end_idx]
        
        bass_note = predict_deterministic(melody_slice, chord_slice)
        generated_bass.append(bass_note)
        print(f"Generated bass note {i+1}: {bass_note} ({note.Note(midi=bass_note).nameWithOctave})")
    
    # Play the result
    print("\nPlaying the generated bass line...")
    self._play_audio_simulation(generated_bass)
    
    print("\nDone! Press Enter to exit.")
    input()

def interactive_testing():
    print("Starting interactive testing environment...")
    
    # Create a new model instead of loading one
    model = JazzImprovisationModel()
    
    # Skip loading the saved model entirely to avoid Lambda layer issues
    print("Using a new model without loading saved weights (for testing purposes)")
    
    print("Testing MIDI connection...")
    midi_success = model.setup_midi_input("any")
    
    if not midi_success:
        print("Failed to connect to MIDI. Running in test mode instead...")
        model.test_without_midi(audio=True)
        return
    
    print("\nChoose testing mode:")
    print("1. Interactive real-time (press keys, get immediate responses)")
    print("2. Collection mode (play a sequence, then get a complete bass line)")
    
    try:
        choice = int(input("Enter your choice (1 or 2): "))
        
        if choice == 1:
            # Original interactive mode
            model.set_chord_progression(['Cmaj7', 'Dm7', 'G7', 'Cmaj7'])
            model.set_tempo(120)
            model.interactive_test()
        elif choice == 2:
            # New collection mode
            model.interactive_collection_mode()
        else:
            print("Invalid choice. Running in test mode instead...")
            model.test_without_midi(audio=True)
    except ValueError:
        print("Invalid input. Running in test mode instead...")
        model.test_without_midi(audio=True)

# Main function to run the test
def main():
    print("Jazz Improvisation Model Test - Improved Version")
    print("==============================================")

    # Check if TensorFlow is using GPU
    if tf.config.list_physical_devices('GPU'):
        print("GPU is available and will be used for training")
    else:
        print("No GPU found, training will use CPU (may be slower)")
    
    # Create the model
    jazz_model = JazzImprovisationModel()
    
    

    # Generate musically structured training data
    print("\nGenerating musically structured training data...")
    X_melody_train, X_chord_train, X_tempo_train, y_train, X_melody_test, X_chord_test, X_tempo_test, y_test = generate_musical_training_data(1000)
    
    # Train the model with a small number of epochs for testing
    print("\nTraining model with training data...")
    history = jazz_model.train(
        X_melody_train, X_tempo_train, X_chord_train, y_train, 
        epochs=10,  # Reduced for faster testing
        batch_size=64
    )
    
    # Evaluate on test data
    print("\nEvaluating model on test data...")
    test_results = jazz_model.model.evaluate(
        [X_melody_test, X_chord_test, X_tempo_test], 
        y_test, 
        verbose=1
    )
    print(f"Test Loss: {test_results[0]:.4f}")
    print(f"Test Accuracy: {test_results[1]:.4f}")
    
    # Test the model - set audio to False if you're having audio issues
    print("\nStarting test simulation...")
    jazz_model.test_without_midi(visualization=True, audio=True, test_tempo=120)
    
    # Save the model with the .keras extension as recommended
    model_path = 'improved_jazz_model.keras'
    try:
        jazz_model.model.save(model_path)
        print(f"\nModel saved as '{model_path}'")
    except Exception as e:
        print(f"Error saving model with .keras extension: {e}")
        # Fall back to HDF5 format
        jazz_model.model.save('improved_jazz_model.h5')
        print("\nModel saved as 'improved_jazz_model.h5'")
    
    print("\nTest complete!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_testing()
    else:
        main()