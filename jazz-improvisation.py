import numpy as np
import tensorflow as tf
from tensorflow import keras
from music21 import chord, note, stream
import mido
import threading
import time

class JazzImprovisationModel:
    def __init__(self, model_path=None):
        """Initialize the jazz improvisation model"""
        self.sequence_length = 32  # Number of notes to consider for prediction
        self.is_playing = False
        self.notes_buffer = []
        self.chords_buffer = []
        
        if model_path:
            self.model = keras.models.load_model(model_path)
        else:
            self.build_model()
            
        # Scale degrees and jazz-specific note mappings
        self.scale_degrees = {
            'C': [0, 2, 4, 5, 7, 9, 11],  # Major scale
            'C7': [0, 4, 7, 10],          # Dominant 7th
            'Cm7': [0, 3, 7, 10],         # Minor 7th
            'Cmaj7': [0, 4, 7, 11],       # Major 7th
            'C7b9': [0, 4, 7, 10, 13],    # Dominant 7th flat 9
            'C7#11': [0, 4, 7, 10, 18]    # Dominant 7th sharp 11
        }
        
        self.jazz_patterns = {
            'ii-V-I': [2, 5, 1],
            'I-vi-ii-V': [1, 6, 2, 5],
            'blues': [1, 4, 1, 5, 4, 1]
        }
    
    def build_model(self):
        """Build a deep learning model for jazz improvisation"""
        melody_input = keras.layers.Input(shape=(self.sequence_length, 128))
        chord_input = keras.layers.Input(shape=(self.sequence_length, 24))
        
        x1 = keras.layers.LSTM(256, return_sequences=True)(melody_input)
        x1 = keras.layers.Dropout(0.3)(x1)
        x1 = keras.layers.LSTM(256)(x1)
        
        x2 = keras.layers.LSTM(128, return_sequences=True)(chord_input)
        x2 = keras.layers.Dropout(0.3)(x2)
        x2 = keras.layers.LSTM(128)(x2)
        
        combined = keras.layers.Concatenate()([x1, x2])
        
        x = keras.layers.Dense(512, activation='relu')(combined)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        
        # Output layer (128 = MIDI note range)
        output = keras.layers.Dense(128, activation='softmax')(x)
        
        # Create model
        self.model = keras.Model(inputs=[melody_input, chord_input], outputs=output)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
    
    def train(self, melody_data, chord_data, note_targets, epochs=50, batch_size=64):
        """Train the model with jazz music data"""
        self.model.fit(
            [melody_data, chord_data], 
            note_targets,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2
        )
    
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
    
    def predict_next_note(self, melody_input, chord_input):
        """Predict the next note based on melody and chord context"""
        melody_processed = self.preprocess_midi_input(melody_input)
        chord_processed = self.preprocess_chord_input(chord_input)
        
        # Get model prediction
        prediction = self.model.predict([melody_processed, chord_processed])[0]
        
        # Apply some temperature to add randomness
        temperature = 1.2  # Higher values increase randomness
        prediction = np.log(prediction + 1e-10) / temperature
        prediction = np.exp(prediction) / np.sum(np.exp(prediction))
        
        # Sample from the probability distribution
        predicted_note = np.random.choice(range(128), p=prediction)
        
        return predicted_note
    
    def add_jazz_feel(self, note_sequence, swing_ratio=0.67):
        """Add swing feel to the generated notes"""
        # Apply swing rhythm (long-short pattern)
        swung_sequence = []
        for i, note_val in enumerate(note_sequence):
            if i % 2 == 0:  # Notes on the beat
                duration = swing_ratio
            else:  # Notes off the beat
                duration = 1 - swing_ratio
            swung_sequence.append((note_val, duration))
        return swung_sequence
    
    def generate_bass_line(self, chord_progression, rhythm_pattern="walking"):
        """Generate a jazz bass line based on chord progression"""
        bass_line = []
        
        for chord_name in chord_progression:
            # Parse the chord
            try:
                c = chord.Chord(chord_name)
                root = c.root().midi
                
                if rhythm_pattern == "walking":
                    # Walking bass pattern: root, 5th, approach notes
                    bass_line.append(root)
                    bass_line.append(root + 7)  # 5th
                    bass_line.append(root + 4)  # 3rd or approach note
                    
                    # Add a leading tone to the next chord
                    next_chord_idx = chord_progression.index(chord_name) + 1
                    if next_chord_idx < len(chord_progression):
                        next_chord = chord.Chord(chord_progression[next_chord_idx])
                        next_root = next_chord.root().midi
                        
                        # Choose an approach note (half step below or whole step above)
                        if abs(next_root - root) > 2:
                            if next_root > root:
                                bass_line.append(next_root - 1)  # Half step below
                            else:
                                bass_line.append(next_root + 2)  # Whole step above
                        else:
                            bass_line.append(root - 2)  # Another approach option
                    else:
                        bass_line.append(root - 2)
                
                elif rhythm_pattern == "two_feel":
                    # Simpler two-feel rhythm
                    bass_line.append(root)
                    bass_line.append(root + 7)  # 5th
            
            except:
                # Default pattern if chord parsing fails
                bass_line.append(40)  # Default low E
                bass_line.append(47)  # Default B
        
        return bass_line
    
    def start_improvisation(self, input_port_name, output_port_name):
        """Start real-time jazz improvisation from MIDI input"""
        self.is_playing = True
        threading.Thread(target=self._improvisation_loop, 
                         args=(input_port_name, output_port_name)).start()
    
    def stop_improvisation(self):
        """Stop real-time improvisation"""
        self.is_playing = False
    
    def _improvisation_loop(self, input_port_name, output_port_name):
        """Main improvisation loop listening to MIDI input and generating output"""
        try:
            # Set up MIDI input and output
            inport = mido.open_input(input_port_name)
            outport = mido.open_output(output_port_name)
            
            # Initialize buffers
            current_notes = []
            current_chords = []
            last_prediction_time = time.time()
            
            # Main loop
            while self.is_playing:
                # Process incoming MIDI messages
                for msg in inport.iter_pending():
                    if msg.type == 'note_on' and msg.velocity > 0:
                        current_notes.append(msg.note)
                        self.notes_buffer.append(msg.note)
                        
                        # Detect chords from incoming notes
                        if len(current_notes) >= 3:
                            try:
                                c = chord.Chord([note.Note(midi=n) for n in current_notes])
                                chord_name = c.commonName
                                current_chords.append(chord_name)
                                self.chords_buffer.append(chord_name)
                                current_notes = []
                            except:
                                pass
                    
                    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                        if msg.note in current_notes:
                            current_notes.remove(msg.note)
                
                current_time = time.time()
                if current_time - last_prediction_time >= 0.25 and self.notes_buffer and self.chords_buffer:
                    last_prediction_time = current_time
                    
                    # Keep buffer size manageable
                    if len(self.notes_buffer) > self.sequence_length * 2:
                        self.notes_buffer = self.notes_buffer[-self.sequence_length:]
                    if len(self.chords_buffer) > self.sequence_length * 2:
                        self.chords_buffer = self.chords_buffer[-self.sequence_length:]
                    
                    # Generate bass note
                    bass_note = self.predict_next_note(self.notes_buffer, self.chords_buffer)
                    
                    while bass_note > 60:  # Keep it in bass range
                        bass_note -= 12
                    
                    duration = 0.25  # Quarter note duration
                    if len(self.notes_buffer) % 2 == 0:
                        duration *= 0.67  # Swing ratio
                    else:
                        duration *= 0.33
                    
                    # Send bass note to MIDI output
                    outport.send(mido.Message('note_on', note=bass_note, velocity=80, channel=1))
                    time.sleep(duration)
                    outport.send(mido.Message('note_off', note=bass_note, velocity=0, channel=1))
                
                time.sleep(0.01)  # Small sleep to prevent CPU overload
        
        except Exception as e:
            print(f"Improvisation error: {e}")
            self.is_playing = False

# Example usage
def main():
    # Create the model
    jazz_model = JazzImprovisationModel()
    

    example_melodies = np.random.rand(100, 32, 128)
    example_chords = np.random.rand(100, 32, 24)
    example_targets = np.random.rand(100, 128)
    jazz_model.train(example_melodies, example_chords, example_targets, epochs=5)
    
    # For demo, just generate a bassline from a chord progression
    chord_prog = ["Cmaj7", "Am7", "Dm7", "G7"]
    bassline = jazz_model.generate_bass_line(chord_prog)
    print("Generated bassline:", bassline)
    
    print("Available MIDI input ports:", mido.get_input_names())
    print("Available MIDI output ports:", mido.get_output_names())


if __name__ == "__main__":
    main()