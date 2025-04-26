import numpy as np
import tensorflow as tf
from tensorflow import keras
from music21 import chord, note, stream, scale, interval, pitch
import time
import matplotlib.pyplot as plt
import pygame
import os
import random
from collections import deque # For replay buffer
import sys

# --- Constants ---
SEQUENCE_LENGTH = 16 # Shorter sequence for RL state might be more manageable
BASS_NOTE_MIN = 32  # E1
BASS_NOTE_MAX = 60  # C4
NUM_BASS_NOTES = BASS_NOTE_MAX - BASS_NOTE_MIN + 1
STATE_SHAPE_MELODY = (SEQUENCE_LENGTH, 128) # Melody history
STATE_SHAPE_CHORD = (SEQUENCE_LENGTH, 24)   # Chord history
STATE_SHAPE_BASS = (SEQUENCE_LENGTH, 1)     # Bass history (MIDI values)
STATE_SHAPE_CONTEXT = (3,) # Tempo, Current Chord Root, Current Chord Type Simplified


class JazzEnvironment:
    """
    Simulates the musical environment for the RL agent.
    Provides states and rewards.
    """
    def __init__(self, chord_progression, melody_sequence, tempo=120):
        self.chord_progression = chord_progression # e.g., ['Cmaj7', 'Dm7', 'G7', 'Cmaj7']
        self.melody_sequence = melody_sequence # A list of MIDI notes (can be longer than progression)
        self.tempo = tempo
        self.beats_per_chord = 4 # Assume 4 beats per chord for simplicity

        self.current_step = 0
        self.current_chord_index = 0
        self.current_melody_index = 0
        self.max_steps = len(melody_sequence) # Or define a fixed episode length

        # State components
        self.melody_history = deque(maxlen=SEQUENCE_LENGTH)
        self.chord_history = deque(maxlen=SEQUENCE_LENGTH) # Store chord names or simplified encodings
        self.bass_note_history = deque(maxlen=SEQUENCE_LENGTH)

        # Initialize histories with padding (e.g., zeros or rests)
        for _ in range(SEQUENCE_LENGTH):
            self.melody_history.append(-1) # Use -1 for rests/padding
            self.chord_history.append(None)
            self.bass_note_history.append(-1) # Use -1 for rests/padding

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

        # Note: Bass history is updated *after* the agent takes an action in step()

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
                    chord_state[i, root] = 1 # One-hot encode root

                    # Simplified chord type encoding (example)
                    if c.isDominantSeventh(): type_idx = 12
                    elif c.isMinorSeventh(): type_idx = 13
                    elif c.isMajorSeventh(): type_idx = 14
                    elif c.isMinorTriad(): type_idx = 15
                    elif c.isMajorTriad(): type_idx = 16
                    else: type_idx = 17 # Other/Unknown
                    chord_state[i, type_idx] = 1
                except Exception:
                    chord_state[i, 0] = 1 # Default to C if parsing fails
                    chord_state[i, 16] = 1 # Default to Major

        # --- Bass History ---
        # Normalize bass notes (e.g., to 0-1) or keep as MIDI values
        bass_state = np.array(list(self.bass_note_history)).reshape(STATE_SHAPE_BASS)
        bass_state = (bass_state - BASS_NOTE_MIN) / NUM_BASS_NOTES # Normalize 0-1
        bass_state[bass_state < 0] = -1 # Keep rests/padding as -1


        # --- Context ---
        tempo_norm = self.tempo / 240.0 # Normalize tempo
        current_chord_name = self.chord_history[-1]
        root_norm = 0.0
        type_norm = 0.0 # 0: Major, 0.33: Minor, 0.66: Dominant, 1: Other
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

        # Return state components (can be structured as a dictionary or tuple)
        # Using dict for clarity
        return {
            "melody": melody_state.reshape(1, *STATE_SHAPE_MELODY),
            "chord": chord_state.reshape(1, *STATE_SHAPE_CHORD),
            "bass": bass_state.reshape(1, *STATE_SHAPE_BASS),
            "context": context_state.reshape(1, *STATE_SHAPE_CONTEXT)
        }


    def _calculate_reward(self, bass_action):
        """
        Calculate reward with additional factors to encourage variation and musicality.
        """
        reward = 0.0
        bass_note_midi = bass_action + BASS_NOTE_MIN  # Convert action index back to MIDI
        
        current_chord_name = self.chord_history[-1]
        
        # Get the last few bass notes to check for repetition
        recent_bass_notes = [n for n in list(self.bass_note_history) if n >= 0]
        
        if not current_chord_name:
            return 0.0  # No chord context, neutral reward
        
        try:
            c = chord.Chord(current_chord_name)
            chord_pitches = c.pitches
            chord_tone_classes = {p.pitchClass for p in chord_pitches}  # Pitch classes (0-11)
            root_pc = c.root().pitchClass
            bass_note_obj = note.Note(midi=bass_note_midi)
            bass_pc = bass_note_obj.pitch.pitchClass
            
            # 1. Chord Tone Bonus: Is the note part of the current chord?
            if bass_pc in chord_tone_classes:
                reward += 0.5
                # Bonus for being the root
                if bass_pc == root_pc:
                    reward += 0.3
                # Bonus for being 3rd or 5th 
                elif bass_pc == (root_pc + 4) % 12 or bass_pc == (root_pc + 3) % 12 or bass_pc == (root_pc + 7) % 12:
                    reward += 0.1
                    
            # 2. Scale Tone Bonus: Check if note is in a scale appropriate for the chord
            # For major chords: major scale, for minor: minor scale, for dominant: mixolydian
            is_scale_tone = False
            if "maj" in current_chord_name.lower():
                # Major scale
                major_scale_pcs = [(root_pc + i) % 12 for i in [0, 2, 4, 5, 7, 9, 11]]
                if bass_pc in major_scale_pcs:
                    is_scale_tone = True
                    reward += 0.2
            elif "m" in current_chord_name.lower() and "maj" not in current_chord_name.lower():
                # Minor scale
                minor_scale_pcs = [(root_pc + i) % 12 for i in [0, 2, 3, 5, 7, 8, 10]]
                if bass_pc in minor_scale_pcs:
                    is_scale_tone = True
                    reward += 0.2
            elif "7" in current_chord_name and "maj7" not in current_chord_name:
                # Mixolydian for dominant 7 chords
                mixolydian_pcs = [(root_pc + i) % 12 for i in [0, 2, 4, 5, 7, 9, 10]]
                if bass_pc in mixolydian_pcs:
                    is_scale_tone = True
                    reward += 0.2
                    
            # Penalize non-chord, non-scale tones more significantly
            if bass_pc not in chord_tone_classes and not is_scale_tone:
                reward -= 0.4
                    
            # 3. Voice Leading Reward: Smooth transition from previous note?
            if len(recent_bass_notes) > 0:
                prev_bass_note_midi = recent_bass_notes[-1]
                if prev_bass_note_midi > 0:
                    prev_bass_note_obj = note.Note(midi=prev_bass_note_midi)
                    inter = interval.Interval(prev_bass_note_obj, bass_note_obj)
                    
                    # Penalize large leaps (e.g., > octave)
                    if abs(inter.semitones) > 12:
                        reward -= 0.3
                    # Reward steps (major/minor 2nd)
                    elif 1 <= abs(inter.semitones) <= 2:
                        reward += 0.2
                    # Small reward for consonant leaps (3rd, 4th, 5th, 6th)
                    elif 3 <= abs(inter.semitones) <= 7:
                        reward += 0.1
                    # Smaller reward for larger consonant leaps
                    elif 7 < abs(inter.semitones) <= 12:
                        reward += 0.05
                    
            # 4. Dissonance Penalty: Does it clash badly?
            # Check for half-step clashes with primary chord tones
            clash_penalty = 0
            for p in chord_pitches:
                # Check against root, 3rd, 5th primarily
                if p.pitchClass in [root_pc, (root_pc + 3) % 12, (root_pc + 4) % 12, (root_pc + 7) % 12]:
                    diff = abs(bass_pc - p.pitchClass)
                    if diff == 1 or diff == 11:  # Semitone clash
                        clash_penalty = -0.6
                        break
            reward += clash_penalty
            
            # 5. NEW: Anti-repetition reward to avoid getting stuck on one note
            if len(recent_bass_notes) >= 3:
                # Check if the last 3 notes were all the same
                if recent_bass_notes[-1] == recent_bass_notes[-2] == recent_bass_notes[-3]:
                    # Strong penalty for playing the same note 3+ times in a row
                    reward -= 0.7
                
                # Check for lack of variety in a longer window
                if len(recent_bass_notes) >= 8:
                    unique_notes = len(set(recent_bass_notes[-8:]))
                    if unique_notes < 3:  # Less than 3 unique notes in the last 8 notes
                        reward -= 0.4
                    elif unique_notes >= 4:  # Good variety
                        reward += 0.3
                        
            # 6. NEW: Rhythmic placement reward
            beat_in_measure = (self.current_step % self.beats_per_chord)
            # Root on beat 1 is good
            if beat_in_measure == 0 and bass_pc == root_pc:
                reward += 0.2
            # 5th on beat 3 is good in a walking bass pattern
            elif beat_in_measure == 2 and bass_pc == (root_pc + 7) % 12:
                reward += 0.1
                
        except Exception as e:
            pass  # Return default reward if chord parsing fails
        
        # Normalize reward to a range, e.g., [-1, 1]
        reward = np.clip(reward, -1.0, 1.0)
        return reward


    def step(self, action):
        """
        Apply the agent's action, update state, calculate reward, and return results.
        Action is the index (0 to NUM_BASS_NOTES-1).
        """
        if not (0 <= action < NUM_BASS_NOTES):
            raise ValueError(f"Invalid action: {action}")

        # Calculate reward based on the *current* state *before* updating history
        reward = self._calculate_reward(action)

        # Update bass note history with the chosen action (converted to MIDI)
        bass_note_midi = action + BASS_NOTE_MIN
        self.bass_note_history.append(bass_note_midi)

        # Advance time
        self.current_step += 1
        self.current_melody_index += 1

        # Update melody and chord histories for the *next* state
        self._update_history()

        # Check if done
        done = self.current_step >= self.max_steps

        # Get the next state
        next_state = self._get_state()

        return next_state, reward, done, {} # {} is placeholder for info dict


class RLJazzAgent:
    """
    The Reinforcement Learning Agent that learns to improvise bass lines.
    Uses a Deep Q-Network (DQN).
    """
    def __init__(self, state_shapes, num_actions, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_shapes = state_shapes # Dict of shapes for melody, chord, bass, context
        self.num_actions = num_actions   # Number of possible bass notes
        self.memory = deque(maxlen=2000) # Replay buffer
        self.gamma = gamma               # Discount factor
        self.epsilon = epsilon           # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        """Build the Q-Network model."""
        # Inputs for each part of the state
        melody_input = keras.layers.Input(shape=self.state_shapes["melody"][1:], name="melody_input") # Shape needs to exclude batch dim
        chord_input = keras.layers.Input(shape=self.state_shapes["chord"][1:], name="chord_input")
        bass_input = keras.layers.Input(shape=self.state_shapes["bass"][1:], name="bass_input")
        context_input = keras.layers.Input(shape=self.state_shapes["context"][1:], name="context_input")

        # Process melody (similar to original model, maybe simpler)
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
        print("RL Model Summary:")
        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        """Choose action using epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions) # Explore
        # Exploit: Predict Q-values and choose the best action
        # Ensure state components are correctly shaped (add batch dimension if needed)
        act_values = self.model.predict([state["melody"], state["chord"], state["bass"], state["context"]], verbose=0)
        return np.argmax(act_values[0]) # Returns action index

    def replay(self, batch_size):
        """Train the network using experiences from memory."""
        if len(self.memory) < batch_size:
            return # Not enough memory yet

        minibatch = random.sample(self.memory, batch_size)

        # Prepare batch data (more complex due to multiple inputs)
        states_melody = np.vstack([t[0]["melody"] for t in minibatch])
        states_chord = np.vstack([t[0]["chord"] for t in minibatch])
        states_bass = np.vstack([t[0]["bass"] for t in minibatch])
        states_context = np.vstack([t[0]["context"] for t in minibatch])

        next_states_melody = np.vstack([t[3]["melody"] for t in minibatch])
        next_states_chord = np.vstack([t[3]["chord"] for t in minibatch])
        next_states_bass = np.vstack([t[3]["bass"] for t in minibatch])
        next_states_context = np.vstack([t[3]["context"] for t in minibatch])

        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])

        # Predict Q-values for current states and next states
        current_q_values = self.model.predict([states_melody, states_chord, states_bass, states_context], verbose=0)
        next_q_values = self.model.predict([next_states_melody, next_states_chord, next_states_bass, next_states_context], verbose=0)

        # Calculate target Q-values using the Bellman equation
        targets = current_q_values.copy()
        for i in range(batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.amax(next_q_values[i])

        # Train the model
        self.model.fit([states_melody, states_chord, states_bass, states_context], targets, epochs=1, verbose=0, batch_size=batch_size)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)
        # Potentially load epsilon value too if saved

    def save(self, name):
        self.model.save_weights(name)
        # Potentially save epsilon value

class ImprovedRLJazzAgent(RLJazzAgent):
    """
    Enhanced version of the RLJazzAgent with better exploration strategies
    and architectural improvements.
    """
    def __init__(self, state_shapes, num_actions, learning_rate=0.001, gamma=0.95, 
                 epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        super().__init__(state_shapes, num_actions, learning_rate, gamma, 
                        epsilon, epsilon_min, epsilon_decay)
        # Override the model with an improved version
        self.model = self._build_improved_model()
        # Temperature parameter for softmax exploration
        self.temperature = 1.0
        self.temperature_min = 0.2
        self.temperature_decay = 0.995
        
    def _build_improved_model(self):
        """Build an improved Q-Network model with deeper architecture."""
        # Inputs for each part of the state
        melody_input = keras.layers.Input(shape=self.state_shapes["melody"][1:], name="melody_input")
        chord_input = keras.layers.Input(shape=self.state_shapes["chord"][1:], name="chord_input")
        bass_input = keras.layers.Input(shape=self.state_shapes["bass"][1:], name="bass_input")
        context_input = keras.layers.Input(shape=self.state_shapes["context"][1:], name="context_input")
        
        # Process melody with bidirectional LSTM for better sequence understanding
        m = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(melody_input)
        m = keras.layers.Bidirectional(keras.layers.LSTM(32))(m)
        m = keras.layers.LayerNormalization()(m)
        
        # Process chords
        c = keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True))(chord_input)
        c = keras.layers.Bidirectional(keras.layers.LSTM(16))(c)
        c = keras.layers.LayerNormalization()(c)
        
        # Process bass history with attention mechanism
        b = keras.layers.Bidirectional(keras.layers.LSTM(24, return_sequences=True))(bass_input)
        # Simple attention mechanism
        b_attention = keras.layers.Dense(1)(b)
        b_attention = keras.layers.Reshape((-1,))(b_attention)
        b_attention = keras.layers.Activation('softmax')(b_attention)
        b_attention = keras.layers.Reshape((-1, 1))(b_attention)
        b = keras.layers.Multiply()([b, b_attention])
        # Replace the problematic line with Lambda layer for sum pooling
        b = keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(b)
        b = keras.layers.LayerNormalization()(b)
        
        # Process context with expanded network
        ctx = keras.layers.Dense(24, activation='relu')(context_input)
        ctx = keras.layers.Dense(16, activation='relu')(ctx)
        
        # Combine features with additional layer
        combined = keras.layers.Concatenate()([m, c, b, ctx])
        
        # Deeper network with residual connections
        x = keras.layers.Dense(128, activation='relu')(combined)
        x = keras.layers.Dropout(0.25)(x)
        
        # Residual block 1
        x_res = keras.layers.Dense(128, activation='relu')(x)
        x_res = keras.layers.Dropout(0.25)(x_res)
        x_res = keras.layers.Dense(128, activation='relu')(x_res)
        x = keras.layers.Add()([x, x_res])
        x = keras.layers.LayerNormalization()(x)
        
        # Second dense layer
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        
        # Output advantage and value streams (Dueling DQN architecture)
        advantage_stream = keras.layers.Dense(64, activation='relu')(x)
        advantage_stream = keras.layers.Dense(self.num_actions)(advantage_stream)
        
        value_stream = keras.layers.Dense(64, activation='relu')(x)
        value_stream = keras.layers.Dense(1)(value_stream)
        
        # Combine streams to get Q-values
        q_values = keras.layers.Add()([
            value_stream,
            keras.layers.Subtract()([
                advantage_stream,
                keras.layers.Lambda(lambda x: keras.backend.mean(x, axis=1, keepdims=True))(advantage_stream)
            ])
        ])
        
        model = keras.Model(
            inputs=[melody_input, chord_input, bass_input, context_input], 
            outputs=q_values
        )
        model.compile(
            loss='mse', 
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate)
        )
        print("Improved RL Model Summary:")
        model.summary()
        return model
    
    def choose_action(self, state):
        """
        Choose action using a combination of epsilon-greedy and softmax exploration.
        This gives better exploration of the action space.
        """
        if np.random.rand() <= self.epsilon:
            # Random exploration
            return random.randrange(self.num_actions)
        
        # Get Q-values for all actions
        q_values = self.model.predict([
            state["melody"], state["chord"], 
            state["bass"], state["context"]], 
            verbose=0
        )[0]
        
        # Use softmax with temperature for probabilistic selection
        # This allows exploration proportional to the estimated values
        if np.random.rand() <= 0.3:  # 30% chance to use softmax exploration
            # Apply temperature scaling
            scaled_q_values = q_values / self.temperature
            # Softmax calculation
            exp_q = np.exp(scaled_q_values - np.max(scaled_q_values))  # Subtract max for numerical stability
            probabilities = exp_q / np.sum(exp_q)
            # Sample according to probabilities
            return np.random.choice(self.num_actions, p=probabilities)
        
        # Otherwise, use greedy selection
        return np.argmax(q_values)
    
    def replay(self, batch_size):
        """Enhanced replay with prioritization of diverse experiences."""
        if len(self.memory) < batch_size:
            return
        
        # Standard minibatch sampling - could be enhanced with prioritized experience replay
        minibatch = random.sample(self.memory, batch_size)
        
        # Prepare batch data (more complex due to multiple inputs)
        states_melody = np.vstack([t[0]["melody"] for t in minibatch])
        states_chord = np.vstack([t[0]["chord"] for t in minibatch])
        states_bass = np.vstack([t[0]["bass"] for t in minibatch])
        states_context = np.vstack([t[0]["context"] for t in minibatch])
        
        next_states_melody = np.vstack([t[3]["melody"] for t in minibatch])
        next_states_chord = np.vstack([t[3]["chord"] for t in minibatch])
        next_states_bass = np.vstack([t[3]["bass"] for t in minibatch])
        next_states_context = np.vstack([t[3]["context"] for t in minibatch])
        
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])
        
        # Predict Q-values for next states (for target calculation)
        next_q_values = self.model.predict([
            next_states_melody, next_states_chord, 
            next_states_bass, next_states_context
        ], verbose=0)
        
        # Calculate target Q-values using the Bellman equation
        targets = self.model.predict([
            states_melody, states_chord, 
            states_bass, states_context
        ], verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                # Use double DQN strategy: select action using model, evaluate using target
                targets[i, actions[i]] = rewards[i] + self.gamma * np.amax(next_q_values[i])
        
        # Train the model
        history = self.model.fit(
            [states_melody, states_chord, states_bass, states_context], 
            targets, epochs=1, verbose=0, batch_size=32
        )
        
        # Decay exploration parameters
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        if self.temperature > self.temperature_min:
            self.temperature *= self.temperature_decay

# --- Training Loop Example ---
def train_improved_agent(episodes=500, batch_size=64):
    """Train the improved RL agent with more diverse musical patterns."""
    # Define several chord progressions for more variety during training
    chord_progressions = [
        # Basic jazz progressions
        ['Cmaj7', 'Dm7', 'G7', 'Cmaj7'] * 4,  # II-V-I in C
        ['Dm7', 'G7', 'Cmaj7', 'Cmaj7'] * 4,  # II-V-I with extra I
        ['Dm7', 'G7', 'Em7', 'A7', 'Dm7', 'G7', 'Cmaj7', 'Cmaj7'] * 2,  # Chain of II-V's
        
        # Modal progressions
        ['Dm7', 'G7', 'Cmaj7', 'Am7', 'Dm7', 'G7', 'Em7', 'A7'] * 2,
        
        # Minor key progressions
        ['Am7', 'D7', 'Gmaj7', 'Cmaj7', 'F#m7b5', 'B7', 'Em7', 'E7'] * 2,
        
        # Blues-based progressions
        ['C7', 'F7', 'C7', 'C7', 'F7', 'F7', 'C7', 'C7', 'G7', 'F7', 'C7', 'G7'] * 1,
    ]
    
    # Define varied melodies - using both scales and more interesting patterns
    melody_sequences = [
        # C major scale ascending and descending
        [60, 62, 64, 65, 67, 69, 71, 72, 71, 69, 67, 65, 64, 62, 60] * 3,
        
        # Chromatic runs
        [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61] * 2,
        
        # Arpeggios
        [60, 64, 67, 72, 67, 64, 60, 64, 67, 72, 67, 64] * 3,
        
        # More melodic pattern with some leaps
        [60, 64, 62, 67, 65, 69, 67, 72, 71, 67, 65, 62, 60] * 3,
        
        # Minor pentatonic pattern
        [60, 63, 65, 67, 70, 72, 70, 67, 65, 63] * 3,
    ]
    
    # Create training environment with first progression/melody
    env = JazzEnvironment(
        chord_progression=chord_progressions[0], 
        melody_sequence=melody_sequences[0], 
        tempo=120
    )
    
    # Initialize the improved agent
    state_shapes = {
        "melody": (1, *STATE_SHAPE_MELODY),
        "chord": (1, *STATE_SHAPE_CHORD),
        "bass": (1, *STATE_SHAPE_BASS),
        "context": (1, *STATE_SHAPE_CONTEXT)
    }
    agent = ImprovedRLJazzAgent(
        state_shapes=state_shapes, 
        num_actions=NUM_BASS_NOTES, 
        epsilon=1.0,  # Start with high exploration
        epsilon_min=0.05,  # Keep some minimal exploration
        epsilon_decay=0.996,  # Slower decay
        gamma=0.97,  # Slightly higher discount factor
        learning_rate=0.001
    )
    
    episode_rewards = []
    running_reward = None  # For tracking progress
    
    for e in range(episodes):
        # Every N episodes, change chord progression and melody for more variety
        if e % 20 == 0:
            prog_idx = (e // 20) % len(chord_progressions)
            melody_idx = (e // 20) % len(melody_sequences)
            env = JazzEnvironment(
                chord_progression=chord_progressions[prog_idx],
                melody_sequence=melody_sequences[melody_idx],
                tempo=120
            )
            print(f"\nSwitching to progression {prog_idx+1} and melody {melody_idx+1}")
        
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < env.max_steps:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # Store this experience
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Train the agent
            if len(agent.memory) >= batch_size:
                agent.replay(batch_size)
        
        # Update running reward
        if running_reward is None:
            running_reward = total_reward
        else:
            running_reward = 0.05 * total_reward + 0.95 * running_reward
        
        episode_rewards.append(total_reward)
        
        # Print status
        print(f"Episode: {e+1}/{episodes}, Steps: {steps}, " + 
              f"Reward: {total_reward:.2f}, Running Reward: {running_reward:.2f}, " +
              f"Epsilon: {agent.epsilon:.3f}, Temp: {agent.temperature:.2f}")
        
        # Save periodically
        if (e + 1) % 50 == 0 or e == episodes - 1:
            agent.save(f"improved_jazz_agent_ep{e+1}.weights.h5")
            
            # Visualize progress with a test run
            if (e + 1) % 100 == 0:
                print("\nRunning quick test...")
                test_rl_agent(agent, chord_progressions[-1], melody_sequences[-1], save_plot=True)
    
    # Plot learning curve
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, alpha=0.6, label='Episode Rewards')
    
    # Plot smoothed rewards
    smoothed_rewards = []
    window_size = 20
    for i in range(len(episode_rewards)):
        if i < window_size:
            smoothed_rewards.append(sum(episode_rewards[:i+1]) / (i+1))
        else:
            smoothed_rewards.append(sum(episode_rewards[i-window_size+1:i+1]) / window_size)
    plt.plot(smoothed_rewards, 'r', linewidth=2, label='Smoothed Rewards')
    
    plt.title('Improved RL Jazz Agent Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("jazz_rl_training_progress.png")
    plt.show()
    
    return agent

# --- Function to Test the Trained Agent ---
def test_rl_agent(agent, chord_prog, melody, tempo=120, save_plot=False, filename=None):
    """Test the trained agent and visualize results."""
    print("\n--- Testing RL Agent ---")
    env = JazzEnvironment(chord_progression=chord_prog, melody_sequence=melody, tempo=tempo)
    state = env.reset()
    done = False
    generated_bass_line = []
    steps = 0
    rewards = []
    
    # Turn off exploration for deterministic output
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    chord_symbols = []  # Track chord changes
    
    while not done and steps < env.max_steps:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        bass_note_midi = action + BASS_NOTE_MIN
        generated_bass_line.append(bass_note_midi)
        rewards.append(reward)
        
        # Record current chord
        current_chord = env.chord_history[-1]
        chord_symbols.append(current_chord)
        
        # Print less frequently for longer sequences
        if steps % 4 == 0 or steps < 10:
            print(f"Step {steps+1}: Chord={current_chord}, " +
                  f"Bass={bass_note_midi} ({note.Note(midi=bass_note_midi).nameWithOctave}), " +
                  f"Reward={reward:.2f}")
        
        steps += 1
    
    # Restore exploration parameter
    agent.epsilon = original_epsilon
    
    print(f"\nGenerated {len(generated_bass_line)} bass notes")
    
    # Calculate statistics
    unique_notes = len(set(generated_bass_line))
    total_reward = sum(rewards)
    
    print(f"Performance metrics:")
    print(f"- Unique notes used: {unique_notes}")
    print(f"- Total reward: {total_reward:.2f}")
    print(f"- Average reward per step: {total_reward/len(rewards):.3f}")
    
    # Plot the generated bass line
    plt.figure(figsize=(14, 6))
    
    # Plot bass line
    plt.subplot(2, 1, 1)
    plt.plot(generated_bass_line, 'bo-', markersize=4, linewidth=1, label='Bass Notes')
    
    # Add chord change markers
    chord_changes = []
    prev_chord = None
    for i, chord in enumerate(chord_symbols):
        if chord != prev_chord:
            chord_changes.append(i)
            plt.axvline(x=i, color='r', linestyle='--', alpha=0.3)
            prev_chord = chord
    
    plt.title('Bass Line Generated by RL Agent')
    plt.ylabel('MIDI Note')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot rewards
    plt.subplot(2, 1, 2)
    plt.plot(rewards, 'g-', label='Rewards')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('Rewards per Step')
    plt.xlabel('Time Step')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        filename = filename or f"bass_line_test_{int(time.time())}.png"
        plt.savefig(filename)
    
    plt.show()
    
    # Try to play the bass line
    try:
        print("\nPlaying generated bass line...")
        play_audio_simulation(generated_bass_line, tempo)
    except Exception as e:
        print(f"Couldn't play audio: {e}")
    
    return generated_bass_line

# --- Helper function for audio playback (adapted from original) ---
def play_audio_simulation(bass_line, tempo):
    """Play the generated bass line using pygame"""
    try:
        pygame.mixer.quit() # Ensure clean state
        pygame.mixer.init(44100, -16, 2, 512)
        pygame.init() # Ensure pygame itself is initialized

        quarter_note_duration_ms = int(60000 / tempo) # Duration in milliseconds

        def create_sound(note_value, duration_sec=0.25):
            freq = 440 * (2 ** ((note_value - 69) / 12))
            sample_rate = 44100
            t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
            wave = 0.7 * np.sin(freq * 2 * np.pi * t) + \
                   0.2 * np.sin(2 * freq * 2 * np.pi * t) + \
                   0.1 * np.sin(3 * freq * 2 * np.pi * t)
            envelope = np.ones_like(wave)
            attack = int(0.01 * sample_rate)
            release = int(0.15 * sample_rate)
            if len(envelope) > release: # Prevent index error on short sounds
                 envelope[:attack] = np.linspace(0, 1, attack)
                 envelope[-release:] = np.linspace(1, 0, release)
            wave = wave * envelope
            stereo_wave = np.column_stack((wave, wave))
            stereo_wave = (stereo_wave * 32767).astype(np.int16)
            return pygame.sndarray.make_sound(stereo_wave)

        sound_cache = {}
        print(f"Playing {len(bass_line)} bass notes at {tempo} BPM...")

        for note_value in bass_line:
            if note_value < BASS_NOTE_MIN or note_value > BASS_NOTE_MAX:
                 print(f"Skipping invalid note: {note_value}")
                 pygame.time.wait(quarter_note_duration_ms) # Wait even if note is invalid
                 continue

            if note_value not in sound_cache:
                sound_cache[note_value] = create_sound(note_value)

            sound_cache[note_value].play()
            pygame.time.wait(quarter_note_duration_ms) # Wait for duration of the note

        pygame.time.wait(500) # Wait a bit longer at the end

    except Exception as e:
        print(f"Audio playback error: {e}")
        print("Ensure pygame is installed and audio output is configured.")
    finally:
        try:
            pygame.mixer.quit()
            # pygame.quit() # Quit pygame itself if it was initialized here
        except Exception:
            pass


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Improved Reinforcement Learning Jazz Bass Generator...")
    
    # Train the improved agent (reduce episodes for testing)
    trained_agent = train_improved_agent(episodes=50, batch_size=64)
    
    # Test the trained agent on a challenging progression
    test_chord_prog = ['Am7', 'D7', 'Gmaj7', 'Cmaj7', 'F#m7b5', 'B7', 'Em7', 'A7']
    
    # Create a more interesting melody
    test_melody = []
    for _ in range(4):  # Create a longer melody pattern
        test_melody.extend([67, 69, 71, 72, 74, 71, 69, 67])  # Ascending/descending pattern
        test_melody.extend([69, 71, 72, 74, 76, 74, 72, 69])  # Higher variation
    
    # Run the test
    test_rl_agent(trained_agent, test_chord_prog, test_melody, tempo=140)
    
    print("\nRL Jazz Bass Generation Complete.")