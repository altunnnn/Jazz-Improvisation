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
        Calculate the reward for the chosen bass note (action).
        This requires musical rules.
        """
        reward = 0.0
        bass_note_midi = bass_action + BASS_NOTE_MIN # Convert action index back to MIDI

        current_chord_name = self.chord_history[-1]
        prev_bass_note_midi = self.bass_note_history[-1] if len(self.bass_note_history) > 0 else -1

        if not current_chord_name:
            return 0.0 # No chord context, neutral reward

        try:
            c = chord.Chord(current_chord_name)
            chord_pitches = c.pitches
            chord_tone_classes = {p.pitchClass for p in chord_pitches} # Pitch classes (0-11)
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

            # 2. Scale Tone Bonus: Is the note part of a related scale? (e.g., Mixolydian for Dom7)
            # (Simplified: Check if it's diatonic to a major scale of the root for major/dom, minor for minor)
            # This is complex, let's keep it simple for now. If not a chord tone, check if dissonant.

            # 3. Voice Leading Reward: Smooth transition from previous note?
            if prev_bass_note_midi > 0:
                prev_bass_note_obj = note.Note(midi=prev_bass_note_midi)
                inter = interval.Interval(prev_bass_note_obj, bass_note_obj)
                # Penalize large leaps (e.g., > octave)
                if abs(inter.semitones) > 12:
                    reward -= 0.3
                # Reward steps (major/minor 2nd)
                elif abs(inter.semitones) <= 2:
                    reward += 0.2
                # Small reward for consonant leaps (3rd, 4th, 5th, 6th)
                elif abs(inter.semitones) <= 9:
                    reward += 0.05

            # 4. Dissonance Penalty: Does it clash badly? (e.g., minor 2nd against root/3rd/5th)
            # Check for half-step clashes with primary chord tones
            clash_penalty = 0
            for p in chord_pitches:
                 # Check against root, 3rd, 5th primarily
                 if p.pitchClass in [root_pc, (root_pc + 3)%12, (root_pc + 4)%12, (root_pc + 7)%12]:
                     diff = abs(bass_pc - p.pitchClass)
                     if diff == 1 or diff == 11: # Semitone clash
                         clash_penalty = -0.6
                         break
            reward += clash_penalty

            # 5. Rhythmic Placement (Simple): Bonus for root on beat 1 (needs beat tracking)
            # beat_in_measure = (self.current_step % self.beats_per_chord) + 1
            # if beat_in_measure == 1 and bass_pc == root_pc:
            #    reward += 0.2

        except Exception as e:
            # print(f"Reward calculation error: {e}") # Avoid printing during training
            pass # Return default reward if chord parsing fails

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


# --- Training Loop Example ---
def train_rl_agent(episodes=1000, batch_size=32):
    # Define a sample chord progression and melody
    chord_prog = ['Cmaj7', 'Fmaj7', 'Dm7', 'G7'] * 4 # Repeat progression
    # Simple C major scale melody
    melody = [60, 62, 64, 65, 67, 69, 71, 72] * (len(chord_prog)) # Match length roughly

    env = JazzEnvironment(chord_progression=chord_prog, melody_sequence=melody, tempo=120)
    state_shapes = { # Get shapes from an initial state
        "melody": (1, *STATE_SHAPE_MELODY),
        "chord": (1, *STATE_SHAPE_CHORD),
        "bass": (1, *STATE_SHAPE_BASS),
        "context": (1, *STATE_SHAPE_CONTEXT)
    }
    agent = RLJazzAgent(state_shapes=state_shapes, num_actions=NUM_BASS_NOTES)

    episode_rewards = []

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < env.max_steps:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

            # Train the agent if enough memory samples are available
            agent.replay(batch_size)

        episode_rewards.append(total_reward)
        print(f"Episode: {e+1}/{episodes}, Steps: {steps}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

        # Optionally save the model periodically
        if (e + 1) % 50 == 0:
            agent.save(f"rl_jazz_agent_episode_{e+1}.weights.h5")

    # Plot rewards
    plt.plot(episode_rewards)
    plt.title('RL Agent Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

    return agent # Return the trained agent

# --- Function to Test the Trained Agent ---
def test_rl_agent(agent, chord_prog, melody, tempo=120):
    print("\n--- Testing Trained RL Agent ---")
    env = JazzEnvironment(chord_progression=chord_prog, melody_sequence=melody, tempo=tempo)
    state = env.reset()
    done = False
    generated_bass_line = []
    steps = 0

    agent.epsilon = 0.0 # Turn off exploration for testing

    while not done and steps < env.max_steps:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        bass_note_midi = action + BASS_NOTE_MIN
        generated_bass_line.append(bass_note_midi)
        print(f"Step {steps+1}: Chord={env.chord_history[-1]}, Melody={env.melody_history[-1]}, Chosen Bass={bass_note_midi} ({note.Note(midi=bass_note_midi).nameWithOctave}), Reward={reward:.2f}")
        steps += 1

    print("\nGenerated Bass Line (MIDI):", generated_bass_line)

    # Play the generated bass line (requires pygame setup similar to original code)
    play_audio_simulation(generated_bass_line, tempo)

    # Visualize
    plt.figure(figsize=(12, 4))
    plt.plot(generated_bass_line, 'bo-', label='Generated Bass')
    # You could overlay melody or chord changes here too
    plt.title('Bass Line Generated by RL Agent')
    plt.ylabel('MIDI Note')
    plt.xlabel('Time Step')
    plt.grid(True)
    plt.show()


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
    print("Starting Reinforcement Learning Jazz Improvisation...")

    # Train the agent
    # NOTE: Training can take a significant amount of time!
    # Reduce episodes for a quick test.
    trained_agent = train_rl_agent(episodes=2, batch_size=32) # Reduced episodes

    # Test the trained agent
    test_chord_prog = ['Am7', 'D7', 'Gmaj7', 'Cmaj7', 'F#m7b5', 'B7', 'Em7', 'A7']
    test_melody = [random.randint(60, 75) for _ in range(len(test_chord_prog) * 4)] # Random melody
    test_rl_agent(trained_agent, test_chord_prog, test_melody, tempo=140)

    print("\nRL Jazz Improvisation Test Complete.")

    # The interactive parts from the original code would need significant
    # rewriting to work with the RL agent/environment structure.
    # You'd essentially run the test loop but get melody input from MIDI
    # and update the environment state accordingly.