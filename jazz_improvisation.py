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
from midi_dataset import MidiDataset
import logging
import music21
# --- Constants ---
SEQUENCE_LENGTH = 16 # Shorter sequence for RL state might be more manageable
BASS_NOTE_MIN = 32  # E1
BASS_NOTE_MAX = 60  # C4
NUM_BASS_NOTES = BASS_NOTE_MAX - BASS_NOTE_MIN + 1
STATE_SHAPE_MELODY = (SEQUENCE_LENGTH, 128) # Melody history
STATE_SHAPE_CHORD = (SEQUENCE_LENGTH, 24)   # Chord history
STATE_SHAPE_BASS = (SEQUENCE_LENGTH, 1)     # Bass history (MIDI values)
STATE_SHAPE_CONTEXT = (3,) # Tempo, Current Chord Root, Current Chord Type Simplified


import numpy as np
import music21
import logging

logger = logging.getLogger(__name__)

class JazzEnvironment:
    def __init__(self, chord_progression, melody_sequence, tempo):
        self.chord_progression = chord_progression
        self.melody_sequence = melody_sequence
        self.tempo = tempo
        self.max_steps = 128  # Maximum steps per episode
        self.sequence_length = 16  # For state history
        
        # Constants (adjust these based on your original code)
        self.bass_note_min = 32  # E1
        self.bass_note_max = 60  # C4
        self.midi_note_range = 128
        
        # State variables
        self.current_step = 0
        self.chord_history = []
        self.melody_history = []
        self.bass_history = []
        
        # Initialize state
        self.reset()

    def reset(self):
        """Reset the environment to the initial state."""
        self.current_step = 0
        self.chord_history = [self.chord_progression[0]] * self.sequence_length
        self.melody_history = [self.melody_sequence[0] if self.melody_sequence else 60] * self.sequence_length
        self.bass_history = [0] * self.sequence_length
        
        state = self._get_state()
        logger.debug("Environment reset: Initial state prepared")
        return state

    def _get_state(self):
        """Prepare the current state for the agent."""
        melody_state = np.zeros((self.sequence_length, self.midi_note_range))
        for t in range(self.sequence_length):
            if t < len(self.melody_history):
                note = self.melody_history[t]
                if note is not None and 0 <= note < self.midi_note_range:
                    melody_state[t, note] = 1
        
        chord_state = np.zeros((self.sequence_length, 24))  # 12 pitch classes + 12 for chord types
        for t in range(self.sequence_length):
            if t < len(self.chord_history):
                try:
                    chord = music21.harmony.ChordSymbol(self.chord_history[t])
                    root = chord.root().midi % 12
                    chord_type = 0  # Simplified: 0 for major, 1 for minor, etc.
                    if "m7" in self.chord_history[t]:
                        chord_type = 1
                    elif "7" in self.chord_history[t]:
                        chord_type = 2
                    chord_state[t, root] = 1
                    chord_state[t, 12 + chord_type] = 1
                except Exception as e:
                    logger.warning(f"Invalid chord at step {self.current_step}: {self.chord_history[t]}, {e}")
                    chord_state[t, 0] = 1  # Default to Cmaj
        
        bass_state = np.array(self.bass_history[-self.sequence_length:], dtype=np.float32).reshape(-1, 1)
        bass_state = (bass_state - self.bass_note_min) / (self.bass_note_max - self.bass_note_min)
        
        context_state = np.array([
            self.tempo / 240.0,  # Normalize tempo
            (music21.harmony.ChordSymbol(self.chord_history[-1]).root().midi % 12) / 12.0 if self.chord_history else 0,
            0  # Chord type (simplified)
        ], dtype=np.float32)
        
        return {
            "melody": melody_state[np.newaxis, :],
            "chord": chord_state[np.newaxis, :],
            "bass": bass_state[np.newaxis, :],
            "context": context_state[np.newaxis, :]
        }

    def step(self, action):
        """Take a step in the environment based on the agent's action."""
        self.current_step += 1
        
        # Convert action to MIDI note
        bass_note = action + self.bass_note_min
        self.bass_history.append(bass_note)
        
        # Update chord and melody histories
        chord_idx = (self.current_step // 4) % len(self.chord_progression)
        melody_idx = self.current_step % len(self.melody_sequence)
        current_chord = self.chord_progression[chord_idx]
        current_melody = self.melody_sequence[melody_idx] if self.melody_sequence else 60
        
        self.chord_history.append(current_chord)
        self.melody_history.append(current_melody)
        
        # Calculate reward
        reward = self._calculate_reward(bass_note, current_chord, current_melody)
        
        # Determine if episode is done
        done = False
        if self.current_step >= self.max_steps:
            done = True
            logger.debug(f"Episode done: Reached max steps ({self.max_steps})")
        elif self.current_step >= len(self.melody_sequence):
            done = True
            logger.debug(f"Episode done: Reached end of melody sequence ({len(self.melody_sequence)} steps)")
        
        # Prepare next state
        next_state = self._get_state()
        
        # Additional info for debugging
        info = {
            "bass_note": bass_note,
            "current_chord": current_chord,
            "current_melody": current_melody
        }
        
        logger.debug(f"Step {self.current_step}: Action={action}, Bass Note={bass_note}, Chord={current_chord}, Melody={current_melody}, Reward={reward:.2f}, Done={done}")
        
        return next_state, reward, done, info

    def _calculate_reward(self, bass_note, chord, melody_note):
        """Calculate the reward based on the bass note, chord, and melody."""
        reward = 0.0
        try:
            chord_obj = music21.harmony.ChordSymbol(chord)
            chord_pitches = [p.midi % 12 for p in chord_obj.pitches]
            
            # Reward for playing a chord tone
            bass_pitch_class = bass_note % 12
            if bass_pitch_class in chord_pitches:
                reward += 1.0
                if bass_pitch_class == (chord_obj.root().midi % 12):
                    reward += 0.5  # Extra reward for root note
            
            # Reward for smooth voice leading (penalize large jumps)
            if len(self.bass_history) > 1:
                prev_bass = self.bass_history[-2]
                interval = abs(bass_note - prev_bass)
                if interval <= 4:
                    reward += 0.3
                elif interval > 7:
                    reward -= 0.2
            
            # Penalize dissonance with melody
            melody_pitch_class = melody_note % 12
            interval_with_melody = min(abs(bass_pitch_class - melody_pitch_class), 12 - abs(bass_pitch_class - melody_pitch_class))
            if interval_with_melody == 1:  # Penalize semitone clashes
                reward -= 0.5
        
        except Exception as e:
            logger.warning(f"Error calculating reward: {e}")
            reward = 0.0
        
        return reward


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




logger = logging.getLogger(__name__)

def train_rl_agent(episodes=1000, batch_size=32, midi_dir="midi/lmd_matched"):
    """
    Train the RL agent using MIDI dataset for melody and chord progressions.
    
    Args:
        episodes (int): Number of training episodes.
        batch_size (int): Batch size for replay.
        midi_dir (str): Directory containing MIDI files.
    """
    # Initialize dataset
    try:
        dataset = MidiDataset(midi_dir=midi_dir, max_files=100)
    except ValueError as e:
        logger.error(f"Failed to initialize dataset: {e}")
        raise
    
    env = JazzEnvironment(chord_progression=['Cmaj7'], melody_sequence=[60], tempo=120)  # Placeholder
    state_shapes = {
        "melody": (1, *STATE_SHAPE_MELODY),
        "chord": (1, *STATE_SHAPE_CHORD),
        "bass": (1, *STATE_SHAPE_BASS),
        "context": (1, *STATE_SHAPE_CONTEXT)
    }
    agent = RLJazzAgent(state_shapes=state_shapes, num_actions=NUM_BASS_NOTES)
    
    episode_rewards = []
    
    for e in range(episodes):
        # Load a new melody and chord progression from dataset
        melody, chord_prog = dataset.get_single_sequence()  # Randomly select a sequence
        if not melody or not chord_prog:
            logger.warning(f"Episode {e+1}: No valid data, using default.")
            melody = [60] * 32
            chord_prog = ['Cmaj7', 'Dm7', 'G7', 'Cmaj7'] * 2
        
        # Ensure melody length is sufficient
        if len(melody) < SEQUENCE_LENGTH:
            melody = melody * (SEQUENCE_LENGTH // len(melody) + 1)
        melody = melody[:env.max_steps]  # Truncate to max_steps if needed
        
        # Initialize environment with dataset-derived data
        env = JazzEnvironment(chord_progression=chord_prog, melody_sequence=melody, tempo=120)
        
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
            
            # Train the agent
            agent.replay(batch_size)
        
        episode_rewards.append(total_reward)
        logger.info(f"Episode: {e+1}/{episodes}, Steps: {steps}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
        
        # Save model periodically
        if (e + 1) % 50 == 0:
            agent.save(f"rl_jazz_agent_episode_{e+1}.weights.h5")
    
    # Plot rewards
    import matplotlib.pyplot as plt
    plt.plot(episode_rewards)
    plt.title('RL Agent Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig('training_rewards.png')
    plt.close()
    
    return agent
# --- Function to Test the Trained Agent ---
logger = logging.getLogger(__name__)

def test_rl_agent(agent, midi_dir="midi/lmd_matched", tempo=120):
    """
    Test the trained RL agent using a melody and chord progression from the dataset.
    
    Args:
        agent: Trained RLJazzAgent instance.
        midi_dir (str): Directory containing MIDI files.
        tempo (int): Tempo in BPM.
    """
    logger.info("\n--- Testing Trained RL Agent ---")
    
    # Load dataset
    try:
        dataset = MidiDataset(midi_dir=midi_dir, max_files=100)
    except ValueError as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    melody, chord_prog = dataset.get_single_sequence()  # Randomly select a sequence
    
    if not melody or not chord_prog:
        logger.warning("No valid data, using default.")
        melody = [60, 62, 64, 65, 67, 69, 71, 72] * 4
        chord_prog = ['Am7', 'D7', 'Gmaj7', 'Cmaj7'] * 2
    
    # Initialize environment
    env = JazzEnvironment(chord_progression=chord_prog, melody_sequence=melody, tempo=tempo)
    state = env.reset()
    done = False
    generated_bass_line = []
    steps = 0
    
    agent.epsilon = 0.0  # Disable exploration
    
    while not done and steps < env.max_steps:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        bass_note_midi = action + BASS_NOTE_MIN
        generated_bass_line.append(bass_note_midi)
        logger.info(f"Step {steps+1}: Chord={env.chord_history[-1]}, Melody={env.melody_history[-1]}, Chosen Bass={bass_note_midi} ({music21.note.Note(midi=bass_note_midi).nameWithOctave}), Reward={reward:.2f}")
        steps += 1
    
    logger.info("\nGenerated Bass Line (MIDI): %s", generated_bass_line)
    
    # Play the generated bass line
    play_audio_simulation(generated_bass_line, tempo)
    
    # Visualize
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    plt.plot(generated_bass_line, 'bo-', label='Generated Bass')
    plt.title('Bass Line Generated by RL Agent')
    plt.ylabel('MIDI Note')
    plt.xlabel('Time Step')
    plt.grid(True)
    plt.savefig('bass_line.png')
    plt.close()


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
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Reinforcement Learning Jazz Improvisation...")
    
    # Train the agent
    trained_agent = train_rl_agent(episodes=2, batch_size=32, midi_dir="midi/lmd_matched")
    
    # Test the trained agent
    test_rl_agent(trained_agent, midi_dir="midi/lmd_matched", tempo=140)
    
    logger.info("\nRL Jazz Improvisation Test Complete.")

    # The interactive parts from the original code would need significant
    # rewriting to work with the RL agent/environment structure.
    # You'd essentially run the test loop but get melody input from MIDI
    # and update the environment state accordingly.