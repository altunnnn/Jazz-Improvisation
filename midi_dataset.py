import os
import music21
import numpy as np
from typing import List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MidiDataset:
    """
    Loads and preprocesses MIDI files to extract melody and chord progressions.
    """
    def __init__(self, midi_dir: str, max_files: int = 100):
        """
        Initialize the dataset loader.
        
        Args:
            midi_dir (str): Directory containing MIDI files.
            max_files (int): Maximum number of MIDI files to process.
        """
        self.midi_dir = midi_dir
        self.max_files = max_files
        self.midi_files = []
        self._load_midi_files()

    def _load_midi_files(self):
        """Discover MIDI files in the specified directory and its subdirectories."""
        logger.info(f"Searching for MIDI files in {self.midi_dir}")
        if not os.path.exists(self.midi_dir):
            logger.error(f"Directory does not exist: {self.midi_dir}")
            raise ValueError(f"Directory does not exist: {self.midi_dir}")

        for root, dirs, files in os.walk(self.midi_dir):
            logger.debug(f"Scanning directory: {root}")
            for file in files:
                if file.lower().endswith(('.mid', '.midi')):
                    file_path = os.path.join(root, file)
                    self.midi_files.append(file_path)
                    logger.debug(f"Found MIDI file: {file_path}")
                    if len(self.midi_files) >= self.max_files:
                        break
            if len(self.midi_files) >= self.max_files:
                break
        
        if not self.midi_files:
            logger.error(f"No MIDI files found in {self.midi_dir} or its subdirectories")
            raise ValueError(f"No MIDI files found in {self.midi_dir}")
        logger.info(f"Found {len(self.midi_files)} MIDI files")

    def _parse_midi(self, midi_path: str) -> Tuple[List[int], List[str]]:
        """
        Parse a MIDI file to extract melody and chord progression.
        
        Args:
            midi_path (str): Path to the MIDI file.
        
        Returns:
            Tuple[List[int], List[str]]: Melody as MIDI note sequence, chords as list of chord symbols.
        """
        try:
            # Load MIDI file
            score = music21.converter.parse(midi_path, quarterLengthDivisor=1)
            
            # Extract melody (assume highest-pitched instrument or first track with notes)
            melody_notes = []
            for part in score.parts:
                notes = part.flat.notes
                if notes:  # Check if part contains notes
                    for element in notes:
                        try:
                            if isinstance(element, music21.note.Note):
                                if element.pitch.midi is not None:  # Ensure valid pitch
                                    melody_notes.append(element.pitch.midi)
                            elif isinstance(element, music21.chord.Chord):
                                # Take the highest note in the chord for melody
                                valid_pitches = [p.midi for p in element.pitches if p.midi is not None]
                                if valid_pitches:
                                    melody_notes.append(max(valid_pitches))
                        except AttributeError as e:
                            logger.warning(f"Skipping invalid note/chord in {midi_path}: {e}")
                            continue
                    break  # Use the first part with notes as melody
            
            # Extract chords using a more robust method
            chord_progression = []
            for part in score.parts:
                chords = part.flat.getElementsByClass(music21.chord.Chord)
                if chords:
                    for c in chords:
                        try:
                            # Try to derive chord symbol using pitch classes
                            pitches = [p.midi % 12 for p in c.pitches if p.midi is not None]
                            if not pitches:
                                continue
                            root = min(pitches)  # Simplistic root detection
                            chord_type = "maj"  # Default to major
                            # Basic chord type detection
                            pitch_set = set(pitches)
                            if (root + 4) % 12 in pitch_set and (root + 7) % 12 in pitch_set:
                                chord_type = "maj7"
                            elif (root + 3) % 12 in pitch_set and (root + 7) % 12 in pitch_set:
                                chord_type = "m7"
                            elif (root + 4) % 12 in pitch_set and (root + 10) % 12 in pitch_set:
                                chord_type = "7"
                            root_note = music21.note.Note(root).name
                            chord_symbol = f"{root_note}{chord_type}"
                            chord_progression.append(chord_symbol)
                        except Exception as e:
                            logger.warning(f"Skipping invalid chord in {midi_path}: {e}")
                            continue
                    break  # Use the first part with chords
            
            # If no chords found, infer basic chords
            if not chord_progression and melody_notes:
                chord_progression = self._infer_chords(melody_notes)
            
            if not melody_notes or not chord_progression:
                logger.error(f"No valid melody or chords extracted from {midi_path}")
                return [], []
            
            logger.info(f"Parsed {midi_path}: {len(melody_notes)} melody notes, {len(chord_progression)} chords")
            return melody_notes, chord_progression
        
        except Exception as e:
            logger.error(f"Error parsing {midi_path}: {e}")
            return [], []

    def _infer_chords(self, melody_notes: List[int]) -> List[str]:
        """
        Infer a basic chord progression from melody notes if no chords are found.
        
        Args:
            melody_notes (List[int]): List of MIDI note numbers.
        
        Returns:
            List[str]: List of inferred chord symbols.
        """
        chords = []
        for i in range(0, len(melody_notes), 4):  # Assume 4 notes per chord
            if i < len(melody_notes):
                try:
                    note = music21.note.Note(melody_notes[i])
                    chord_symbol = f"{note.name}maj7"
                    chords.append(chord_symbol)
                except Exception:
                    continue
        return chords if chords else ['Cmaj7']  # Fallback to single chord

    def get_dataset(self) -> List[Tuple[List[int], List[str]]]:
        """
        Process all MIDI files and return a list of (melody, chords) pairs.
        
        Returns:
            List[Tuple[List[int], List[str]]]: List of (melody sequence, chord progression) pairs.
        """
        dataset = []
        for midi_file in self.midi_files:
            melody, chords = self._parse_midi(midi_file)
            if melody and chords:
                dataset.append((melody, chords))
        logger.info(f"Loaded {len(dataset)} valid MIDI sequences")
        return dataset

    def get_single_sequence(self, index: int = None) -> Tuple[List[int], List[str]]:
        """
        Get a single (melody, chords) pair, either by index or randomly.
        
        Args:
            index (int, optional): Index of the MIDI file to load. If None, choose randomly.
        
        Returns:
            Tuple[List[int], List[str]]: Melody sequence and chord progression.
        """
        if not self.midi_files:
            logger.warning("No MIDI files available, returning default")
            return [], ['Cmaj7']
        
        if index is None:
            index = np.random.randint(0, len(self.midi_files))
        index = min(index, len(self.midi_files) - 1)
        
        return self._parse_midi(self.midi_files[index])