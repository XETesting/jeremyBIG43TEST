#!/usr/bin/env python

import numpy as np
from scipy import signal
import sounddevice as sd
import time
import threading
import sys

# ==================== GUITAR STRING SIMULATION ====================
# A simple but realistic plucked string synthesizer with Karplus-Strong + lowpass + body resonance
# Run this in a terminal with Python + numpy + scipy + sounddevice installed:
#   pip install numpy scipy sounddevice

class GuitarString:
    def __init__(self, fs=44100):
        self.fs = fs
        self.frequency = 440.0          # A4 default
        self.tension_factor = 1.0       # 0.5 = loose, 2.0 = very tight
        self.damping = 0.995            # how fast the string decays
        self.pick_pos = 0.15            # where you pluck (0-1)
        self.buffer = np.zeros(0)
        self.ptr = 0
        self.is_playing = False
        self.volume = 0.8
        self.note_name = "A4"
        
        self.body_filter = signal.butter(2, [80, 800], btype='band', fs=fs, output='sos')
        self.zi = signal.sosfilt_zi(self.body_filter)

    def note_to_freq(self, note):
        """Convert scientific pitch notation (e.g. 'E2', 'A4', 'G#5') to frequency"""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = int(note[-1])
        note_name = note[:-1]
        if note_name[-1] in ['#', 'b']:
            note_name = note[:-2] + note[-1] if len(note) > 2 else note_name
        semitone = notes.index(note_name.replace('b', 'A#').replace('Bb','A#')) if '#' in note_name or 'b' in note_name else notes.index(note_name)
        if 'b' in note_name:
            semitone -= 1
        return 440.0 * (2 ** ((semitone + (octave-4)*12 - 9) / 12.0))

    def freq_to_note(self, freq):
        """Convert frequency to closest note name"""
        if freq <= 0: return "?"
        n = 12 * np.log2(freq / 440.0) + 69
        n_round = int(round(n))
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = n_round // 12 - 1
        note = notes[n_round % 12]
        return f"{note}{octave}"

    def tune(self, semitones):
        """Tune the string up or down by semitones (positive = up)"""
        self.frequency *= 2 ** (semitones / 12.0)
        self.update_buffer()
        self.note_name = self.freq_to_note(self.frequency)
        print(f"→ Tuned to {self.note_name} ({self.frequency:.1f} Hz)")

    def update_buffer(self):
        """Rebuild the delay line for current frequency"""
        period = int(self.fs / (self.frequency * self.tension_factor))
        if period < 10:
            period = 10
        self.buffer = np.zeros(period, dtype=np.float32)
        # Initial pluck shape (triangle for realism)
        pluck_len = max(5, int(period * self.pick_pos))
        self.buffer[:pluck_len] = np.linspace(1.0, 0.0, pluck_len)
        self.buffer[pluck_len:] = np.linspace(0.0, -0.3, period - pluck_len)
        self.ptr = 0

    def pluck(self):
        """Pluck the string - resets the buffer with a new excitation"""
        self.update_buffer()
        self.is_playing = True
        self.zi = signal.sosfilt_zi(self.body_filter)  # reset body filter state

    def get_sample(self):
        """Generate one audio sample using Karplus-Strong with filtering"""
        if not self.is_playing or len(self.buffer) == 0:
            return 0.0

        out = self.buffer[self.ptr]

        # Low-pass average (the "string" part)
        avg = 0.5 * (out + self.buffer[(self.ptr + 1) % len(self.buffer)])
        
        # Apply damping
        self.buffer[self.ptr] = avg * self.damping

        # Guitar body resonance (simple bandpass to give woody timbre)
        filtered, self.zi = signal.sosfilt(self.body_filter, [avg], zi=self.zi)
        sample = filtered[0] * 1.8   # boost a bit

        self.ptr = (self.ptr + 1) % len(self.buffer)

        # Stop when amplitude is tiny
        if abs(sample) < 0.0005:
            self.is_playing = False

        return np.clip(sample * self.volume, -1.0, 1.0)

# ==================== INTERACTIVE PLAYER ====================
def play_guitar():
    fs = 44100
    string = GuitarString(fs)
    
    # Default to low E string (standard guitar 6th string)
    string.frequency = 82.41
    string.tune(0)  # force note name update
    
    print("\n🎸  ONE GUITAR STRING SIMULATOR  🎸")
    print("Commands:")
    print("  p          = Pluck the string (play note)")
    print("  + / up     = Tune up 1 semitone")
    print("  - / down   = Tune down 1 semitone")
    print("  = / 440    = Tune to A4 (440 Hz)")
    print("  e2 e3 a4 d5 etc. = Set to that note (e.g. e2, g3, bb4)")
    print("  q          = Quit")
    print("-" * 60)
    print(f"Current: {string.note_name} ({string.frequency:.1f} Hz)")

    # Audio callback
    audio_buffer = []
    block_size = 512

    def callback(outdata, frames, time, status):
        if status:
            print(status)
        for i in range(frames):
            sample = string.get_sample()
            outdata[i] = [sample, sample]   # stereo

    try:
        stream = sd.OutputStream(samplerate=fs, channels=2, blocksize=block_size, callback=callback)
        stream.start()
        
        while True:
            cmd = input("\n> ").strip().lower()
            
            if cmd in ['q', 'quit', 'exit']:
                break
            elif cmd == 'p':
                string.pluck()
                print(f"Plucked! → {string.note_name} ({string.frequency:.1f} Hz)")
            elif cmd in ['+', 'up']:
                string.tune(1)
            elif cmd in ['-', 'down']:
                string.tune(-1)
            elif cmd == '=' or cmd == '440':
                string.frequency = 440.0
                string.tune(0)
            elif cmd.isdigit() and 20 <= int(cmd) <= 2000:
                string.frequency = float(cmd)
                string.tune(0)
            else:
                # Try to parse as note name (e.g. a4, e2, f#5, bb3)
                try:
                    note = cmd.upper().replace('B', 'Bb') if 'b' in cmd and not cmd.endswith('b') else cmd.upper()
                    if any(c in note for c in 'CDEFGAB') and note[-1].isdigit():
                        string.frequency = string.note_to_freq(note)
                        string.tune(0)
                        continue
                except:
                    pass
                print("Unknown command. Try p, +, -, =, or a note like 'e2' or 'a4'")

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()
        print("\nGoodbye! 🎸")

if __name__ == "__main__":
    # Install requirements if needed:
    # pip install numpy scipy sounddevice
    
    play_guitar()
