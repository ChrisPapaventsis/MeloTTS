from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.spinner import Spinner, SpinnerOption
from kivy.uix.slider import Slider
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.progressbar import ProgressBar
from kivy.uix.scrollview import ScrollView
from kivy.clock import Clock
from kivy.metrics import dp, sp
from kivy.lang import Builder
from kivy.properties import ListProperty, StringProperty, NumericProperty, ObjectProperty, BooleanProperty
from kivy.animation import Animation
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
from kivy.core.window import Window
import threading
import re
import pygame
import os
import torch
import io
import tempfile
import logging
import sys
import numpy as np
import queue
import sounddevice as sd
from melo.api import TTS
import time
import soundfile as sf

# Suppress all logging output
logging.getLogger().setLevel(logging.ERROR)
os.environ['KIVY_NO_CONSOLELOG'] = '1'
os.environ['NUMBA_WARNINGS'] = '0'
os.environ['NUMBA_DISABLE_JIT'] = '0'

# Audio stream configuration
CHUNK_SIZE = 4096
SAMPLE_RATE = 44100
CHANNELS = 1
DTYPE = np.float32

class AudioPlayer:
    def __init__(self):
        self.stream = None
        self.audio_data = None
        self.sample_rate = SAMPLE_RATE
        self.current_frame = 0
        self.is_playing = False
        self.audio_queue = queue.Queue()
        self.current_file_index = 0
        self.files = []
        self.paused_position = 0
        self.processed_duration = 0
        self.estimated_total_duration = 0
        self.remaining_words = 0
        
    def load_file(self, filename):
        """Load an audio file"""
        try:
            audio_data, sample_rate = sf.read(filename, dtype=DTYPE)
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]  # Convert to mono if stereo
            return audio_data, sample_rate
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return None, None
            
    def audio_callback(self, outdata, frames, time, status):
        """Callback for the sounddevice stream"""
        if status:
            print(status)
        
        if self.audio_data is None or not self.is_playing:
            outdata.fill(0)
            return
            
        if self.current_frame + frames > len(self.audio_data):
            # End of current chunk
            remaining = len(self.audio_data) - self.current_frame
            outdata[:remaining, 0] = self.audio_data[self.current_frame:self.current_frame + remaining]
            outdata[remaining:, 0] = 0
            
            # Move to next file
            self.current_file_index += 1
            if self.current_file_index < len(self.files):
                next_data, next_sr = self.load_file(self.files[self.current_file_index])
                if next_data is not None:
                    self.audio_data = next_data
                    self.current_frame = 0
                    # Fill remaining buffer with start of next chunk
                    remaining_frames = frames - remaining
                    if remaining_frames > 0:
                        outdata[remaining:, 0] = next_data[:remaining_frames]
                        self.current_frame = remaining_frames
            else:
                self.is_playing = False
                self.stream.stop()
        else:
            outdata[:, 0] = self.audio_data[self.current_frame:self.current_frame + frames]
            self.current_frame += frames
            
    def play(self, files, start_from_beginning=True):
        """Start playing a list of audio files"""
        if not files:
            return False
            
        if start_from_beginning:
            self.files = files
            self.current_file_index = 0
            self.current_frame = 0
            self.paused_position = 0
            
            # Load first file
            audio_data, sample_rate = self.load_file(files[0])
            if audio_data is None:
                return False
                
            self.audio_data = audio_data
            self.sample_rate = sample_rate
        else:
            # Resume from paused position
            self.seek(self.paused_position)
            
        if self.stream is None or not self.stream.active:
            self.stream = sd.OutputStream(
                channels=CHANNELS,
                samplerate=self.sample_rate,
                callback=self.audio_callback,
                dtype=DTYPE
            )
            
        self.stream.start()
        self.is_playing = True
        return True
        
    def pause(self):
        """Pause playback"""
        if self.is_playing:
            self.paused_position = self.get_position()
            self.is_playing = False
            if self.stream:
                self.stream.stop()
            
    def resume(self):
        """Resume playback from paused position"""
        if not self.is_playing and self.files:
            return self.play(self.files, start_from_beginning=False)
        return False
        
    def seek(self, position):
        """Seek to a specific position in seconds"""
        if not self.files:
            return
            
        # Calculate total duration up to target position
        current_pos = 0
        for i, file in enumerate(self.files):
            audio_data, sr = sf.read(file)
            duration = len(audio_data) / sr
            
            if current_pos + duration > position:
                # Found the target file
                self.current_file_index = i
                self.audio_data = audio_data
                self.current_frame = int((position - current_pos) * sr)
                self.current_frame = min(self.current_frame, len(self.audio_data))
                self.paused_position = position  # Update paused position
                break
            current_pos += duration
            
    def get_position(self):
        """Get current position in seconds"""
        if not self.files:
            return 0
            
        # Add up durations of completed files
        position = 0
        for i in range(self.current_file_index):
            audio_data, sr = sf.read(self.files[i])
            position += len(audio_data) / sr
            
        # Add current file position
        if self.audio_data is not None and self.sample_rate > 0:
            position += self.current_frame / self.sample_rate
            
        return position
        
    def get_duration(self):
        """Get the total duration (either actual if fully processed, or estimated)"""
        if not self.files:
            return 0
        if self.remaining_words == 0:  # All processed
            return self.get_processed_duration()
        return self.estimated_total_duration

    def stop(self):
        """Stop playback and clean up"""
        self.is_playing = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.audio_data = None
        self.current_frame = 0
        self.current_file_index = 0
        self.files = []

    def get_processed_duration(self):
        """Get total duration of processed chunks"""
        if not self.files:
            return 0
            
        total_duration = 0
        for file in self.files:
            try:
                audio_data, sr = sf.read(file)
                total_duration += len(audio_data) / sr
            except Exception:
                break  # Stop if we hit an unprocessed or invalid chunk
        return total_duration

    def estimate_duration_from_words(self, word_count):
        """Estimate duration in seconds based on word count"""
        return (word_count / 165.0) * 60  # Convert to seconds (165 words per minute)

    def update_duration_estimate(self, remaining_words):
        """Update duration estimate based on processed chunks and remaining words"""
        self.remaining_words = remaining_words
        processed_duration = self.get_processed_duration()
        remaining_duration = self.estimate_duration_from_words(remaining_words)
        self.estimated_total_duration = processed_duration + remaining_duration
        return self.estimated_total_duration

# Redirect stdout to devnull for TTS model output
class DummyFile:
    def write(self, x): pass
    def flush(self): pass

# Save the original stdout
original_stdout = sys.stdout

# Initialize pygame mixer with no logging
pygame.mixer.init(frequency=44100)
pygame.mixer.set_num_channels(2)  # Use 2 channels for crossfading

# Initialize TTS models silently
sys.stdout = DummyFile()
device = 'cpu'
models = {
    'EN': TTS(language='EN', device=device),
}
sys.stdout = original_stdout

# Add streaming functionality to TTS class
def stream_audio_callback(audio_chunk, sr, is_last=False):
    """Callback function to handle streaming audio chunks"""
    global AUDIO_QUEUE, BUFFERED_CHUNKS, PROGRESS_CALLBACK
    try:
        AUDIO_QUEUE.put((audio_chunk, sr))
        BUFFERED_CHUNKS += 1
        if PROGRESS_CALLBACK:
            PROGRESS_CALLBACK(BUFFERED_CHUNKS, TOTAL_CHUNKS)
    except Exception as e:
        global STREAM_ERROR
        STREAM_ERROR = str(e)

def reset_stream():
    """Reset all stream-related variables"""
    global STREAM, AUDIO_QUEUE, TOTAL_CHUNKS, BUFFERED_CHUNKS, STREAM_ERROR
    if STREAM is not None:
        try:
            STREAM.stop()
            STREAM.close()
        except:
            pass
        STREAM = None
    
    while not AUDIO_QUEUE.empty():
        AUDIO_QUEUE.get()
    
    TOTAL_CHUNKS = 0
    BUFFERED_CHUNKS = 0
    STREAM_ERROR = None

def set_progress_callback(callback):
    """Set callback for progress updates"""
    global PROGRESS_CALLBACK
    PROGRESS_CALLBACK = callback

def audio_stream_thread():
    """Background thread to handle audio streaming"""
    global STREAM, AUDIO_QUEUE, STREAM_ERROR
    
    while True:
        try:
            if STREAM_ERROR:
                # If there's an error, clear the queue and wait
                while not AUDIO_QUEUE.empty():
                    AUDIO_QUEUE.get()
                time.sleep(0.1)
                continue

            audio_chunk, sr = AUDIO_QUEUE.get(timeout=0.1)  # Add timeout to prevent busy waiting
            if STREAM is None or STREAM.samplerate != sr:
                if STREAM is not None:
                    STREAM.stop()
                    STREAM.close()
                STREAM = sd.OutputStream(
                    samplerate=sr,
                    channels=1,
                    dtype=np.float32,
                    callback=audio_callback
                )
                STREAM.start()
            
            # Write audio chunk to stream
            STREAM.write(audio_chunk)
            
        except queue.Empty:
            continue
        except Exception as e:
            STREAM_ERROR = str(e)
            print(f"Audio streaming error: {e}")
            if STREAM is not None:
                try:
                    STREAM.stop()
                    STREAM.close()
                except:
                    pass
                STREAM = None

def audio_callback(outdata, frames, time, status):
    """Callback for sounddevice stream"""
    if status:
        print(status)
    try:
        data = AUDIO_QUEUE.get_nowait()
        if len(data) < len(outdata):
            outdata[:len(data)] = data
            outdata[len(data):] = 0
            raise sd.CallbackStop()
        else:
            outdata[:] = data[:len(outdata)]
    except queue.Empty:
        outdata.fill(0)
        raise sd.CallbackStop()

# Start audio streaming thread
streaming_thread = threading.Thread(target=audio_stream_thread, daemon=True)
streaming_thread.start()

# Add sentence splitting to TTS class
def split_sentences(text, language):
    """Split text into sentences based on language"""
    if language in ['ZH', 'JP', 'KR']:
        # For Asian languages, split only on sentence endings and preserve them
        pattern = r'([。！？])'
        sentences = re.split(pattern, text)
        # Combine punctuation with previous sentence
        result = []
        for i in range(0, len(sentences)-1, 2):
            if i+1 < len(sentences):
                result.append(sentences[i] + sentences[i+1])
            else:
                result.append(sentences[i])
        return [s for s in result if s.strip()]
    else:
        # For other languages, split only on sentence endings
        # Look for: period/exclamation/question mark + space/newline
        # But don't split on common abbreviations like Mr., Dr., etc.
        common_abbrev = r'(?<!Mr)(?<!Mrs)(?<!Dr)(?<!Ms)(?<!Prof)(?<!Sr)(?<!Jr)(?<!vs)(?<!etc)'
        pattern = common_abbrev + r'[.!?][\s\n]+'
        sentences = re.split(pattern, text)
        # Clean up and remove empty sentences
        return [s.strip() for s in sentences if s.strip()]

# Modify TTS class to add streaming capability
for model in models.values():
    model.stream_tts = lambda text, speaker_id: model.tts_streaming(
        text, speaker_id, stream_audio_callback
    )

class RoundedSpinner(Spinner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.option_cls = SpinnerOption
        self.background_color = (0.95, 0.95, 0.95, 1)
        self.color = (0.2, 0.2, 0.2, 1)
        self.font_size = sp(18)
        self.bold = True
        self.background_normal = ''

class RoundedButton(Button):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_color = (0.2, 0.6, 1, 1)
        self.color = (1, 1, 1, 1)
        self.font_size = sp(18)
        self.bold = True
        self.height = dp(60)

class RoundedTextInput(TextInput):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_normal = ''
        self.background_active = ''
        self.background_color = (1, 1, 1, 1)
        self.foreground_color = (0, 0, 0, 1)
        self.cursor_color = (0, 0, 0, 1)
        self.font_size = sp(18)
        self.padding = [dp(20), dp(15)]
        self.halign = 'left'
        self.valign = 'top'
        self._text_color = (0, 0, 0, 1)
        self.text_color = (0, 0, 0, 1)

class ProgressLabel(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint_y = None
        self.height = dp(30)
        self.color = (1, 1, 1, 1)
        self.font_size = '14sp'
        self.halign = 'center'
        self.bold = True
        self.outline_width = 1
        self.outline_color = (0, 0, 0, 0.5)

class RoundedSlider(Slider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cursor_size = (dp(30), dp(30))
        self.background_width = dp(10)
        self.padding = dp(20)

class CustomProgressBar(ProgressBar):
    value = NumericProperty(0)  # Current playback position
    buffer_value = NumericProperty(0)  # Processed/buffered progress
    estimated_value = NumericProperty(0)  # Total estimated duration progress
    total_duration = NumericProperty(0)
    on_seek = ObjectProperty(None)  # Callback for seek events

    def set_value(self, value):
        Animation(value=value, duration=0.3).start(self)

    def set_buffer_value(self, value):
        Animation(buffer_value=value, duration=0.3).start(self)

    def set_estimated_value(self, value):
        Animation(estimated_value=value, duration=0.3).start(self)

    def set_total_duration(self, duration):
        self.total_duration = duration

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos) and self.on_seek:
            # Calculate the position as a percentage
            seek_pos = (touch.x - self.x) / self.width
            seek_pos = max(0, min(1, seek_pos))  # Clamp between 0 and 1
            
            # Convert to time position
            time_pos = seek_pos * self.total_duration
            
            # Call the seek callback
            self.on_seek(time_pos)
            return True
        return super().on_touch_down(touch)

class AudioProgressBar(CustomProgressBar):
    sound = ObjectProperty(None)
    _update_event = ObjectProperty(None)
    buffered_chunks = NumericProperty(0)
    total_chunks = NumericProperty(0)

    def on_sound(self, instance, value):
        if self._update_event:
            self._update_event.cancel()
        if value:
            self._update_event = Clock.schedule_interval(self._update_progress, 0.1)

    def _update_progress(self, dt):
        if self.sound:
            if self.sound.state == 'play':
                self.value = (self.sound.get_pos() / self.sound.length) * 100
            elif self.sound.state == 'stop':
                if self.sound.get_pos() == 0:
                    self.value = 0
                if self.sound.get_pos() >= self.sound.length - 0.1:
                    self.value = 100
        
        # Update buffer progress
        if self.total_chunks > 0:
            self.buffer_value = (self.buffered_chunks / self.total_chunks) * 100

    def update_buffer_progress(self, buffered, total):
        self.buffered_chunks = buffered
        self.total_chunks = total
        self._update_progress(0)

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self._seek_to_pos(touch.x)
            return True
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if self.collide_point(*touch.pos):
            self._seek_to_pos(touch.x)
            return True
        return super().on_touch_move(touch)

    def _seek_to_pos(self, x_pos):
        if self.sound:
            seek_pos_ratio = (x_pos - self.x) / self.width
            seek_pos_ratio = max(0, min(1, seek_pos_ratio))
            new_pos = seek_pos_ratio * self.sound.length
            self.sound.seek(new_pos)
            self.parent.parent.current_position = new_pos
            self.value = seek_pos_ratio * 100

class FileSelectButton(RoundedButton):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)
        self.width = dp(150)
        self.height = dp(60)
        self.font_size = '14sp'
        self.text = 'Select File'

class RemoveFileButton(RoundedButton):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text = 'Remove'
        self.size_hint = (None, None)
        self.width = dp(150)
        self.height = dp(60)
        self.font_size = '14sp'
        self.background_color = (0.9, 0.3, 0.3, 1)

class FileChooserPopup(Popup):
    def __init__(self, callback, **kwargs):
        super().__init__(**kwargs)
        self.callback = callback
        self.title = 'Select a Text File'
        self.size_hint = (0.9, 0.9)
        self.auto_dismiss = True

        layout = BoxLayout(orientation='vertical', spacing=dp(10))

        self.file_chooser = FileChooserListView(
            path=os.path.expanduser('~'),
            filters=['*.txt']
        )
        # Add double click binding
        self.file_chooser.bind(on_submit=self._on_submit)

        buttons = BoxLayout(
            size_hint_y=None,
            height=dp(50),
            spacing=dp(10)
        )

        cancel_btn = Button(
            text='Cancel',
            size_hint_x=None,
            width=dp(100)
        )
        select_btn = Button(
            text='Select',
            size_hint_x=None,
            width=dp(100)
        )

        cancel_btn.bind(on_release=self.dismiss)
        select_btn.bind(on_release=self._select_file)

        buttons.add_widget(Widget())
        buttons.add_widget(cancel_btn)
        buttons.add_widget(select_btn)
        buttons.add_widget(Widget())

        layout.add_widget(self.file_chooser)
        layout.add_widget(buttons)

        self.content = layout

    def _on_submit(self, instance, selection, touch=None, *args):
        # Called on double click
        # selection is a list of selected files
        # touch is the touch event that triggered the submission
        if selection:
            self._select_file(None)

    def _select_file(self, instance):
        if self.file_chooser.selection:
            selected_file = self.file_chooser.selection[0]
            self.dismiss()  # Dismiss the popup first
            Clock.schedule_once(lambda dt: self.callback(selected_file), 0)  # Call callback in next frame

class PlayPauseButton(RoundedButton):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)
        self.width = dp(70)
        self.height = dp(70)
        self.text = 'Play'
        self.font_size = '16sp'
        self.background_color = (0.2, 0.6, 1, 1)

class ControlButton(RoundedButton):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)
        self.width = dp(60)
        self.height = dp(60)
        self.font_size = '20sp'
        self.background_color = (0.2, 0.6, 1, 1)

class CloseButton(RoundedButton):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text = 'Close'
        self.size_hint = (None, None)
        self.width = dp(100)
        self.height = dp(50)
        self.background_color = (0.8, 0.2, 0.2, 1)  # Red color for close button

Builder.load_string('''
<RoundedSpinner>:
    canvas.before:
        Color:
            rgba: self.background_color
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [15]

<RoundedButton>:
    canvas.before:
        Color:
            rgba: self.background_color if self.state == 'normal' else (0.1, 0.5, 0.9, 1)
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [20]

<RoundedTextInput>:
    canvas.before:
        Color:
            rgba: (1, 1, 1, 1)
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [15]
        Color:
            rgba: (0.8, 0.8, 0.8, 1)
        Line:
            rounded_rectangle: (self.x, self.y, self.width, self.height, 15)
            width: 1

<RoundedSlider>:
    canvas.before:
        Color:
            rgba: 0.9, 0.9, 0.9, 1
        RoundedRectangle:
            pos: self.x, self.center_y - self.background_width/2
            size: self.width, self.background_width
            radius: [self.background_width/2]
    canvas:
        Color:
            rgba: 0.2, 0.6, 1, 1
        RoundedRectangle:
            pos: self.x, self.center_y - self.background_width/2
            size: self.value_pos[0] - self.x, self.background_width
            radius: [self.background_width/2]
        Color:
            rgba: 0.2, 0.6, 1, 1
        Ellipse:
            pos: self.value_pos[0] - self.cursor_size[0]/2, self.center_y - self.cursor_size[1]/2
            size: self.cursor_size

<CustomProgressBar>:
    canvas:
        Color:
            rgba: 0.9, 0.9, 0.9, 1  # Background color
        RoundedRectangle:
            pos: self.x, self.y
            size: self.width, self.height
            radius: [self.height/2]
        Color:
            rgba: 0.85, 0.85, 0.85, 1  # Total estimated duration color (lightest gray)
        RoundedRectangle:
            pos: self.x, self.y
            size: self.width * (self.estimated_value / 100.0), self.height
            radius: [self.height/2]
        Color:
            rgba: 0.7, 0.7, 0.7, 1  # Buffered progress color (darker gray)
        RoundedRectangle:
            pos: self.x, self.y
            size: self.width * (self.buffer_value / 100.0), self.height
            radius: [self.height/2]
        Color:
            rgba: 0.2, 0.6, 1, 1  # Playback progress color (blue)
        RoundedRectangle:
            pos: self.x, self.y
            size: self.width * (self.value / 100.0), self.height
            radius: [self.height/2]
''')

def count_words(text):
    """
    Count words in text more accurately:
    - Handles multiple spaces/newlines
    - Handles punctuation
    - Handles special characters
    - Counts hyphenated words as one word
    """
    # Remove extra whitespace and normalize
    text = ' '.join(text.split())
    
    # Count words, handling special cases
    words = text.split()
    word_count = 0
    
    for word in words:
        # Remove punctuation from word edges
        word = word.strip('.,!?()[]{}:;"\'/\\')
        
        # Skip if empty after cleaning
        if not word:
            continue
            
        # Count numbers as words
        if word.replace('.', '').replace(',', '').isdigit():
            word_count += 1
            continue
            
        # Count hyphenated words as one word
        if '-' in word:
            word_count += 1
            continue
            
        # Count regular words
        if any(c.isalpha() for c in word):
            word_count += 1
            
    return word_count

class MeloTTSUI(BoxLayout):
    current_sound = ObjectProperty(None, allownone=True)
    is_playing = BooleanProperty(False)
    selected_file = StringProperty('')
    current_position = NumericProperty(0)
    sound_length = NumericProperty(0)
    is_processing = BooleanProperty(False)
    temp_files = []
    _update_event = None
    file_chooser_popup = None
    current_sentence_index = NumericProperty(0)
    total_sentences = NumericProperty(0)
    current_channel = 0
    should_stop = False
    playback_position = 0  # Total elapsed time in seconds
    last_update_time = 0   # Last time we updated position

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.should_stop = False
        self.audio_player = AudioPlayer()
        self.default_speaker = 'EN-BR'  # Set British English as default
        self.default_speed = 1.0  # Set default speed
        
        # Set minimum window size and default size
        Window.minimum_width = dp(200)
        Window.minimum_height = dp(300)
        Window.size = (dp(200), dp(300))
        
        # Initialize update event
        self._update_event = None
        self.start_update_event()
        
        # Create a ScrollView
        scroll_view = ScrollView(
            do_scroll_x=False,
            do_scroll_y=True,
            size_hint=(1, 1),
            bar_width=dp(10),
            scroll_type=['bars', 'content']
        )
        
        # Main content layout
        main_layout = BoxLayout(
            orientation='vertical',
            padding=[dp(20), dp(40)],
            spacing=dp(20),
            size_hint_y=None
        )
        
        # Bind the layout height to its minimum height for proper scrolling
        main_layout.bind(minimum_height=main_layout.setter('height'))

        # File selection
        file_box = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            height=dp(120),
            spacing=dp(10)
        )

        self.file_label = Label(
            text='No file selected',
            size_hint_y=None,
            height=dp(50),
            color=(1, 1, 1, 1),
            font_size=sp(18)
        )
        file_box.add_widget(self.file_label)

        buttons_box = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(60),
            spacing=dp(15)
        )

        self.file_btn = FileSelectButton()
        self.file_btn.bind(on_press=self.show_file_chooser)
        
        self.remove_btn = RemoveFileButton()
        self.remove_btn.bind(on_press=self.remove_file)
        self.remove_btn.disabled = True
        self.remove_btn.opacity = 0.5

        buttons_box.add_widget(Widget())  # Spacer
        buttons_box.add_widget(self.file_btn)
        buttons_box.add_widget(self.remove_btn)
        buttons_box.add_widget(Widget())  # Spacer

        file_box.add_widget(buttons_box)
        main_layout.add_widget(file_box)

        # Synthesize button
        self.synthesize_btn = RoundedButton(
            text='Generate',
            size_hint_y=None,
            height=dp(70)
        )
        self.synthesize_btn.bind(on_press=self.synthesize_audio)
        main_layout.add_widget(self.synthesize_btn)

        # Controls box with forward/backward buttons
        controls_box = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(80),
            spacing=dp(15)
        )
        
        self.backward_btn = ControlButton(text='<--15s')
        self.backward_btn.bind(on_press=self.seek_backward)
        
        self.play_pause_btn = PlayPauseButton()
        self.play_pause_btn.bind(on_press=self.toggle_play_pause)
        
        self.forward_btn = ControlButton(text='15s-->')
        self.forward_btn.bind(on_press=self.seek_forward)
        
        controls_box.add_widget(Widget())  # Spacer
        controls_box.add_widget(self.backward_btn)
        controls_box.add_widget(self.play_pause_btn)
        controls_box.add_widget(self.forward_btn)
        controls_box.add_widget(Widget())  # Spacer
        
        main_layout.add_widget(controls_box)

        # Progress bar and time labels
        progress_box = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            height=dp(60),
            spacing=dp(5)
        )

        self.progress_bar = CustomProgressBar(
            max=100,
            value=0,
            size_hint_y=None,
            height=dp(30)
        )
        # Bind the seek callback
        self.progress_bar.on_seek = self.on_progress_seek

        time_box = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(20)
        )

        self.current_time = Label(
            text='0:00',
            size_hint_x=None,
            width=dp(60),
            color=(1, 1, 1, 1),
            font_size=sp(16)
        )
        self.total_time = Label(
            text='0:00',
            size_hint_x=None,
            width=dp(60),
            color=(1, 1, 1, 1),
            font_size=sp(16)
        )

        time_box.add_widget(self.current_time)
        time_box.add_widget(Widget())
        time_box.add_widget(self.total_time)

        progress_box.add_widget(self.progress_bar)
        progress_box.add_widget(time_box)
        main_layout.add_widget(progress_box)

        # Status label
        self.status_label = Label(
            text='Ready',
            size_hint_y=None,
            height=dp(40),
            color=(1, 1, 1, 1),
            font_size=sp(18)
        )
        main_layout.add_widget(self.status_label)

        # Add close button at the bottom
        close_box = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(60),
            spacing=dp(15)
        )
        
        self.close_btn = CloseButton()
        self.close_btn.bind(on_press=self.close_application)
        
        close_box.add_widget(Widget())  # Spacer
        close_box.add_widget(self.close_btn)
        close_box.add_widget(Widget())  # Spacer
        
        main_layout.add_widget(close_box)

        # Add padding at the bottom
        main_layout.add_widget(Widget(size_hint_y=None, height=dp(20)))

        # Add the main layout to the scroll view
        scroll_view.add_widget(main_layout)
        
        # Add the scroll view to the root layout
        self.add_widget(scroll_view)

    def start_update_event(self):
        """Start or restart the update event for progress bar and time labels"""
        if self._update_event:
            self._update_event.cancel()
        self._update_event = Clock.schedule_interval(self.update_playback, 0.1)
        Clock.schedule_interval(self.update_time_labels, 0.1)

    def update_playback(self, dt):
        if not self.temp_files:
            return
            
        try:
            if self.is_playing:
                current_channel = pygame.mixer.Channel(self.current_channel)
                if current_channel.get_busy():
                    # Calculate total position including all previous chunks
                    total_pos = sum(pygame.mixer.Sound(f).get_length() 
                                  for f in self.temp_files[:self.current_sentence_index])
                    
                    # Add current chunk position based on current sound length and busy state
                    current_sound = pygame.mixer.Sound(self.temp_files[self.current_sentence_index])
                    current_length = current_sound.get_length()
                    
                    # Estimate current position based on time elapsed since playback started
                    if not hasattr(self, 'playback_start_time'):
                        self.playback_start_time = time.time()
                    elapsed = time.time() - self.playback_start_time
                    current_pos = min(elapsed, current_length)
                    
                    total_pos += current_pos
                    
                    # Calculate total duration
                    total_duration = sum(pygame.mixer.Sound(f).get_length() 
                                      for f in self.temp_files)
                    
                    # Update progress
                    if total_duration > 0:
                        self.progress_bar.value = (total_pos / total_duration) * 100
                        self.current_time.text = self.format_time(total_pos)
                        self.total_time.text = self.format_time(total_duration)
                        
        except Exception as e:
            print(f"Error updating playback: {e}")
            
    def update_time_labels(self, dt):
        if self.is_playing:
            current_channel = pygame.mixer.Channel(self.current_channel)
            if current_channel.get_busy():
                # Calculate total position including all previous chunks
                total_pos = sum(pygame.mixer.Sound(f).get_length() 
                              for f in self.temp_files[:self.current_sentence_index])
                
                # Add current chunk position based on time elapsed
                if hasattr(self, 'playback_start_time'):
                    elapsed = time.time() - self.playback_start_time
                    current_sound = pygame.mixer.Sound(self.temp_files[self.current_sentence_index])
                    current_pos = min(elapsed, current_sound.get_length())
                    total_pos += current_pos
                
                self.current_time.text = self.format_time(total_pos)
                
                total_duration = sum(pygame.mixer.Sound(f).get_length() 
                                  for f in self.temp_files)
                self.total_time.text = self.format_time(total_duration)

    def start_playback(self):
        """Start playing from the beginning"""
        if not self.temp_files:
            return
            
        # Reset playback state
        self.playback_position = 0
        self.last_update_time = time.time()
        self.current_sentence_index = 0
        
        # Start playback
        self.is_playing = True
        self.play_pause_btn.text = 'Pause'
        self.status_label.text = 'Playing'
        self.play_next_sentence()
        
        # Start progress updates
        Clock.schedule_interval(self.check_playback, 0.1)

    def play_next_sentence(self):
        """Play the next sentence in the queue"""
        if not self.temp_files or self.current_sentence_index >= len(self.temp_files):
            # Reset playback state when we reach the end
            self.is_playing = False
            self.play_pause_btn.text = 'Play'
            self.current_sentence_index = 0
            self.playback_position = 0
            self.last_update_time = 0
            return

        try:
            # Stop any currently playing audio on both channels
            pygame.mixer.Channel(self.current_channel).stop()
            pygame.mixer.Channel((self.current_channel + 1) % 2).stop()
            
            # Load and play the current sentence
            current_channel = pygame.mixer.Channel(self.current_channel)
            current_sound = pygame.mixer.Sound(self.temp_files[self.current_sentence_index])
            
            # Set up callback for when sound finishes
            def sound_finished():
                if self.is_playing and self.current_sentence_index < len(self.temp_files) - 1:
                    self.current_sentence_index += 1
                    Clock.schedule_once(lambda dt: self.play_next_sentence(), 0)
            
            current_channel.set_endevent(pygame.USEREVENT)
            current_channel.play(current_sound)
            
            # Update timing
            self.last_update_time = time.time()
            
            # Pre-load next sentence
            next_index = self.current_sentence_index + 1
            if next_index < len(self.temp_files):
                pygame.mixer.Sound(self.temp_files[next_index])
                
        except Exception as e:
            print(f"Error playing sentence: {str(e)}")

    def check_playback(self, dt):
        """Update playback progress"""
        if not self.temp_files:
            return True
            
        # Get current position and total duration
        position = self.audio_player.get_position()
        total_duration = self.audio_player.get_duration()  # Use estimated total duration
        
        # Update progress even when paused
        if total_duration > 0:
            # Scale progress against total estimated duration
            progress = (position / total_duration) * 100
            self.progress_bar.value = min(progress, 100)
            
            # Update time labels
            self.current_time.text = self.format_time(position)
            self.total_time.text = self.format_time(total_duration)
                
        return True

    def show_file_chooser(self, instance):
        self.file_chooser_popup = FileChooserPopup(callback=self.on_file_selected)
        self.file_chooser_popup.open()

    def on_file_selected(self, filepath):
        try:
            # Read the file content
            with open(filepath, 'r', encoding='utf-8') as f:
                file_content = f.read().strip()
            
            if not file_content:
                self.status_label.text = 'Selected file is empty'
                return
            
            self.selected_file = filepath
            filename = os.path.basename(filepath)
            if len(filename) > 30:
                filename = filename[:27] + "..."
            self.file_label.text = filename
            
            # Trigger synthesis automatically
            self.synthesize_audio(None)
            
        except UnicodeDecodeError:
            self.status_label.text = 'Error: File must be a valid text file'
        except Exception as e:
            self.status_label.text = f'Error reading file: {str(e)}'

    def remove_file(self, instance):
        self.selected_file = ''
        self.file_label.text = 'No file selected'
        # Disable and hide remove button
        self.remove_btn.disabled = True
        self.remove_btn.opacity = 0.5
        # Stop any playing audio
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        self.is_playing = False
        self.progress_bar.value = 0
        self.current_time.text = '0:00'
        self.total_time.text = '0:00'

    def get_text_to_synthesize(self):
        """Get text from selected file"""
        if not self.selected_file:
            self.status_label.text = 'Please select a file'
            return None
            
        try:
            with open(self.selected_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            if not text:
                self.status_label.text = 'Selected file is empty'
                return None
            return text
        except Exception as e:
            self.status_label.text = f'Error reading file: {str(e)}'
            return None

    def synthesize_audio(self, instance):
        try:
            text = self.get_text_to_synthesize()
            if not text:
                return

            # Calculate initial word count using the robust counter
            total_words = count_words(text)
            print(f"Total words in text: {total_words}")  # Debug output
            
            # Calculate initial duration estimate (in seconds)
            initial_estimate = (total_words / 165.0) * 60  # 165 words per minute
            print(f"Initial duration estimate: {initial_estimate:.2f} seconds ({initial_estimate/60:.2f} minutes)")
            
            # Set initial values in audio player
            self.audio_player.remaining_words = total_words
            self.audio_player.estimated_total_duration = initial_estimate
            
            # Set initial total duration and show full estimated length
            self.progress_bar.set_total_duration(initial_estimate)
            self.progress_bar.set_estimated_value(100)  # Show full estimated length
            self.progress_bar.set_buffer_value(0)  # Reset buffer progress
            self.progress_bar.set_value(0)  # Reset playback progress
            self.total_time.text = self.format_time(initial_estimate)

            self.status_label.text = 'Initializing...'
            self.synthesize_btn.disabled = True
            
            # Reset stop flag
            self.should_stop = False
            
            # Stop any current playback
            self.audio_player.stop()
            self.is_playing = False
            self.current_sentence_index = 0
            
            # Clean up previous temp files
            for temp_file in self.temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
            self.temp_files = []
            
            # Split text into sentences
            sentences = split_sentences(text, 'EN')
            self.total_sentences = len(sentences)
            self.processed_sentences = 0

            def process_sentence(sentence):
                if self.should_stop:
                    return
                    
                try:
                    # Generate audio for the sentence
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    models['EN'].tts_to_file(
                        sentence,
                        models['EN'].hps.data.spk2id[self.default_speaker],
                        temp_file.name,
                        speed=self.default_speed  # Use default speed
                    )
                    
                    if self.should_stop:
                        os.unlink(temp_file.name)
                        return
                        
                    # Add to temp files list
                    self.temp_files.append(temp_file.name)
                    
                    # Update progress and processed duration
                    self.processed_sentences += 1
                    
                    # Calculate remaining words from unprocessed sentences using robust counter
                    remaining_words = sum(count_words(s) for s in sentences[self.processed_sentences:])
                    
                    # Update duration estimates
                    total_duration = self.audio_player.update_duration_estimate(remaining_words)
                    processed_duration = self.audio_player.get_processed_duration()
                    
                    # Debug output
                    print(f"Processed duration: {processed_duration:.2f}s, Remaining words: {remaining_words}")
                    print(f"Total estimate: {total_duration:.2f}s ({total_duration/60:.2f} minutes)")
                    
                    # Update progress bar components
                    if total_duration > 0:
                        # Keep the total duration at the initial estimate
                        total_duration = max(total_duration, self.audio_player.estimated_total_duration)
                        
                        # Update total duration
                        Clock.schedule_once(lambda dt: self.progress_bar.set_total_duration(total_duration))
                        
                        # Show full estimated length
                        Clock.schedule_once(lambda dt: self.progress_bar.set_estimated_value(100))
                        
                        # Update buffer progress (processed chunks)
                        buffer_progress = (processed_duration / total_duration) * 100
                        Clock.schedule_once(lambda dt: self.progress_bar.set_buffer_value(buffer_progress))
                        
                        # Update playback progress
                        if self.is_playing:
                            playback_progress = (self.audio_player.get_position() / total_duration) * 100
                            Clock.schedule_once(lambda dt: self.progress_bar.set_value(playback_progress))
                        
                        # Update time labels
                        Clock.schedule_once(lambda dt: setattr(self.total_time, 'text', self.format_time(total_duration)))
                    
                    # Start playback if this is the first sentence
                    if len(self.temp_files) == 1 and not self.should_stop:
                        Clock.schedule_once(lambda dt: self.start_playback())
                        
                except Exception as e:
                    print(f"Error processing sentence: {str(e)}")

            def generate_audio():
                try:
                    # Split text into sentences
                    texts = split_sentences(text, 'EN')
                    if not texts:
                        return
                    
                    # Process sentences one by one
                    for sentence in texts:
                        if self.should_stop:
                            break
                        process_sentence(sentence)
                        
                    if not self.should_stop:
                        self.status_label.text = 'Ready'
                    self.synthesize_btn.disabled = False
                    
                except Exception as e:
                    print(f"Error generating audio: {str(e)}")
                    self.status_label.text = 'Error'
                    self.synthesize_btn.disabled = False

            # Start audio generation in a separate thread
            threading.Thread(target=generate_audio, daemon=True).start()
            
        except Exception as e:
            print(f"Error in synthesize_audio: {str(e)}")
            self.status_label.text = 'Error'
            self.synthesize_btn.disabled = False

    def toggle_play_pause(self, instance):
        """Toggle between play and pause states"""
        if not self.temp_files:
            return
            
        if self.audio_player.is_playing:
            self.audio_player.pause()
            self.play_pause_btn.text = 'Play'
            self.status_label.text = 'Paused'
        else:
            # If we have a paused position, resume from there
            if hasattr(self.audio_player, 'paused_position') and self.audio_player.paused_position > 0:
                if self.audio_player.resume():
                    self.play_pause_btn.text = 'Pause'
                    self.status_label.text = 'Playing'
            else:
                # Start from beginning
                if self.audio_player.play(self.temp_files):
                    self.play_pause_btn.text = 'Pause'
                    self.status_label.text = 'Playing'

    def start_playback(self):
        """Start playing from the beginning"""
        if not self.temp_files:
            return
            
        # Start playback with all files
        if self.audio_player.play(self.temp_files):
            self.is_playing = True
            self.play_pause_btn.text = 'Pause'
            self.status_label.text = 'Playing'
            
            # Start progress updates
            Clock.schedule_interval(self.check_playback, 0.1)

    def cleanup(self):
        """Clean up resources before closing"""
        self.should_stop = True
        
        # Stop playback
        pygame.mixer.Channel(self.current_channel).stop()
        pygame.mixer.Channel((self.current_channel + 1) % 2).stop()
        self.is_playing = False
        
        # Cancel any scheduled events
        if self._update_event:
            self._update_event.cancel()
            
        # Clean up temp files
        for temp_file in self.temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        self.temp_files = []

    def format_time(self, seconds):
        if not seconds or seconds < 0:
            return '0:00'
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f'{minutes}:{seconds:02d}'

    def seek_forward(self, instance):
        """Seek forward by 15 seconds"""
        if not self.temp_files or not self.audio_player.files:
            return
            
        current_pos = self.audio_player.get_position()
        processed_duration = self.audio_player.get_processed_duration()
        
        # Calculate maximum safe seek position (15 seconds before processed duration)
        max_safe_seek = max(0, processed_duration - 15.0)
        
        # Calculate target position and clamp it
        target_pos = min(current_pos + 15.0, max_safe_seek)
        
        # Store playing state
        was_playing = self.audio_player.is_playing
        
        # Seek to new position
        self.audio_player.seek(target_pos)
        
        # Update progress bar and time labels immediately
        total_duration = self.audio_player.get_duration()
        if total_duration > 0:
            self.progress_bar.value = (target_pos / total_duration) * 100
            self.current_time.text = self.format_time(target_pos)
            self.total_time.text = self.format_time(total_duration)
        
        # Resume if was playing
        if was_playing:
            self.audio_player.resume()
            
        # Show feedback if we hit the safe seek limit
        if target_pos == max_safe_seek:
            self.status_label.text = 'Processing more audio...'
            Clock.schedule_once(lambda dt: setattr(self.status_label, 'text', 'Playing' if self.is_playing else 'Paused'), 1.5)

    def seek_backward(self, instance):
        """Seek backward by 15 seconds"""
        if not self.temp_files or not self.audio_player.files:
            return
            
        current_pos = self.audio_player.get_position()
        target_pos = max(0, current_pos - 15.0)
        
        # Store playing state
        was_playing = self.audio_player.is_playing
        
        # Seek to new position
        self.audio_player.seek(target_pos)
        
        # Update progress bar and time labels immediately
        total_duration = self.audio_player.get_duration()
        if total_duration > 0:
            self.progress_bar.value = (target_pos / total_duration) * 100
            self.current_time.text = self.format_time(target_pos)
            self.total_time.text = self.format_time(total_duration)
        
        # Resume if was playing
        if was_playing:
            self.audio_player.resume()

    def on_progress_seek(self, seek_time):
        """Handle seek events from progress bar"""
        if not self.temp_files or not self.audio_player.files:
            return
            
        # Get the processed duration (buffered audio)
        processed_duration = self.audio_player.get_processed_duration()
        
        # Calculate maximum safe seek position (15 seconds before processed duration)
        max_safe_seek = max(0, processed_duration - 15.0)
        
        # Clamp the seek position to safe range
        target_pos = min(seek_time, max_safe_seek)
        
        # Store playing state
        was_playing = self.audio_player.is_playing
        
        # Seek to new position
        self.audio_player.seek(target_pos)
        
        # Update progress bar and time labels immediately
        total_duration = self.audio_player.get_duration()
        if total_duration > 0:
            progress = (target_pos / total_duration) * 100
            self.progress_bar.set_value(progress)
            self.current_time.text = self.format_time(target_pos)
            
        # Show feedback if we hit the safe seek limit
        if target_pos == max_safe_seek and seek_time > max_safe_seek:
            self.status_label.text = 'Processing more audio...'
            Clock.schedule_once(lambda dt: setattr(self.status_label, 'text', 
                'Playing' if self.is_playing else 'Paused'), 1.5)
        
        # Resume if was playing
        if was_playing:
            self.audio_player.resume()

    def close_application(self, instance):
        """Properly close the application"""
        try:
            # Stop any playing audio
            if self.audio_player:
                self.audio_player.stop()
            
            # Cancel any scheduled events
            if self._update_event:
                self._update_event.cancel()
            
            # Clean up temp files
            self.cleanup()
            
            # Stop pygame mixer
            pygame.mixer.quit()
            
            # Close the application
            App.get_running_app().stop()
        except Exception as e:
            print(f"Error during cleanup: {e}")
            # Force quit if cleanup fails
            App.get_running_app().stop()

class MeloTTSApp(App):
    def build(self):
        self.ui = MeloTTSUI()
        # Bind the close event to the cleanup method
        Window.bind(on_request_close=self.on_request_close)
        return self.ui

    def on_request_close(self, *args):
        """Handle window close button click"""
        self.ui.close_application(None)
        return True

    def on_stop(self):
        """Called when the application is about to close"""
        if hasattr(self, 'ui'):
            self.ui.cleanup()
        pygame.mixer.quit()

if __name__ == '__main__':
    # --- START: Memory Limiting Code ---
    import sys
    # This code only works on Linux and macOS
    if sys.platform.startswith('linux') or sys.platform == 'darwin':
        import resource

        def limit_memory(max_bytes):
            """
            Set the maximum memory usage for the current process.

            This function is platform-specific and works on Unix-like systems.
            """
            try:
                # Set the soft and hard limits for the address space
                soft, hard = resource.getrlimit(resource.RLIMIT_AS)
                print(f"Current memory limits: Soft={soft / 1024**2:.2f}MB, Hard={hard / 1024**2:.2f}MB")
                resource.setrlimit(resource.RLIMIT_AS, (max_bytes, hard))
                print(f"New memory limit set to: {max_bytes / 1024**3:.2f}GB")
            except (ImportError, ValueError, resource.error) as e:
                print(f"Warning: Could not set memory limit. {e}")

        # Set the memory limit to 2 GB (2 * 1024 * 1024 * 1024 bytes).
        limit_memory(1 * 1024 * 1024 * 1024)

    else:
        print("Warning: Memory limiting is not supported on this OS (e.g., Windows).")
    # --- END: Memory Limiting Code ---


    print("Make sure you've downloaded unidic (python -m unidic download) for this app to work.")
    MeloTTSApp().run()