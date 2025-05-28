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

# Suppress all logging output
logging.getLogger().setLevel(logging.ERROR)
os.environ['KIVY_NO_CONSOLELOG'] = '1'
os.environ['NUMBA_WARNINGS'] = '0'
os.environ['NUMBA_DISABLE_JIT'] = '0'

# Audio stream configuration
CHUNK_SIZE = 4096
AUDIO_QUEUE = queue.Queue()
STREAM = None
TOTAL_CHUNKS = 0
BUFFERED_CHUNKS = 0
PROGRESS_CALLBACK = None
STREAM_ERROR = None

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
    'ES': TTS(language='ES', device=device),
    'FR': TTS(language='FR', device=device),
    'ZH': TTS(language='ZH', device=device),
    'JP': TTS(language='JP', device=device),
    'KR': TTS(language='KR', device=device),
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

default_text_dict = {
    'EN': 'The field of text-to-speech has seen rapid development recently.',
    'ES': 'El campo de la conversión de texto a voz ha experimentado un rápido desarrollo recientemente.',
    'FR': 'Le domaine de la synthèse vocale a connu un développement rapide récemment',
    'ZH': 'text-to-speech 领域近年来发展迅速',
    'JP': 'テキスト読み上げの分野は最近急速な発展を遂げています',
    'KR': '최근 텍스트 음성 변환 분야가 급속도로 발전하고 있습니다.',
}

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
    value = NumericProperty(0)
    buffer_value = NumericProperty(0)

    def set_value(self, value):
        Animation(value=value, duration=0.3).start(self)

    def set_buffer_value(self, value):
        Animation(buffer_value=value, duration=0.3).start(self)

class PlaybackButton(RoundedButton):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)
        self.width = dp(70)
        self.height = dp(70)
        self.font_size = '14sp'

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
            rgba: 0.6, 0.6, 0.6, 1  # Buffer color (gray)
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Set minimum window size and default size
        Window.minimum_width = dp(400)
        Window.minimum_height = dp(800)
        Window.size = (dp(500), dp(900))
        
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

        # Title and Model Selection
        header_box = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(70),
            spacing=dp(15),
            padding=[dp(20), 0]
        )

        # Model Selection
        self.model_spinner = RoundedSpinner(
            text='MeloTTS',
            values=('MeloTTS', 'Verbadik (Pro Version)'),
            size_hint=(None, None),
            width=dp(250),
            height=dp(60)
        )
        self.model_spinner.bind(text=self.on_model_change)

        header_box.add_widget(Widget())
        header_box.add_widget(self.model_spinner)
        header_box.add_widget(Widget())

        main_layout.add_widget(header_box)

        # Language selection
        self.language_spinner = RoundedSpinner(
            text='EN',
            values=('EN', 'ES', 'FR', 'ZH', 'JP', 'KR'),
            size_hint_y=None,
            height=dp(60)
        )
        self.language_spinner.bind(text=self.on_language_change)
        main_layout.add_widget(self.language_spinner)

        # Speaker selection
        self.speaker_spinner = RoundedSpinner(
            text='EN-US',
            values=list(models['EN'].hps.data.spk2id.keys()),
            size_hint_y=None,
            height=dp(60)
        )
        main_layout.add_widget(self.speaker_spinner)

        # Speed slider and layout
        speed_layout = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(60)
        )
        
        speed_label = Label(
            text='Speed:',
            size_hint_x=0.3,
            font_size=sp(18)
        )
        
        self.speed_slider = RoundedSlider(
            min=0.3,
            max=3.0,
            value=1.0,
            step=0.1,
            cursor_size=(dp(30), dp(30))
        )
        self.speed_slider.bind(value=self.on_speed_change)
        
        self.speed_value_label = Label(
            text='1.0',
            size_hint_x=0.2,
            font_size=sp(18)
        )
        
        speed_layout.add_widget(speed_label)
        speed_layout.add_widget(self.speed_slider)
        speed_layout.add_widget(self.speed_value_label)
        
        main_layout.add_widget(speed_layout)

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

        # Text input
        text_label = Label(
            text='Or enter text directly:',
            size_hint_y=None,
            height=dp(40),
            color=(1, 1, 1, 1),
            font_size=sp(18),
            halign='left'
        )
        text_label.bind(size=text_label.setter('text_size'))
        main_layout.add_widget(text_label)

        self.text_input = RoundedTextInput(
            text=default_text_dict['EN'],
            multiline=True,
            size_hint_y=None,
            height=dp(150)
        )
        main_layout.add_widget(self.text_input)

        # Synthesize button
        self.synthesize_btn = RoundedButton(
            text='Generate',
            size_hint_y=None,
            height=dp(70)
        )
        self.synthesize_btn.bind(on_press=self.synthesize_audio)
        main_layout.add_widget(self.synthesize_btn)

        # Audio controls
        controls_box = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(80),
            spacing=dp(15)
        )

        self.backward_btn = PlaybackButton(
            text='<<',
            on_press=self.seek_backward
        )
        self.play_pause_btn = PlaybackButton(
            text='Play',
            on_press=self.toggle_playback
        )
        self.forward_btn = PlaybackButton(
            text='>>',
            on_press=self.seek_forward
        )

        controls_box.add_widget(Widget())
        controls_box.add_widget(self.backward_btn)
        controls_box.add_widget(self.play_pause_btn)
        controls_box.add_widget(self.forward_btn)
        controls_box.add_widget(Widget())

        main_layout.add_widget(controls_box)

        # Progress bar and time labels
        progress_box = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            height=dp(60),
            spacing=dp(5)
        )

        self.progress_bar = AudioProgressBar(
            max=100,
            value=0,
            size_hint_y=None,
            height=dp(30)
        )

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
                # Calculate total position including all previous chunks
                current_pos = pygame.mixer.music.get_pos() / 1000.0
                if current_pos >= 0:
                    total_pos = sum(pygame.mixer.Sound(f).get_length() 
                                  for f in self.temp_files[:self.current_sentence_index])
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
            current = pygame.mixer.music.get_pos() / 1000.0
            self.current_time.text = self.format_time(current)
            self.total_time.text = self.format_time(self.sound_length)

    def seek_to_position(self, position):
        """Seek to a specific position in the audio"""
        if not self.temp_files:
            return

        try:
            # Find the appropriate chunk for the target position
            total_duration = 0
            target_chunk = 0
            chunk_start_times = [0]  # Start time of each chunk
            
            # Calculate duration of each chunk and find target chunk
            for i, temp_file in enumerate(self.temp_files):
                sound = pygame.mixer.Sound(temp_file)
                duration = sound.get_length()
                total_duration += duration
                if i < len(self.temp_files) - 1:
                    chunk_start_times.append(total_duration)
            
            # Find which chunk contains our target position
            for i, start_time in enumerate(chunk_start_times):
                if start_time <= position < (chunk_start_times[i + 1] if i + 1 < len(chunk_start_times) else total_duration):
                    target_chunk = i
                    break
            
            # If seeking beyond processed chunks, go to last completed chunk
            if target_chunk >= len(self.temp_files):
                target_chunk = len(self.temp_files) - 1
                position = chunk_start_times[target_chunk]
            
            # Calculate position within the chunk
            chunk_position = position - chunk_start_times[target_chunk]
            
            # Update playback state
            was_playing = self.is_playing
            if was_playing:
                pygame.mixer.pause()
            
            # Load and seek to position in target chunk
            self.current_sentence_index = target_chunk
            sound = pygame.mixer.Sound(self.temp_files[target_chunk])
            channel = pygame.mixer.Channel(self.current_channel)
            channel.play(sound)
            
            # Seek within the chunk
            pygame.mixer.music.set_pos(chunk_position)
            
            if not was_playing:
                pygame.mixer.pause()
            else:
                self.is_playing = True
                self.play_pause_btn.text = 'Pause'
            
            # Update UI
            self.progress_bar.value = (position / total_duration) * 100 if total_duration > 0 else 0
            self.current_time.text = self.format_time(position)
            self.total_time.text = self.format_time(total_duration)
            
        except Exception as e:
            print(f"Error seeking: {e}")
            self.status_label.text = 'Seek error'

    def seek_forward(self, instance):
        if not self.temp_files:
            return
            
        try:
            current_pos = pygame.mixer.music.get_pos() / 1000.0
            if current_pos < 0:
                current_pos = 0
                
            # Add the current chunk's start time
            current_pos += sum(pygame.mixer.Sound(f).get_length() 
                             for f in self.temp_files[:self.current_sentence_index])
            
            # Seek 5 seconds forward
            new_pos = current_pos + 5
            self.seek_to_position(new_pos)
            
        except Exception as e:
            print(f"Error seeking forward: {e}")
            self.status_label.text = 'Seek error'

    def seek_backward(self, instance):
        if not self.temp_files:
            return
            
        try:
            current_pos = pygame.mixer.music.get_pos() / 1000.0
            if current_pos < 0:
                current_pos = 0
                
            # Add the current chunk's start time
            current_pos += sum(pygame.mixer.Sound(f).get_length() 
                             for f in self.temp_files[:self.current_sentence_index])
            
            # Seek 5 seconds backward
            new_pos = max(0, current_pos - 5)
            self.seek_to_position(new_pos)
            
        except Exception as e:
            print(f"Error seeking backward: {e}")
            self.status_label.text = 'Seek error'

    def toggle_playback(self, instance):
        if not self.temp_files:
            return

        try:
            if self.is_playing:
                pygame.mixer.pause()
                self.play_pause_btn.text = 'Play'
                self.is_playing = False
            else:
                self.is_playing = True
                self.play_pause_btn.text = 'Pause'
                
                # If we're at the end, start from beginning
                if self.current_sentence_index >= len(self.temp_files):
                    self.current_sentence_index = 0
                
                # Resume from current position
                pygame.mixer.unpause()
                if not pygame.mixer.get_busy():
                    self.play_next_sentence()
                    
        except Exception as e:
            print(f"Error toggling playback: {e}")
            self.status_label.text = 'Playback error'
            self.is_playing = False

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
            
            self.text_input.text = file_content
            self.remove_btn.disabled = False
            self.remove_btn.opacity = 1
            
            # Trigger synthesis automatically
            self.synthesize_audio(None)
            
        except UnicodeDecodeError:
            self.status_label.text = 'Error: File must be a valid text file'
        except Exception as e:
            self.status_label.text = f'Error reading file: {str(e)}'

    def remove_file(self, instance):
        self.selected_file = ''
        self.file_label.text = 'No file selected'
        # Restore default text
        self.text_input.text = default_text_dict[self.language_spinner.text]
        # Disable and hide remove button
        self.remove_btn.disabled = True
        self.remove_btn.opacity = 0.5
        # Stop any playing audio
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        self.is_playing = False
        self.play_pause_btn.text = 'Play'
        self.progress_bar.value = 0
        self.current_time.text = '0:00'
        self.total_time.text = '0:00'

    def get_text_to_synthesize(self):
        text = self.text_input.text.strip()
        if not text:
            self.status_label.text = 'Please enter text or select a file'
            return None
        return text

    def play_next_sentence(self):
        """Play the next sentence in the queue"""
        if not self.temp_files or self.current_sentence_index >= len(self.temp_files):
            # Reset playback state when we reach the end
            self.is_playing = False
            self.play_pause_btn.text = 'Play'
            self.current_sentence_index = 0
            return

        try:
            # Load and play the next sentence
            sound = pygame.mixer.Sound(self.temp_files[self.current_sentence_index])
            channel = pygame.mixer.Channel(self.current_channel)
            
            # Set up callback for when this sentence finishes
            def sentence_finished():
                if self.current_sentence_index < len(self.temp_files) - 1:
                    self.current_sentence_index += 1
                    Clock.schedule_once(lambda dt: self.play_next_sentence(), 0)
                else:
                    # Last sentence finished
                    self.is_playing = False
                    self.play_pause_btn.text = 'Play'
                    self.current_sentence_index = 0
            
            # Play the sound and set up the callback
            channel.play(sound)
            channel.set_endevent(pygame.USEREVENT + self.current_channel)
            
            # Switch channels for next sentence (ping-pong between 0 and 1)
            self.current_channel = 1 - self.current_channel
            
            # Schedule checking for sentence completion
            def check_completion(dt):
                if not channel.get_busy():
                    sentence_finished()
                    return False  # Stop the schedule
                return True  # Continue checking
            
            # Start checking completion with a shorter interval
            Clock.schedule_interval(check_completion, 0.05)
            
        except Exception as e:
            print(f"Error playing sentence {self.current_sentence_index}: {e}")
            self.status_label.text = 'Playback error'
            self.is_playing = False
            self.play_pause_btn.text = 'Play'

    def synthesize_audio(self, instance):
        try:
            text = self.get_text_to_synthesize()
            if not text:
                return

            self.status_label.text = 'Initializing...'
            self.progress_bar.value = 0
            self.progress_bar.buffer_value = 0
            self.synthesize_btn.disabled = True
            
            # Stop any currently playing audio
            pygame.mixer.stop()
            self.is_playing = False
            self.play_pause_btn.text = 'Play'
            self.current_sentence_index = 0
            
            # Clean up previous temp files
            for temp_file in self.temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
            self.temp_files = []

            def generate_audio():
                try:
                    language = self.language_spinner.text
                    speaker = self.speaker_spinner.text
                    speed = self.speed_slider.value
                    
                    # Split text into sentences
                    sentences = split_sentences(text, language)
                    if not sentences:
                        Clock.schedule_once(lambda dt: setattr(self.status_label, 'text', 'No text to process'), 0)
                        Clock.schedule_once(lambda dt: setattr(self.synthesize_btn, 'disabled', False), 0)
                        return
                        
                    self.total_sentences = len(sentences)
                    processed_sentences = []
                    
                    def process_sentence(index):
                        if index >= len(sentences):
                            return
                            
                        sentence = sentences[index].strip()
                        if not sentence:
                            process_next_sentence(index + 1)
                            return
                            
                        try:
                            # Create temp file for this sentence
                            temp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                            self.temp_files.append(temp.name)
                            
                            # Generate audio for the sentence
                            audio = models[language].tts_to_file(
                                sentence,
                                models[language].hps.data.spk2id[speaker],
                                temp.name,
                                speed=speed
                            )
                            
                            processed_sentences.append(temp.name)
                            
                            def update_progress(dt):
                                progress = ((index + 1) / len(sentences)) * 100
                                self.progress_bar.buffer_value = progress
                                
                                # Start playing if this is the first sentence
                                if index == 0:
                                    self.current_sentence_index = 0
                                    self.is_playing = True
                                    self.play_pause_btn.text = 'Pause'
                                    self.status_label.text = 'Playing'
                                    self.play_next_sentence()
                                
                                # Process next sentence
                                process_next_sentence(index + 1)
                                
                            Clock.schedule_once(update_progress)
                            
                        except Exception as e:
                            print(f"Error processing sentence {index}: {e}")
                            self.status_label.text = f'Error processing sentence {index + 1}'
                            self.synthesize_btn.disabled = False
                    
                    def process_next_sentence(index):
                        if index < len(sentences):
                            process_sentence(index)
                        else:
                            Clock.schedule_once(lambda dt: setattr(self.synthesize_btn, 'disabled', False), 0)
                    
                    # Start with first sentence
                    process_sentence(0)

                except Exception as e:
                    def show_error(dt):
                        self.status_label.text = f'Error: {str(e)}'
                        self.synthesize_btn.disabled = False
                        self.is_playing = False
                    Clock.schedule_once(show_error)

            threading.Thread(target=generate_audio, daemon=True).start()

        except Exception as e:
            self.status_label.text = f'Error: {str(e)}'
            self.synthesize_btn.disabled = False

    def __del__(self):
        # Clean up temp files
        for temp_file in self.temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        if self._update_event:
            self._update_event.cancel()

    def on_model_change(self, spinner, text):
        if text == 'Verbadik (Pro Version)':
            self.status_label.text = 'Upgrade to Unlock Verbadik'
            Clock.schedule_once(lambda dt: setattr(self.model_spinner, 'text', 'MeloTTS'), 1.5)
            return

        self.status_label.text = 'Ready'

    def on_language_change(self, spinner, text):
        self.speaker_spinner.values = list(models[text].hps.data.spk2id.keys())
        self.speaker_spinner.text = self.speaker_spinner.values[0]

        current_text = self.text_input.text
        if current_text in default_text_dict.values():
            self.text_input.text = default_text_dict[text]

    def on_speed_change(self, instance, value):
        self.speed_value_label.text = f'{value:.1f}'

    def format_time(self, seconds):
        if not seconds or seconds < 0:
            return '0:00'
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f'{minutes}:{seconds:02d}'

class MeloTTSApp(App):
    def build(self):
        return MeloTTSUI()

    def on_stop(self):
        pygame.mixer.quit()

if __name__ == '__main__':
    print("Make sure you've downloaded unidic (python -m unidic download) for this app to work.")
    MeloTTSApp().run()