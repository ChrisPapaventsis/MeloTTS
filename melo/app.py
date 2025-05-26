from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.spinner import Spinner, SpinnerOption
from kivy.uix.slider import Slider
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.progressbar import ProgressBar
from kivy.core.audio import SoundLoader
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.lang import Builder
from kivy.properties import ListProperty, StringProperty, NumericProperty, ObjectProperty, BooleanProperty
from kivy.animation import Animation
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
from kivy.core.window import Window
import threading
import re

import os
import torch
import io
import tempfile
from melo.api import TTS

# Set window size to match a typical smartphone aspect ratio (e.g., iPhone 12)
Window.size = (390, 844)
Window.minimum_width = 390
Window.minimum_height = 844

# Initialize TTS models
device = 'cpu'
models = {
    'EN': TTS(language='EN', device=device),
    'ES': TTS(language='ES', device=device),
    'FR': TTS(language='FR', device=device),
    'ZH': TTS(language='ZH', device=device),
    'JP': TTS(language='JP', device=device),
    'KR': TTS(language='KR', device=device),
}

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
        self.font_size = '16sp'
        self.bold = True
        self.background_normal = ''

class RoundedButton(Button):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_color = (0.2, 0.6, 1, 1)
        self.color = (1, 1, 1, 1)
        self.font_size = '18sp'
        self.bold = True

class RoundedTextInput(TextInput):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_normal = ''
        self.background_active = ''
        self.background_color = (1, 1, 1, 1)  # Pure white background
        self.foreground_color = (0, 0, 0, 1)  # Pure black text
        self.cursor_color = (0, 0, 0, 1)
        self.font_size = '16sp'
        self.padding = [20, 10]
        self.halign = 'left'
        self.valign = 'top'
        self._text_color = (0, 0, 0, 1)  # Force black text color
        self.text_color = (0, 0, 0, 1)  # Force black text color

class ProgressLabel(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint_y = None
        self.height = dp(30)
        self.color = (1, 1, 1, 1)  # White text color
        self.font_size = '14sp'
        self.halign = 'center'
        self.bold = True  # Make text bold for better visibility
        self.outline_width = 1  # Add outline for better visibility
        self.outline_color = (0, 0, 0, 0.5)  # Semi-transparent black outline

class RoundedSlider(Slider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cursor_size = (dp(20), dp(20))
        self.background_width = dp(10)
        self.padding = dp(20)

class CustomProgressBar(ProgressBar):
    value = NumericProperty(0)

    def set_value(self, value):
        Animation(value=value, duration=0.3).start(self)

class PlaybackButton(RoundedButton):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)
        self.width = dp(50)
        self.height = dp(50)
        self.font_size = '14sp'  # Smaller font size for text

class AudioProgressBar(CustomProgressBar):
    sound = ObjectProperty(None)
    _update_event = ObjectProperty(None)

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
                # Don't reset to 0 if we are just paused
                if self.sound.get_pos() == 0:
                    self.value = 0
                if self.sound.get_pos() >= self.sound.length - 0.1: # Account for float precision
                    self.value = 100


    def seek(self, touch):
        if self.sound and self.collide_point(*touch.pos):
            seek_pos_ratio = (touch.x - self.x) / self.width
            seek_pos_ratio = max(0, min(1, seek_pos_ratio))
            new_pos = seek_pos_ratio * self.sound.length
            self.sound.seek(new_pos)
            self.parent.parent.current_position = new_pos
            return True
        return False

    def on_touch_down(self, touch):
        if self.seek(touch):
            return True
        return super().on_touch_down(touch)

class FileSelectButton(RoundedButton):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)
        self.width = dp(110)  # Adjusted width
        self.height = dp(40)  # Slightly smaller height
        self.font_size = '14sp'  # Smaller font size
        self.text = 'Select File'

class RemoveFileButton(RoundedButton):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text = 'Remove'  # Changed from ✕
        self.size_hint = (None, None)
        self.width = dp(110)  # Match Select File button width
        self.height = dp(40)
        self.font_size = '14sp'  # Match Select File button font
        self.background_color = (0.9, 0.3, 0.3, 1)  # Reddish color

class FileChooserPopup(Popup):
    def __init__(self, callback, **kwargs):
        super().__init__(**kwargs)
        self.callback = callback
        self.title = 'Select a Text File'
        self.size_hint = (0.9, 0.9)

        layout = BoxLayout(orientation='vertical', spacing=dp(10))

        self.file_chooser = FileChooserListView(
            path=os.path.expanduser('~'),
            filters=['*.txt']
        )

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

        buttons.add_widget(Widget())  # Spacer
        buttons.add_widget(cancel_btn)
        buttons.add_widget(select_btn)
        buttons.add_widget(Widget())  # Spacer

        layout.add_widget(self.file_chooser)
        layout.add_widget(buttons)

        self.content = layout

    def _select_file(self, instance):
        if self.file_chooser.selection:
            selected_file = self.file_chooser.selection[0]
            self.callback(selected_file)
        self.dismiss()

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
            rgba: 0.9, 0.9, 0.9, 1
        RoundedRectangle:
            pos: self.x, self.y
            size: self.width, self.height
            radius: [self.height/2]
        Color:
            rgba: 0.2, 0.6, 1, 1
        RoundedRectangle:
            pos: self.x, self.y
            size: self.width * (self.value / 100.0), self.height
            radius: [self.height/2]
''')

class MeloTTSUI(BoxLayout):
    current_sound = ObjectProperty(None, allownone=True)
    is_playing = BooleanProperty(False)
    selected_file = StringProperty('')
    current_position = NumericProperty(0)  # Store current playback position

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = [dp(20), dp(30)]
        self.spacing = dp(15)

        # Title and Model Selection
        header_box = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(50),
            spacing=dp(10),
            padding=[dp(20), 0]  # Add horizontal padding
        )

        # Model Selection
        self.model_spinner = RoundedSpinner(
            text='MeloTTS',
            values=('MeloTTS', 'Verbadik (Pro Version)'),
            size_hint=(None, None),  # Changed from size_hint_x to size_hint
            width=dp(200),
            height=dp(40)
        )
        self.model_spinner.bind(text=self.on_model_change)

        # Add spacers on both sides for centering
        header_box.add_widget(Widget())  # Left spacer
        header_box.add_widget(self.model_spinner)
        header_box.add_widget(Widget())  # Right spacer

        self.add_widget(header_box)

        # Language selection
        self.language_spinner = RoundedSpinner(
            text='EN',
            values=('EN', 'ES', 'FR', 'ZH', 'JP', 'KR'),
            size_hint_y=None,
            height=dp(50)
        )
        self.language_spinner.bind(text=self.on_language_change)
        self.add_widget(self.language_spinner)

        # Speaker selection
        self.speaker_spinner = RoundedSpinner(
            text='EN-US',
            values=list(models['EN'].hps.data.spk2id.keys()),
            size_hint_y=None,
            height=dp(50)
        )
        self.add_widget(self.speaker_spinner)

        # Speed slider
        self.speed_slider = RoundedSlider(
            min=0.3,
            max=3.0,
            value=1.0,
            step=0.1,
            size_hint_y=None,
            height=dp(40)
        )
        self.speed_slider.bind(value=self.on_speed_change)

        # Speed layout
        speed_layout = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(40)
        )
        speed_layout.add_widget(Label(text='Speed:', size_hint_x=0.3))
        speed_layout.add_widget(self.speed_slider)
        self.speed_label = Label(
            text='1.0',
            size_hint_x=0.2
        )
        speed_layout.add_widget(self.speed_label)
        self.add_widget(speed_layout)

        # File selection
        file_box = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            height=dp(90),  # Increased height for two rows
            spacing=dp(5)
        )

        # File name display
        self.file_label = Label(
            text='No file selected',
            size_hint_y=None,
            height=dp(40),
            color=(1, 1, 1, 1)
        )

        # Buttons row
        buttons_box = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(40),
            spacing=dp(10),
            padding=[dp(20), 0]  # Add horizontal padding for centering
        )

        # Add spacer for centering
        buttons_box.add_widget(Widget())

        self.file_btn = FileSelectButton()
        self.file_btn.bind(on_press=self.show_file_chooser)  # Fixed binding

        self.remove_btn = RemoveFileButton()
        self.remove_btn.bind(on_press=self.remove_file)
        self.remove_btn.disabled = True  # Initially disabled
        self.remove_btn.opacity = 0.5  # Initially semi-transparent

        buttons_box.add_widget(self.file_btn)
        buttons_box.add_widget(self.remove_btn)

        # Add spacer for centering
        buttons_box.add_widget(Widget())

        file_box.add_widget(self.file_label)
        file_box.add_widget(buttons_box)

        self.add_widget(file_box)

        # Text input with label
        text_label = Label(
            text='Or enter text directly:',
            size_hint_y=None,
            height=dp(30),
            color=(1, 1, 1, 1),
            halign='left'
        )
        text_label.bind(size=text_label.setter('text_size'))

        self.add_widget(text_label)

        self.text_input = RoundedTextInput(
            text=default_text_dict['EN'],
            multiline=True,
            size_hint_y=None,
            height=dp(120)
        )
        self.add_widget(self.text_input)

        # Synthesize button
        self.synthesize_btn = RoundedButton(
            text='Generate',
            size_hint_y=None,
            height=dp(50),
        )
        self.synthesize_btn.bind(on_press=self.synthesize_audio)
        self.add_widget(self.synthesize_btn)

        # Audio controls
        controls_box = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(60),
            spacing=dp(10),
            padding=[dp(20), 0]
        )

        controls_box.add_widget(Widget())  # Spacer

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

        controls_box.add_widget(self.backward_btn)
        controls_box.add_widget(self.play_pause_btn)
        controls_box.add_widget(self.forward_btn)

        controls_box.add_widget(Widget())  # Spacer

        # Progress bar and time labels
        progress_box = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            height=dp(60),
            spacing=dp(5)
        )

        time_box = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(20)
        )

        self.current_time = Label(
            text='0:00',
            size_hint_x=None,
            width=dp(50),
            color=(1, 1, 1, 1)
        )
        self.total_time = Label(
            text='0:00',
            size_hint_x=None,
            width=dp(50),
            color=(1, 1, 1, 1)
        )

        time_box.add_widget(self.current_time)
        time_box.add_widget(Widget())  # Spacer
        time_box.add_widget(self.total_time)

        self.progress_bar = AudioProgressBar(
            max=100,
            value=0,
            size_hint_y=None,
            height=dp(20)
        )

        progress_box.add_widget(self.progress_bar)
        progress_box.add_widget(time_box)

        self.add_widget(controls_box)
        self.add_widget(progress_box)

        # Status label at the bottom
        self.status_label = Label(
            text='Ready',
            size_hint_y=None,
            height=dp(30),
            color=(1, 1, 1, 1)
        )
        self.add_widget(self.status_label)

        # Schedule time updates and position updates
        Clock.schedule_interval(self.update_time_labels, 0.1)
        Clock.schedule_interval(self.update_position, 0.1)

    def on_model_change(self, spinner, text):
        if text == 'Verbadik (Pro Version)':
            self.status_label.text = 'Upgrade to Unlock Verbadik'
            # Reset back to MeloTTS
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
        self.speed_label.text = f'{value:.1f}'

    def format_time(self, seconds):
        if not seconds:
            return '0:00'
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f'{minutes}:{seconds:02d}'

    def update_time_labels(self, dt):
        if self.current_sound:
            current = self.current_sound.get_pos()
            total = self.current_sound.length
            self.current_time.text = self.format_time(current)
            self.total_time.text = self.format_time(total)
            if self.is_playing and current >= total - 0.1:
                self.current_sound.stop()
                self.is_playing = False
                self.play_pause_btn.text = 'Play'
                self.current_position = 0

    def update_position(self, dt):
        if self.current_sound and self.is_playing:
            self.current_position = self.current_sound.get_pos()

    def seek_to_position(self, position):
        if self.current_sound:
            new_pos = max(0, min(position, self.current_sound.length))
            self.current_sound.seek(new_pos)
            self.current_position = new_pos


    def seek_forward(self, instance):
        if self.current_sound:
            new_pos = self.current_position + 10  # Seek forward 10 seconds
            self.seek_to_position(new_pos)

    def seek_backward(self, instance):
        if self.current_sound:
            new_pos = self.current_position - 10  # Seek backward 10 seconds
            self.seek_to_position(new_pos)

    def toggle_playback(self, instance):
        if not self.current_sound:
            return

        if self.is_playing:
            self.current_sound.stop()
            self.play_pause_btn.text = 'Play'
            self.is_playing = False
        else:
            self.current_sound.play()
            self.current_sound.seek(self.current_position)
            self.play_pause_btn.text = 'Pause'
            self.is_playing = True

    def show_file_chooser(self, instance):
        popup = FileChooserPopup(callback=self.on_file_selected)
        popup.open()

    def on_file_selected(self, filepath):
        self.selected_file = filepath
        filename = os.path.basename(filepath)
        # Truncate filename if too long
        if len(filename) > 30:
            filename = filename[:27] + "..."
        self.file_label.text = filename
        # Clear the text input to avoid confusion
        self.text_input.text = ''
        # Enable and show remove button
        self.remove_btn.disabled = False
        self.remove_btn.opacity = 1

    def remove_file(self, instance):
        self.selected_file = ''
        self.file_label.text = 'No file selected'
        # Restore default text
        self.text_input.text = default_text_dict[self.language_spinner.text]
        # Disable and hide remove button
        self.remove_btn.disabled = True
        self.remove_btn.opacity = 0.5

    def get_text_to_synthesize(self):
        if self.selected_file and os.path.exists(self.selected_file):
            try:
                with open(self.selected_file, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                self.status_label.text = f'Error reading file: {str(e)}'
                return None
        return self.text_input.text if self.text_input.text.strip() else None

    def synthesize_audio(self, instance):
        try:
            text = self.get_text_to_synthesize()
            if not text:
                self.status_label.text = 'Please enter text or select a file'
                return

            self.status_label.text = 'Initializing...'
            self.progress_bar.value = 0
            self.synthesize_btn.disabled = True

            def generate_audio():
                try:
                    language = self.language_spinner.text
                    speaker = self.speaker_spinner.text
                    speed = self.speed_slider.value

                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                        bio = io.BytesIO()

                        self.status_label.text = 'Generating audio...'
                        models[language].tts_to_file(
                            text,
                            models[language].hps.data.spk2id[speaker],
                            bio,
                            speed=speed,
                            format='wav'
                        )

                        self.status_label.text = 'Processing...'
                        temp_file.write(bio.getvalue())
                        temp_file.flush()

                        def play_audio(dt):
                            if self.current_sound:
                                self.current_sound.stop()
                                self.current_sound.unload()
                                self.is_playing = False

                            sound = SoundLoader.load(temp_file.name)
                            if sound:
                                self.current_sound = sound
                                self.progress_bar.sound = sound
                                self.play_pause_btn.text = 'Pause'
                                sound.play()
                                self.is_playing = True
                                self.status_label.text = 'Playing'
                            self.synthesize_btn.disabled = False

                        Clock.schedule_once(play_audio)

                except Exception as e:
                    def show_error(dt):
                        self.status_label.text = f'Error: {str(e)}'
                        self.synthesize_btn.disabled = False
                    Clock.schedule_once(show_error)

            threading.Thread(target=generate_audio, daemon=True).start()

        except Exception as e:
            self.status_label.text = f'Error: {str(e)}'
            self.synthesize_btn.disabled = False

class MeloTTSApp(App):
    def build(self):
        return MeloTTSUI()

if __name__ == '__main__':
    print("Make sure you've downloaded unidic (python -m unidic download) for this app to work.")
    MeloTTSApp().run()