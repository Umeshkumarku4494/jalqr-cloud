from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.uix.label import Label
from kivy.clock import Clock
import cv2
import numpy as np
import jalqr_realtime_engine as engine

class JalQRApp(App):
    def build(self):
        # Main Layout
        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # 1. Camera Widget
        # index=0 is usually the back camera on Android
        self.camera = Camera(play=True, resolution=(640, 480))
        self.layout.add_widget(self.camera)

        # 2. Results Label
        self.result_label = Label(
            text="Awaiting Scan...",
            size_hint_y=0.3,
            font_size='20sp',
            markup=True
        )
        self.layout.add_widget(self.result_label)

        # Schedule the analysis every 1 second
        Clock.schedule_interval(self.analyze_frame, 1.0)

        return self.layout

    def analyze_frame(self, dt):
        if not self.camera.texture:
            return

        # Convert Kivy Texture to OpenCV Image (numpy array)
        texture = self.camera.texture
        size = texture.size
        pixels = texture.pixels
        
        # Create numpy array from pixels
        img = np.frombuffer(pixels, np.uint8).reshape(size[1], size[0], 4)
        # Convert RGBA to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        # Process using your existing engine
        result = engine.process_frame(img_bgr)

        # Update the UI
        color_map = {
            "SAFE": "00FF00",  # Green
            "UNSAFE": "FF0000", # Red
            "CHECK": "FFFF00"   # Yellow
        }
        text_color = color_map.get(result['status'], "FFFFFF")
        
        self.result_label.text = (
            f"[color={text_color}]STATUS: {result['status']}[/color]\n"
            f"Cl: {result['chlorine']} | NO3: {result['nitrate']}\n"
            f"Fe: {result['iron']} | PO4: {result['phosphate']}"
        )

if __name__ == '__main__':
    JalQRApp().run()