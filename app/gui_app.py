from recognition_engine import RecognitionEngine
from pathlib import Path

import cv2
import customtkinter as ctk
from PIL import Image

import threading
import pyttsx3 


APP_TITLE = "SIGNET"
WINDOW_SIZE = "1200x760"

BACKGROUND = "#0B1020"
PANEL = "#141B2D"
SECONDARY_PANEL = "#1B2438"
ACCENT = "#5B8CFF"
GREEN = "#35D07F"
TEXT = "#F4F7FF"
MUTED = "#9BA8C7"


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class SignetGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title(APP_TITLE)
        self.geometry(WINDOW_SIZE)
        self.minsize(1050, 680)
        self.configure(fg_color=BACKGROUND)

        self.camera = None
        self.camera_running = False
        self.frame_counter = 0
        self.speech_running = False
        self.engine = RecognitionEngine() 

        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        self.build_camera_panel()
        self.build_control_panel()

        self.protocol("WM_DELETE_WINDOW", self.close_app)

    def build_camera_panel(self):
        panel = ctk.CTkFrame(
            self,
            fg_color=PANEL,
            corner_radius=18,
        )
        panel.grid(
            row=0,
            column=0,
            padx=(24, 12),
            pady=24,
            sticky="nsew",
        )

        panel.grid_rowconfigure(1, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            panel,
            text="Live Recognition",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=TEXT,
        )
        title.grid(row=0, column=0, padx=24, pady=(20, 12), sticky="w")

        placeholder = Image.new("RGB", (640, 480), color="#080C16")

        self.camera_image = ctk.CTkImage(
            light_image=placeholder,
            dark_image=placeholder,
            size=(640, 480),
        )

        self.camera_label = ctk.CTkLabel(
            panel,
            text="Camera stopped",
            image=self.camera_image,
            compound="center",
            font=ctk.CTkFont(size=20),
            text_color=MUTED,
            fg_color="#080C16",
            corner_radius=14,
        )
        self.camera_label.grid(
            row=1,
            column=0,
            padx=24,
            pady=(0, 18),
            sticky="nsew",
        )

        camera_buttons = ctk.CTkFrame(panel, fg_color="transparent")
        camera_buttons.grid(row=2, column=0, padx=24, pady=(0, 20))

        self.start_button = ctk.CTkButton(
            camera_buttons,
            text="Start Camera",
            command=self.start_camera,
            width=150,
            height=42,
            corner_radius=10,
            fg_color=ACCENT,
            hover_color="#4778E6",
        )
        self.start_button.grid(row=0, column=0, padx=6)

        self.stop_button = ctk.CTkButton(
            camera_buttons,
            text="Stop Camera",
            command=self.stop_camera,
            width=150,
            height=42,
            corner_radius=10,
            fg_color=SECONDARY_PANEL,
            hover_color="#27334D",
        )
        self.stop_button.grid(row=0, column=1, padx=6)

    def build_control_panel(self):
        panel = ctk.CTkFrame(
            self,
            fg_color=PANEL,
            corner_radius=18,
        )
        panel.grid(
            row=0,
            column=1,
            padx=(12, 24),
            pady=24,
            sticky="nsew",
        )

        panel.grid_columnconfigure((0, 1), weight=1)

        title = ctk.CTkLabel(
            panel,
            text="SIGNET",
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color=TEXT,
        )
        title.grid(
            row=0,
            column=0,
            columnspan=2,
            padx=24,
            pady=(24, 2),
            sticky="w",
        )

        subtitle = ctk.CTkLabel(
            panel,
            text="ASL fingerspelling recognition",
            font=ctk.CTkFont(size=14),
            text_color=MUTED,
        )
        subtitle.grid(
            row=1,
            column=0,
            columnspan=2,
            padx=24,
            pady=(0, 20),
            sticky="w",
        )

        prediction_card = ctk.CTkFrame(
            panel,
            fg_color=SECONDARY_PANEL,
            corner_radius=14,
        )
        prediction_card.grid(
            row=2,
            column=0,
            padx=(24, 8),
            pady=8,
            sticky="ew",
        )

        ctk.CTkLabel(
            prediction_card,
            text="LETTER",
            text_color=MUTED,
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(pady=(14, 0))

        self.prediction_label = ctk.CTkLabel(
            prediction_card,
            text="—",
            text_color=GREEN,
            font=ctk.CTkFont(size=44, weight="bold"),
        )
        self.prediction_label.pack(pady=(0, 12))

        confidence_card = ctk.CTkFrame(
            panel,
            fg_color=SECONDARY_PANEL,
            corner_radius=14,
        )
        confidence_card.grid(
            row=2,
            column=1,
            padx=(8, 24),
            pady=8,
            sticky="ew",
        )

        ctk.CTkLabel(
            confidence_card,
            text="CONFIDENCE",
            text_color=MUTED,
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(pady=(14, 0))

        self.confidence_label = ctk.CTkLabel(
            confidence_card,
            text="0%",
            text_color=TEXT,
            font=ctk.CTkFont(size=32, weight="bold"),
        )
        self.confidence_label.pack(pady=(8, 18))

        ctk.CTkLabel(
            panel,
            text="Recognized text",
            text_color=TEXT,
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(
            row=3,
            column=0,
            columnspan=2,
            padx=24,
            pady=(20, 8),
            sticky="w",
        )

        self.text_box = ctk.CTkTextbox(
            panel,
            height=140,
            corner_radius=12,
            fg_color="#0D1424",
            border_width=1,
            border_color="#27334D",
            text_color=TEXT,
            font=ctk.CTkFont(size=22),
            wrap="word",
        )
        self.text_box.grid(
            row=4,
            column=0,
            columnspan=2,
            padx=24,
            pady=(0, 14),
            sticky="ew",
        )

        self.motion_button = ctk.CTkButton(
            panel,
            text="Record J or Z Motion",
            command=self.start_motion, 
            height=48,
            corner_radius=12,
            fg_color=GREEN,
            hover_color="#29B66B",
            text_color="#07120C",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.motion_button.grid(
            row=5,
            column=0,
            columnspan=2,
            padx=24,
            pady=8,
            sticky="ew",
        )

        controls = ctk.CTkFrame(panel, fg_color="transparent")
        controls.grid(
            row=6,
            column=0,
            columnspan=2,
            padx=18,
            pady=8,
        )

        buttons = [
            ("Space", self.add_space),
            ("Backspace", self.backspace),
            ("Clear", self.clear_text),
            ("Copy", self.copy_text),
            ("Speak", self.speak_text),
        ]

        for index, (text, command) in enumerate(buttons):
            button = ctk.CTkButton(
                controls,
                text=text,
                command=command,
                width=90,
                height=38,
                corner_radius=9,
                fg_color=SECONDARY_PANEL,
                hover_color="#27334D",
            )
            button.grid(
                row=index // 2,
                column=index % 2,
                padx=6,
                pady=6,
            )

        self.status_label = ctk.CTkLabel(
            panel,
            text="Ready",
            text_color=MUTED,
            font=ctk.CTkFont(size=13),
        )
        self.status_label.grid(
            row=7,
            column=0,
            columnspan=2,
            padx=24,
            pady=(18, 20),
        )

    def start_camera(self):
        if self.camera_running:
            return

        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"),)
        
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.camera.isOpened():
            self.status_label.configure(
                text="Could not open the camera",
                text_color="#FF6B6B",
            )
            return

        self.camera_running = True
        self.status_label.configure(
            text="Camera running",
            text_color=GREEN,
        )
        self.update_camera()

    def update_camera(self):
        if not self.camera_running:
            return

        success, frame = self.camera.read()

        if not success:
            self.stop_camera()
            return

        self.frame_counter += 1
        
        should_recognize = (
            self.frame_counter % 3 == 0 
            or self.engine.motion_recording
        )
        
        if should_recognize:
            output = self.engine.process(frame) 
            display_frame = output.frame 
            
            self.prediction_label.configure(text=output.label)
            self.confidence_label.configure(
                text=f"{output.confidence:.0%}"
            )
            
            if output.added_text:
                self.text_box.insert("end", output.added_text)
                self.text_box.see("end")
            
            status_color = (
                GREEN
                if output.status.startswith("Added")
                else MUTED 
            )
            
            if not self.speech_running:    
                self.status_label.configure(
                    text=output.status, 
                    text_color = status_color,
                )
        else:
            display_frame = cv2.flip(frame, 1)
            
        rgb_frame = cv2.cvtColor(
            display_frame, 
            cv2.COLOR_BGR2RGB, 
        )
        
        image = Image.fromarray(rgb_frame)
        image = image.resize((640,480))
        
        self.camera_image.configure(
            light_image = image, 
            dark_image = image, 
        )
        
        self.camera_label.configure(
            image=self.camera_image, 
            text = "",
        )
        
        self.after(15, self.update_camera)

    def stop_camera(self):
        self.camera_running = False
        self.frame_counter = 0

        if self.camera is not None:
            self.camera.release()
            self.camera = None

        self.camera_label.configure(text="Camera stopped")
        self.status_label.configure(
            text="Camera stopped",
            text_color=MUTED,
        )

    def add_space(self):
        self.text_box.insert("end", " ")

    def backspace(self):
        text = self.text_box.get("1.0", "end-1c")

        if text:
            self.text_box.delete("1.0", "end")
            self.text_box.insert("1.0", text[:-1])

    def clear_text(self):
        self.text_box.delete("1.0", "end")
        self.engine.reset_text_state()

    def copy_text(self):
        text = self.text_box.get("1.0", "end-1c")

        self.clipboard_clear()
        self.clipboard_append(text)

        self.status_label.configure(
            text="Text copied",
            text_color=GREEN,
        )

    def start_motion(self):
        if not self.camera_running:
            self.status_label.configure(
                text="Start the camera first", 
                text_color = "#FFB454",
            )
            return
        
        started = self.engine.start_motion()
        
        self.status_label.configure(
            text = self.engine.motion_status,
            text_color = GREEN if started else "#FFB454",
        )

    def speak_text(self):
        text = self.text_box.get("1.0", "end-1c").strip()
        
        if not text:
            self.status_label.configure(
                text = "There is no text to speak.",
                text_color = "#FFB454",
            )
            return
        
        if self.speech_running:
            return
        
        self.speech_running = True
        
        self.status_label.configure(
            text = "Speaking...",
            text_color = GREEN, 
        )
        
        thread = threading.Thread(
            target=self.speech_worker, 
            args = (text,), 
            daemon = True, 
        )
        
        thread.start()
        
    def speech_worker(self, text):
        try:
            speech_engine = pyttsx3.init()
            speech_engine.setProperty("rate", 170)
            speech_engine.setProperty("volume", 1.0)
            
            speech_engine.say(text)
            speech_engine.runAndWait()
            speech_engine.stop()
        finally:
            self.after(0, self.speech_finished)
    
    def speech_finished(self):
        self.speech_running = False 
        
        self.status_label.configure(
            text = "Finished speaking", 
            text_color = MUTED,
        )
        
        
    def close_app(self):
        self.stop_camera()
        self.engine.close()
        self.destroy()


if __name__ == "__main__":
    app = SignetGUI()
    app.mainloop()