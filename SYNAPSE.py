import cv2
import asyncio
import mediapipe as mp
import pyautogui
import time
import platform
import numpy as np
from pynput.mouse import Button, Controller
import os
from datetime import datetime
import threading
from collections import deque
import pyttsx3
import speech_recognition as sr
import webbrowser
from dotenv import load_dotenv

load_dotenv(r"C:\Users\KIIT0001\OneDrive\Documents\Vir\.env")

class HandGestureController:
    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.01
        self.mouse = Controller()
        self.screenshot_dir = "screenshots"
        os.makedirs(self.screenshot_dir, exist_ok=True)
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=1
        )
        self.position_smoother = PositionSmoother(
            smoothing_factor=0.3,
            history_size=5
        )
        self.gesture_handler = GestureHandler(debounce_time=0.5)
        self.show_ui = True
        self.ui_overlay = UIOverlay()
        self.config = {
            "sensitivity": 1.0,
            "gesture_recognition_threshold": 0.8,
            "scroll_speed": 30,
            "click_threshold": 0.03
        }
        self.running = False
        self.camera_id = 0
        self.frame_size = (640, 480)
        self.is_dragging = False

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        landmarks_list = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarks_list.append((landmark.x, landmark.y, landmark.z))
                if self.show_ui:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                gesture_info = self.detect_and_handle_gestures(landmarks_list, results)
                if self.show_ui:
                    self.ui_overlay.update(frame, gesture_info)
        return frame

    def detect_and_handle_gestures(self, landmarks_list, results):
        if not landmarks_list or len(landmarks_list) < 21:
            return {"name": "No Hand", "confidence": 0.0}
        hand_landmarks = results.multi_hand_landmarks[0]
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        pinky_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP]
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        thumb_index_dist = self.euclidean_distance(
            (thumb_tip.x, thumb_tip.y), 
            (index_tip.x, index_tip.y)
        )
        thumb_extended = thumb_tip.y < thumb_ip.y
        index_extended = index_tip.y < index_pip.y
        middle_extended = middle_tip.y < middle_pip.y
        ring_extended = ring_tip.y < ring_pip.y
        pinky_extended = pinky_tip.y < pinky_pip.y
        if index_extended and not middle_extended and not ring_extended and not pinky_extended:
            x, y = self.position_smoother.update(
                index_tip.x * self.screen_width,
                index_tip.y * self.screen_height
            )
            pyautogui.moveTo(int(x), int(y))
            if thumb_index_dist < self.config["click_threshold"] and thumb_extended:
                if not self.is_dragging and self.gesture_handler.can_trigger("left_click"):
                    self.mouse.press(Button.left)
                    self.is_dragging = True
                    return {"name": "Left Click & Drag", "confidence": 0.9}
            elif self.is_dragging:
                self.mouse.release(Button.left)
                self.is_dragging = False
            return {"name": "Move Cursor", "confidence": 0.95}
        if (index_extended and middle_extended and not ring_extended and not pinky_extended and 
            not self.is_dragging and self.gesture_handler.can_trigger("left_click")):
            self.mouse.press(Button.left)
            self.mouse.release(Button.left)
            return {"name": "Left Click", "confidence": 0.9}
        if (index_extended and not middle_extended and not ring_extended and pinky_extended and 
            self.gesture_handler.can_trigger("right_click")):
            self.mouse.press(Button.right)
            self.mouse.release(Button.right)
            return {"name": "Right Click", "confidence": 0.9}
        if (index_extended and middle_extended and self.gesture_handler.can_trigger("double_click")):
            pyautogui.doubleClick()
            return {"name": "Double Click", "confidence": 0.85}
        if (index_extended and middle_extended and ring_extended and pinky_extended and thumb_extended):
            wrist_y_change = self.gesture_handler.track_motion("wrist_y", wrist.y)
            if wrist_y_change < -0.015:
                pyautogui.scroll(self.config["scroll_speed"])
                return {"name": "Scroll Up", "confidence": 0.8}
            elif wrist_y_change > 0.015:
                pyautogui.scroll(-self.config["scroll_speed"])
                return {"name": "Scroll Down", "confidence": 0.8}
            return {"name": "Ready to Scroll", "confidence": 0.7}
        if (index_extended and not middle_extended and not ring_extended and pinky_extended and 
            thumb_extended and self.gesture_handler.can_trigger("screenshot")):
            self.take_screenshot()
            return {"name": "Screenshot", "confidence": 0.85}
        pinch_distance = self.euclidean_distance(
            (thumb_tip.x, thumb_tip.y),
            (index_tip.x, index_tip.y)
        )
        if thumb_extended and index_extended:
            pinch_change = self.gesture_handler.track_motion("pinch", pinch_distance)
            if abs(pinch_change) > 0.01:
                if pinch_change > 0:
                    pyautogui.keyDown('ctrl')
                    pyautogui.scroll(10)
                    pyautogui.keyUp('ctrl')
                    return {"name": "Zoom In", "confidence": 0.8}
                else:
                    pyautogui.keyDown('ctrl')
                    pyautogui.scroll(-10)
                    pyautogui.keyUp('ctrl')
                    return {"name": "Zoom Out", "confidence": 0.8}
            return {"name": "Pinch/Zoom Gesture", "confidence": 0.7}
        if (not index_extended and not middle_extended and not ring_extended and 
            not pinky_extended and self.gesture_handler.can_trigger("toggle_ui")):
            self.show_ui = not self.show_ui
            return {"name": "Toggle UI", "confidence": 0.9}
        return {"name": "No Gesture", "confidence": 0.5}

    def take_screenshot(self):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.screenshot_dir}/screenshot_{timestamp}.png"
            screenshot = pyautogui.screenshot()
            screenshot.save(filename)
        except Exception as e:
            pass

    @staticmethod
    def euclidean_distance(point1, point2):
        return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

    async def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
        if not cap.isOpened():
            return
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame = self.process_frame(frame)
                if self.show_ui:
                    cv2.imshow('Hand Gesture Controller', processed_frame)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('u'):
                    self.show_ui = not self.show_ui
                elif key == ord('s'):
                    self.take_screenshot()
                await asyncio.sleep(0.016)
        except Exception as e:
            pass
        finally:
            self.running = False
            cap.release()
            cv2.destroyAllWindows()

    def stop(self):
        self.running = False

class PositionSmoother:
    def __init__(self, smoothing_factor=0.3, history_size=5):
        self.smoothing_factor = smoothing_factor
        self.history_size = history_size
        self.history_x = deque(maxlen=history_size)
        self.history_y = deque(maxlen=history_size)
        self.prev_x, self.prev_y = None, None
        self.velocity_x, self.velocity_y = 0, 0
        self.acceleration_factor = 1.2
        
    def update(self, x, y):
        if self.prev_x is None:
            self.prev_x, self.prev_y = x, y
            self.history_x.append(x)
            self.history_y.append(y)
            return x, y
        self.velocity_x = x - self.prev_x
        self.velocity_y = y - self.prev_y
        velocity_magnitude = np.sqrt(self.velocity_x**2 + self.velocity_y**2)
        adaptive_factor = self.smoothing_factor
        if velocity_magnitude > 20:
            adaptive_factor = min(0.7, self.smoothing_factor * self.acceleration_factor)
        smooth_x = adaptive_factor * x + (1 - adaptive_factor) * self.prev_x
        smooth_y = adaptive_factor * y + (1 - adaptive_factor) * self.prev_y
        self.history_x.append(smooth_x)
        self.history_y.append(smooth_y)
        self.prev_x, self.prev_y = smooth_x, smooth_y
        return smooth_x, smooth_y

class GestureHandler:
    def __init__(self, debounce_time=0.4):
        self.debounce_time = debounce_time
        self.last_gesture_times = {}
        self.motion_tracking = {}
        
    def can_trigger(self, gesture_name):
        current_time = time.time()
        last_time = self.last_gesture_times.get(gesture_name, 0)
        if current_time - last_time >= self.debounce_time:
            self.last_gesture_times[gesture_name] = current_time
            return True
        return False
    
    def track_motion(self, motion_name, current_value):
        if motion_name not in self.motion_tracking:
            self.motion_tracking[motion_name] = {
                "previous": current_value,
                "current": current_value,
                "timestamp": time.time()
            }
            return 0
        previous = self.motion_tracking[motion_name]["current"]
        self.motion_tracking[motion_name] = {
            "previous": previous,
            "current": current_value,
            "timestamp": time.time()
        }
        return current_value - previous

class UIOverlay:
    def __init__(self):
        self.bg_opacity = 0.3
        self.text_color = (0, 255, 0)
        self.active_color = (0, 255, 255)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2
        self.margin = 10
        self.history = []
        self.max_history = 5
        
    def update(self, frame, gesture_info):
        if gesture_info["name"] != "No Gesture" and gesture_info["name"] != "No Hand":
            if not self.history or self.history[-1] != gesture_info["name"]:
                self.history.append(gesture_info["name"])
                if len(self.history) > self.max_history:
                    self.history.pop(0)
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (300, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, self.bg_opacity, frame, 1 - self.bg_opacity, 0, frame)
        self.draw_text(frame, f"Gesture: {gesture_info['name']}", 10, 30)
        self.draw_text(frame, f"Confidence: {gesture_info['confidence']:.2f}", 10, 60)
        self.draw_text(frame, "Recent Gestures:", 10, 90)
        for i, name in enumerate(reversed(self.history)):
            self.draw_text(frame, f"- {name}", 20, 120 + i * 30)
        self.draw_text(frame, "Press 'Q' to quit, 'U' to toggle UI", w - 300, 30, (255, 255, 255))
    
    def draw_text(self, frame, text, x, y, color=None):
        if color is None:
            color = self.text_color
        cv2.putText(frame, text, (x+2, y+2), self.font, self.font_scale, (0, 0, 0), 
                    self.thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), self.font, self.font_scale, color, 
                    self.thickness, cv2.LINE_AA)

class VoiceAssistant:
    def __init__(self, gesture_controller):
        self.gesture_controller = gesture_controller
        self.running = True
        self.sites = {
            "google": "https://www.google.com",
            "youtube": "https://www.youtube.com",
            "amazon": "https://www.amazon.com",
            "netflix": "https://www.netflix.com",
            "instagram": "https://www.instagram.com",
            "leetcode": "https://leetcode.com",
            "github": "https://github.com",
            "linkedin": "https://www.linkedin.com",
            "spotify": "https://www.spotify.com"
        }
        self.silent_errors = True
        self.listen_delay = 0.5

    def say(self, text):
        try:
            engine = pyttsx3.init()
            rate = engine.getProperty('rate')
            engine.setProperty('rate', rate - 50)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            pass

    def ai(self, prompt):
        try:
            import google.generativeai as genai
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                self.say("Gemini API key not configured.")
                return
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    max_output_tokens=256,
                    temperature=0.7,
                )
            )
            if not response.text:
                self.say("Sorry, I couldn't get a response from the AI.")
                return
            ai_response = response.text.strip()
            self.say(ai_response)
            if not os.path.exists("AI_responses"):
                os.makedirs("AI_responses")
            filename = f"AI_responses/gemini_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Prompt: {prompt}\n\nGemini Response: {ai_response}")
        except ImportError:
            self.say("Gemini library is not installed.")
        except Exception as e:
            self.say("Sorry, there was an error with the Gemini AI request.")
            ai_response = self.get_offline_response(prompt)
            self.say(ai_response)

    def get_offline_response(self, prompt):
        prompt = prompt.lower()
        if "time" in prompt:
            return f"The current time is {datetime.now().strftime('%I:%M %p')}"
        elif "date" in prompt:
            return f"Today's date is {datetime.now().strftime('%B %d, %Y')}"
        elif any(word in prompt for word in ["what", "how", "why", "when", "where"]):
            return "That's an interesting question! I'd need internet access to give you a detailed answer."
        elif any(word in prompt for word in ["hello", "hi", "hey"]):
            return "Hello! How can I help you today?"
        else:
            return "I understand you're asking about that topic. For detailed information, I'd recommend doing an online search!"

    def select_microphone(self):
        print("\nAvailable microphones:")
        microphones = sr.Microphone.list_microphone_names()
        input_mics = []
        for i, name in enumerate(microphones):
            if any(keyword in name.lower() for keyword in ['mic', 'input', 'headset', 'array']) and \
               not any(keyword in name.lower() for keyword in ['speaker', 'output', 'playback']):
                input_mics.append((i, name))
                print(f"{len(input_mics)-1}: {name}")
        if not input_mics:
            print("No clear input microphones found. Showing all devices:")
            for i, name in enumerate(microphones):
                print(f"{i}: {name}")
            input_mics = [(i, name) for i in range(len(microphones))]
        while True:
            try:
                choice = input(f"\nSelect microphone (0-{len(input_mics)-1} or press Enter for default: ").strip()
                if choice == "":
                    return None
                choice = int(choice)
                if 0 <= choice < len(input_mics):
                    mic_index = input_mics[choice][0]
                    print(f"Selected: {input_mics[choice][1]}")
                    return mic_index
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    def test_microphone(self, mic_index=None):
        print("\nTesting microphone...")
        r = sr.Recognizer()
        r.energy_threshold = 150
        r.dynamic_energy_threshold = False
        r.pause_threshold = 1.5
        try:
            if mic_index is not None:
                source = sr.Microphone(device_index=mic_index)
            else:
                source = sr.Microphone()
            with source as source:
                print("Adjusting for ambient noise...")
                r.adjust_for_ambient_noise(source, duration=2)
                print(f"Audio threshold set to: {r.energy_threshold:.2f}")
                print("Say 'hello' clearly...")
                try:
                    audio = r.listen(source, timeout=10, phrase_time_limit=10)
                    print(f"Audio captured! Duration: {len(audio.get_raw_data()) / 16000.0:.2f} seconds")
                    print("Processing audio...")
                    text = r.recognize_google(audio, language="en")
                    print(f"Success! Recognized: '{text}'")
                    return True
                except sr.WaitTimeoutError:
                    print("Timeout: No speech detected.")
                    return False
                except sr.UnknownValueError:
                    print("Audio captured but could not understand speech.")
                    return False
                except Exception as e:
                    print(f"Recognition error: {str(e)}")
                    return False
        except Exception as e:
            print(f"Microphone error: {str(e)}")
            return False

    def take_command(self, mic_index=None):
        r = sr.Recognizer()
        r.energy_threshold = 150
        r.dynamic_energy_threshold = False
        r.pause_threshold = 0.5
        max_retries = 2
        retry_count = 0
        while retry_count <= max_retries:
            try:
                if mic_index is not None:
                    source = sr.Microphone(device_index=mic_index)
                else:
                    source = sr.Microphone()
                with source as s:
                    print("🎙️ Listening...")
                    try:
                        audio = r.listen(s, phrase_time_limit=15)
                        print("Processing...")
                        query = r.recognize_google(audio, language="en")
                        print(f"✅ You said: {query}")
                        return query
                    except sr.UnknownValueError:
                        retry_count += 1
                        if not self.silent_errors and retry_count <= max_retries:
                            self.say(f"Please repeat. Attempt {retry_count} of {max_retries}.")
                        return None
                    except sr.RequestError:
                        if not self.silent_errors:
                            self.say("Network error. Check your internet connection.")
                        return None
            except Exception as e:
                if not self.silent_errors:
                    self.say("Microphone error. Check your device.")
                return None

    def run(self):
        print("🚀 Starting AI Assistant...")
        selected_mic = self.select_microphone()
        if not self.test_microphone(selected_mic):
            print("\n❌ Microphone test failed!")
            print("\nTroubleshooting Steps:")
            print("1. Check Windows Settings: Privacy > Microphone > Allow apps access")
            print("2. Try another microphone from the list")
            print("3. Ensure microphone is default in Sound settings")
            print("4. Restart and retry")
            retry = input("\nTry another microphone? (y/n): ").lower()
            if retry == 'y':
                selected_mic = self.select_microphone()
                if not self.test_microphone(selected_mic):
                    print("Still not working. Check your setup.")
                    return
            else:
                return
        print("✅ Microphone test passed!")
        self.say("Hello, I'm SYNAPSE. How can I assist you? Hand gestures are available.")
        print("\n💡 Tips:")
        print("- Speak clearly for voice commands")
        print("- Say 'help' for commands")
        print("- Say 'exit' to quit")
        print("- Use 'start gesture' or 'stop gesture' for gesture mode")
        print("- Say 'toggle ui' for gesture UI")
        print("- Say 'mute voice' or 'unmute voice' for error messages")
        print("-" * 50)
        while self.running:
            try:
                if self.gesture_controller.running:
                    time.sleep(self.listen_delay * 2)
                else:
                    time.sleep(self.listen_delay)
                query = self.take_command(selected_mic)
                if not query:
                    continue
                query = query.lower()
                print(f"🔍 Processing: {query}")
                if query == "exit":
                    self.say("Goodbye! Shutting down...")
                    self.gesture_controller.stop()
                    self.running = False
                    break
                if "start gesture" in query:
                    if not self.gesture_controller.running:
                        self.say("Starting gesture control")
                        threading.Thread(target=lambda: asyncio.run(self.gesture_controller.run()), daemon=True).start()
                    else:
                        self.say("Gesture control already active.")
                elif "stop gesture" in query:
                    if self.gesture_controller.running:
                        self.say("Gesture control stopped.")
                        self.gesture_controller.stop()
                    else:
                        self.say("Gesture control not active.")
                elif "toggle ui" in query:
                    self.gesture_controller.show_ui = not self.gesture_controller.show_ui
                    self.say(f"Gesture UI {'enabled' if self.gesture_controller.show_ui else 'disabled'}")
                elif "mute voice" in query:
                    self.silent_errors = True
                    self.say("Voice errors muted")
                elif "unmute voice" in query:
                    self.silent_errors = False
                    self.say("Voice errors enabled")
                elif "take screenshot" in query:
                    self.gesture_controller.take_screenshot()
                    self.say("Taking screenshot...")
                site_opened = False
                for site in self.sites:
                    if f"open {site}" in query or f"{site}.com" in query:
                        self.say(f"Opening {site}")
                        webbrowser.open(self.sites[site])
                        site_opened = True
                        break
                if site_opened:
                    continue
                if "open document" in query or "documents" in query:
                    try:
                        docs_path = os.path.expanduser("~/Documents")
                        if os.path.exists(docs_path):
                            os.startfile(docs_path)
                            self.say("Opening documents folder")
                        else:
                            self.say("Documents folder not found")
                    except Exception:
                        self.say("Couldn't open documents folder.")
                elif "time" in query:
                    current_time = datetime.now().strftime("%I:%M %p")
                    time_response = f"The time is {current_time}"
                    print(time_response)
                    self.say(time_response)
                elif "date" in query:
                    current_date = datetime.now().strftime("%B %d, %Y")
                    date_response = f"Today's date: {current_date}"
                    print(date_response)
                    self.say(date_response)
                elif any(phrase in query for phrase in ["using ai", "ask ai", "artificial intelligence", "ai question", "snow ai"]):
                    self.say("What would you like to ask the AI?")
                    print("🤖 Waiting for AI query...")
                    ai_query = self.take_command(selected_mic)
                    if ai_query:
                        print(f"🤖 AI Query: {ai_query}")
                        self.ai(ai_query)
                    else:
                        if not self.silent_errors:
                            self.say("I didn't catch your question. Try again.")
                elif "help" in query or "what can you do" in query:
                    help_text = """Available Commands:
- Open websites: 'open google', 'youtube.com', etc.
- Get time: 'what time is it'
- Get date: 'what's the date'
- AI queries: 'ask ai' followed by your question
- Open documents: 'open documents'
- Gesture control: 'start gesture', 'stop gesture', 'toggle ui'
- Take screenshot: 'take screenshot'
- Mute errors: 'mute voice'
- Unmute errors: 'unmute voice'
- Exit: 'exit'"""
                    print(help_text)
                    self.say("I can open websites, tell time, answer AI queries, open documents, control gestures, and more.")
                elif any(word in query for phrase in ["what", "how", "why", "when", "where", "who"]):
                    print(f"🤖 AI Query: {query}")
                    self.ai(query)
                else:
                    self.say("I don't understand. Say 'help' for commands.")
            except KeyboardInterrupt:
                print("\n🖐️ Interrupted by user")
                self.say("Goodbye!")
                self.gesture_controller.stop()
                self.running = False
                break
            except Exception as e:
                if not self.silent_errors:
                    self.say("Something went wrong.")

def main():
    gesture_controller = HandGestureController()
    voice_assistant = VoiceAssistant(gesture_controller)
    voice_assistant.run()

if __name__ == "__main__":
    if platform.system() == "Emscripten":
        asyncio.ensure_future(main())
    else:
        main()