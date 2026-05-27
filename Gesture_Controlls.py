"""
Name:           hand_control
Author:         Edvin Andersson
Date:           2026-05-23
Description:    Controls system functions, key inputs and mouse actions with hand gestures. Built on MediaPipe's landmark detection and OpenCV's image processing.
"""
 
# Libraries
import cv2
import mediapipe as mp
import math
from collections import deque
import screen_brightness_control as sbc
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController

# Configuration
Max_Hands       = 1
Gesture_Confirm = 5
Camera_Id       = 0
Fast_Interval   = 2
Slow_Interval   = 2
Scroll_Steps    = 5
Brightness_Step = 2
Volume_Step     = 0.01

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=Max_Hands, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(Camera_Id)
cv2.namedWindow("Hand Tracker v2", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Hand Tracker v2", cv2.WND_PROP_TOPMOST, 1)

gesture_buffers = {"Right": deque(maxlen=Gesture_Confirm), "Left": deque(maxlen=Gesture_Confirm)}
last_confirmed = {"Right": "", "Left": ""}

# Help functions
def dist(lm, a, b):
    """Calculates the Euclidean distance between two landmarks.
    Parameters: lm (landmark list), a (index of first landmark), b (index of second landmark)
    Returns: Float distance between the two points
    """
    return math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)

def normalize(lm, a, b):
    """Calculates the distance between two landmarks normalized against hand size.
    Parameters: lm (landmark list), a (index of first landmark), b (index of second landmark)
    Returns: Float normalized distance (0.0 - 1.0 roughly)
    """
    hand_size = dist(lm, 0, 9) + 1e-6
    return dist(lm, a, b) / hand_size

def get_hand_facing(lm, is_right):
    """Determines whether the hand is facing palm-side or back-side toward the camera.
    Parameters: lm (landmark list), is_right (bool, True if right hand)
    Returns: String "palm" or "back"
    """
    v1 = (lm[5].x - lm[0].x,  lm[5].y - lm[0].y)
    v2 = (lm[17].x - lm[0].x, lm[17].y - lm[0].y)
    cross_z = v1[0] * v2[1] - v1[1] * v2[0]
    if is_right:
        return "palm" if cross_z > 0 else "back"
    else:
        return "palm" if cross_z < 0 else "back"

def finger_up(lm, tip, pip, is_right=True, is_thumb=False, facing="palm"):
    """Checks whether a given finger is extended upward, with special sideways logic for the thumb.
    Parameters: lm (landmark list), tip (tip landmark index), pip (base landmark index),
                is_right (bool), is_thumb (bool), facing (str "palm" or "back")
    Returns: Bool, True if finger is considered up/extended
    """
    if is_thumb:
        if facing == "palm":
            return lm[4].x < lm[3].x if is_right else lm[4].x > lm[3].x
        else:
            return lm[4].x > lm[3].x if is_right else lm[4].x < lm[3].x
    return lm[tip].y < lm[pip].y

def thumb_direction(lm):
    """Determines whether the thumb is pointing up, down, or sideways based on its vertical position.
    Parameters: lm (landmark list)
    Returns: String "Thumbs Up", "Thumbs Down", or "Thumbs Sideways"
    """
    hand_height = abs(lm[0].y - lm[9].y) + 1e-6
    diff = (lm[0].y - lm[4].y) / hand_height
    if diff > 1:
        return "Thumbs Up"
    elif diff < -0.2:
        return "Thumbs Down"
    else:
        return "Thumbs Sideways"

def detect_gesture(lm, is_right):
    """Identifies the current hand gesture using landmark positions, pinch distances, and finger states.
    Parameters: lm (landmark list), is_right (bool, True if right hand)
    Returns: String name of the detected gesture
    """
    facing = get_hand_facing(lm, is_right)

    thumb  = finger_up(lm, 4,  3,  is_right, is_thumb=True, facing=facing)
    index  = finger_up(lm, 8,  6)
    middle = finger_up(lm, 12, 10)
    ring   = finger_up(lm, 16, 14)
    pinky  = finger_up(lm, 20, 18)
    fingers = [thumb, index, middle, ring, pinky]

    pinch_ti  = normalize(lm, 4, 8)
    index_mid = normalize(lm, 8, 12)

    CLOSE = 0.15
    APART = 0.30

    # Distance-based gestures
    if pinch_ti < CLOSE:
        if middle and ring and pinky:
            return "OK"
        else:
            return "Pinch"

    # Binary gestures
    if fingers == [0, 0, 0, 0, 0]: return "Fist"
    if fingers == [1, 1, 1, 1, 1]: return "Open Hand"
    if fingers == [0, 1, 0, 0, 0]: return "Pointing"
    if fingers == [0, 1, 1, 0, 0] and index_mid > APART: return "Peace"
    if fingers == [0, 1, 1, 0, 0] and index_mid < CLOSE: return "Crossed Fingers"
    if fingers == [0, 0, 1, 0, 0]: return "Middle Finger"
    if fingers == [1, 0, 0, 0, 0]: return thumb_direction(lm)
    if fingers == [0, 0, 0, 0, 1]: return "Pinky Up"
    if fingers == [0, 1, 1, 1, 0]: return "Three Middle"
    if fingers == [1, 1, 1, 0, 0]: return "Three Thumb"
    if fingers == [0, 1, 1, 1, 1]: return "Four Fingers"
    if fingers == [1, 0, 0, 0, 1]: return "One Five"
    if fingers == [0, 1, 0, 0, 1]: return "Two Five"
    if fingers == [1, 1, 0, 0, 0]: return "Take L"
    return "Unknown"

# System controls
keyboard = KeyboardController()
mouse    = MouseController()

try:
    current_brightness = sbc.get_brightness()[0]
except Exception:
    current_brightness = 50

def set_brightness(level: int):
    """Sets the screen brightness to an absolute level and updates the cached value.
    Parameters: level (int, 0-100)
    Returns: Void
    """
    global current_brightness
    try:
        sbc.set_brightness(level)
        current_brightness = level
        print(f"[Brightness]: {level}%")
    except Exception as e:
        print(f"[Brightness]: Could not set {e}")

def adjust_brightness(step: int):
    """Adjusts the screen brightness relative to the current cached value, snapped to Brightness_Step grid.
    Parameters: step (int, positive to increase, negative to decrease)
    Returns: Void
    """
    global current_brightness
    new = current_brightness + step
    new = round(new / Brightness_Step) * Brightness_Step
    new = max(0, min(100, new))
    if new != current_brightness:
        set_brightness(new)

def single_key(key1):
    """Simulates a single keyboard key press and release.
    Parameters: key1 (pynput Key or char)
    Returns: Void
    """
    keyboard.press(key1)  
    keyboard.release(key1)
    print(f"[Keyboard]: {key1}")

def double_key(key1, key2):
    """Simulates a keyboard shortcut by pressing two keys simultaneously.
    Parameters: key1 (modifier key), key2 (second key)
    Returns: Void
    """
    keyboard.press(key1)
    keyboard.press(key2)
    keyboard.release(key2)
    keyboard.release(key1)
    print(f"[Keyboard]: {key1} + {key2}")

def scroll(direction: int):
    """Scrolls the mouse wheel vertically by Scroll_Steps units.
    Parameters: direction (int, positive = up, negative = down)
    Returns: Void
    """
    mouse.scroll(Scroll_Steps, direction)
    print(f"[Scroll]: {'up' if direction > 0 else 'down'}")

def click(button=Button.left):
    """Performs a mouse button click.
    Parameters: button (pynput Button, default left click)
    Returns: Void
    """
    mouse.click(button)
    print(f"[Click]: {button}")

# Volume setup
_devices        = AudioUtilities.GetSpeakers()
_interface      = _devices._dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume_control  = cast(_interface, POINTER(IAudioEndpointVolume))

def adjust_volume(step: float):
    """Adjusts the system master volume relative to the current level using the Windows Audio API.
    Parameters: step (float, positive to increase, negative to decrease)
    Returns: Void
    """
    current = volume_control.GetMasterVolumeLevelScalar()
    new     = round(max(0.0, min(1.0, current + step)), 2)
    volume_control.SetMasterVolumeLevelScalar(new, None)
    print(f"[Volume]: {round(new * 100)}%")

def toggle_mute():
    """Toggles the system master volume mute state on or off.
    Parameters: Void
    Returns: Void
    """
    muted = volume_control.GetMute()
    volume_control.SetMute(not muted, None)
    print(f"[Volume]: {'Muted' if not muted else 'Unmuted'}")

# Gesture actions
def continuous(fn):
    """Marks a lambda as continuous so it repeats every frame while the gesture is held, instead of firing once.
    Parameters: fn (callable)
    Returns: The same callable with _continuous attribute set to True
    """
    fn._continuous = True
    return fn

# Modes
MODES = {
    "Three Thumb": ("Scroll", continuous(lambda: scroll(1)),
                continuous(lambda: scroll(-1)),                              
                (0, 255, 255)),
    "Three Middle": ("Brightness", continuous(lambda: adjust_brightness(Brightness_Step)),
                    continuous(lambda: adjust_brightness(-Brightness_Step)),
                    (255, 0, 255)),
    "Four Fingers": ("Volume", continuous(lambda: adjust_volume(Volume_Step)),
                    continuous(lambda: adjust_volume(-Volume_Step)),
                    (255, 255, 0)),
    "Two Five": ("Game", continuous(lambda: adjust_volume(Volume_Step)),
                    continuous(lambda: adjust_volume(-Volume_Step)),
                    (255, 127, 127)),
}

GESTURE_ACTIONS = {
    "Peace":         lambda: double_key(Key.alt, Key.tab),
    "Pinch":         lambda: click(Button.left),
    "Middle Finger": "QUIT",
}

TOGGLE_GESTURE  = "OK"
actions_enabled = True
active_mode     = None

# Frame counters
prev_time          = 0
repeat_frame_count = 0
mode_frame_counts  = {"Scroll": 0, "Brightness": 0, "Volume": 0, "Game": 0}

def draw_hud(img, fps):
    """Draws FPS counter in top right and actions/mode status in top left.
    Parameters: img (frame), fps (float)
    Returns: Void
    """
    fps_text    = f"FPS: {int(fps)}"
    (tw, _), _  = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(img, fps_text, (img.shape[1] - tw - 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    status_color = (0, 255, 0) if actions_enabled else (0, 0, 255)
    status_text  = "Actions: ON" if actions_enabled else "Actions: OFF"
    cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    if active_mode:
        mode_name, _, _, mode_color = MODES[active_mode]
        cv2.putText(img, f"Mode: {mode_name}", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

def process_continuous_actions():
    """Fires continuous actions from GESTURE_ACTIONS and active MODES based on frame counters.
    Parameters: Void
    Returns: Void
    """
    global repeat_frame_count

    repeat_frame_count += 1
    if repeat_frame_count >= Fast_Interval:
        repeat_frame_count = 0
        if actions_enabled:
            for label in last_confirmed:
                action = GESTURE_ACTIONS.get(last_confirmed[label])
                if callable(action) and getattr(action, "_continuous", False):
                    action()
                if active_mode == "Two Five" and last_confirmed[label] == "Pointing":
                    single_key(Key.space)

    if actions_enabled and active_mode:
        _, up_action, down_action, _ = MODES[active_mode]
        mode_name = MODES[active_mode][0]
        interval  = Slow_Interval if mode_name in ("Brightness", "Volume") else Fast_Interval

        mode_frame_counts[mode_name] += 1
        if mode_frame_counts[mode_name] >= interval:
            mode_frame_counts[mode_name] = 0
            for label in last_confirmed:
                gesture = last_confirmed[label]
                if gesture == "Thumbs Up":
                    up_action()
                elif gesture == "Thumbs Down":
                    down_action()

def handle_confirmed_gesture(gesture, label, facing):
    """Handles a newly confirmed gesture by triggering the appropriate action or mode change.
    Parameters: gesture (str), label (str), facing (str)
    Returns: Bool, True if the program should quit
    """
    global actions_enabled, active_mode

    print(f"{label} [{facing}]: {gesture}")
    last_confirmed[label] = gesture

    action = GESTURE_ACTIONS.get(gesture)
    if gesture == TOGGLE_GESTURE:
        actions_enabled = not actions_enabled
        print(f"[System]: Actions {'enabled' if actions_enabled else 'disabled'}")
    elif gesture == "Thumbs Sideways" and active_mode == "Four Fingers" and actions_enabled:
        toggle_mute()
    elif gesture in MODES:
        if active_mode == gesture:
            active_mode = None
            print(f"[Mode]: {MODES[gesture][0]} off")
        else:
            active_mode = gesture
            print(f"[Mode]: {MODES[gesture][0]} on")
    elif action == "QUIT":
        print("[System] Closing windows")
        return True
    elif callable(action) and not getattr(action, "_continuous", False) and actions_enabled:
        action()
    return False

def process_hands(img, result):
    """Processes all detected hands in the current frame, updates gesture buffers and draws labels.
    Parameters: img (frame), result (MediaPipe result)
    Returns: Bool, True if the program should quit
    """
    if not result.multi_hand_landmarks:
        last_confirmed["Right"] = ""
        last_confirmed["Left"]  = ""
        return False

    for i, (handLms, handedness) in enumerate(
        zip(result.multi_hand_landmarks, result.multi_handedness)
    ):
        mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

        label    = handedness.classification[0].label
        is_right = label == "Right"
        lm       = handLms.landmark
        facing   = get_hand_facing(lm, is_right)
        gesture  = detect_gesture(lm, is_right)

        gesture_buffers[label].append(gesture)
        if gesture_buffers[label].count(gesture) == Gesture_Confirm:
            if gesture != last_confirmed[label]:
                if handle_confirmed_gesture(gesture, label, facing):
                    return True

        gesture_text = f"{label} {facing}: {last_confirmed[label]}"
        y_pos = img.shape[0] - 20 - i * 40
        cv2.putText(img, gesture_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return False

# Main loop
running = True

while running:
    success, img = cap.read()
    if not success:
        continue

    img     = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result  = hands.process(img_rgb)

    curr_time = cv2.getTickCount()
    fps       = cv2.getTickFrequency() / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    draw_hud(img, fps)
    process_continuous_actions()

    if process_hands(img, result):
        break

    cv2.imshow("Hand Tracker v2", img)
    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
        break

# Cleanup
hands.close()
cap.release()
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)