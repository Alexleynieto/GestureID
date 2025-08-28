import base64
import io
import json
import threading
import time
import os
import hashlib
from typing import Optional, List

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox

import requests
import google.auth
from google.auth.transport.requests import Request

import tensorflow as tf
from tensorflow import keras 
import mediapipe as mp
from mediapipe.python.solutions import drawing_utils as du


# --------- Opcional (cargados perezosamente) ---------
LOCAL_MLP_MODEL = None
LOCAL_MLP_INPUT_DIM = None
LOCAL_LABELS = None
MP_HANDS_CTX = None
LOCAL_MODEL_PATH = "modelo_ASL_MLP_Landmarks.h5"
LOCAL_MIN_CONF = 0.70
LOCAL_SMOOTHING = 3

# Buffer circular para suavizado
_local_pred_buffer = []

# ----------------- Config -----------------
PROJECT_ID = "gestureid"
LOCATION = "europe-west4"
TARGET_SIZE = 512
MODEL_NAME = "gemini-2.5-pro"
USERS_DIR = os.path.join(os.path.dirname(__file__), 'users')  # carpeta donde viven los usuarios

PROMPT = (
    "Eres un sistema experto en reconocimiento gestual estático. Durante tu uso los usuarios te "
    "pasarán imágenes de manos realizando gestos del lenguaje de signos americano (ASL), del dataset "
    "de Kaggle. Entre estos encontramos 26 gestos de letras de la A a la Z, más 3 gestos adicionales, "
    "SPACE, DEL y NOTHING. IMPORTANTE: responde únicamente con las letras A-Z o los tokens SPACE, DEL, NOTHING."
)

SIMPLE_PROMPT = (
    "Eres un sistema experto en gestos del American Sign Language (ASL). Devuelve SOLO un token exacto: A-Z o SPACE o DEL o NOTHING. Nada más."
)

ACCENT = "#4CAF50"


def letterbox_to_512_from_bgr(frame: np.ndarray, target: int = TARGET_SIZE) -> Image.Image:
    h, w = frame.shape[:2]
    scale = min(target / w, target / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target, target, 3), dtype=np.uint8)
    x = (target - new_w) // 2
    y = (target - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_png_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def get_access_token_adc() -> Optional[str]:
    try:
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        if not creds.valid:
            creds.refresh(Request())
        return creds.token
    except Exception as e:
        print("ADC token error:", e)
        return None


def extract_text_from_known_paths(parsed):
    try:
        if not isinstance(parsed, dict):
            return None
        candidates = parsed.get("candidates")
        if isinstance(candidates, list) and candidates:
            first_c = candidates[0]
            if isinstance(first_c, dict):
                content = first_c.get("content")
                if isinstance(content, dict):
                    parts = content.get("parts")
                    if isinstance(parts, list) and parts:
                        first_part = parts[0]
                        if isinstance(first_part, dict) and "text" in first_part:
                            t = first_part.get("text")
                            if isinstance(t, str):
                                return t
    except Exception:
        pass
    return None


def find_first_string_in_json(el):
    if el is None:
        return None
    if isinstance(el, dict):
        t = el.get("text")
        if isinstance(t, str) and t.strip():
            return t
        if "parts" in el and isinstance(el["parts"], list):
            for p in el["parts"]:
                found = find_first_string_in_json(p)
                if found:
                    return found
        for k, v in el.items():
            if k.lower() == "role":
                continue
            found = find_first_string_in_json(v)
            if found:
                return found
    elif isinstance(el, list):
        for item in el:
            found = find_first_string_in_json(item)
            if found:
                return found
    elif isinstance(el, str):
        if el.strip():
            return el
    return None


def normalize_gesture(raw: Optional[str]) -> str:
    if raw is None:
        return "NOTHING"
    s_full = raw.strip()
    if not s_full:
        return "NOTHING"
    s = s_full.upper()
    if "SPACE" in s:
        return "SPACE"
    if "DEL" in s:
        return "DEL"
    if s == 'NOTHING':
        return 'NOTHING'
    import re
    pat = re.search(r"(?:LETTER|LETRA|SIGN|SIGNO)[^A-Z]*([A-Z])\b", s)
    if pat:
        c = pat.group(1)
        if 'A' <= c <= 'Z':
            return c
    isolated = re.findall(r"\b([A-Z])\b", s)
    if len(isolated) == 1 and 'A' <= isolated[0] <= 'Z':
        return isolated[0]
    letters = re.findall(r"[A-Z]", s)
    uniq = set(letters)
    if len(letters) == 1:
        return letters[0]
    if len(uniq) == 1 and 1 < len(letters) <= 4:
        return letters[0]
    trail = re.search(r"\bIS\s+([A-Z])\b", s)
    if trail:
        c = trail.group(1)
        if 'A' <= c <= 'Z':
            return c
    return "NOTHING"


def call_vertex_generate(png_bytes: bytes, model_name: Optional[str] = None, system_prompt: str = PROMPT) -> Optional[str]:
    token = get_access_token_adc()
    if not token:
        print("No ADC token")
        return None
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    user_image_part = {"inlineData": {"mimeType": "image/png", "data": b64}}
    user_text_part = {"text": "Identifica el gesto de la imagen."}
    parts = [user_image_part, user_text_part]
    contents = [{"role": "user", "parts": parts}]
    body = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.1,
            "topP": 0.95,
            "candidateCount": 1,
            "maxOutputTokens": 4096,
        },
    "systemInstruction": {"parts": [{"text": system_prompt}]},
    }
    mdl = model_name or MODEL_NAME
    url = (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/"
        f"{LOCATION}/publishers/google/models/{mdl}:generateContent"
    )
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=60)
    except Exception as e:
        print("Request error:", e)
        return None
    if resp.status_code != 200:
        print("Vertex API error", resp.status_code, resp.text[:200])
        return None
    try:
        parsed = resp.json()
    except Exception:
        return normalize_gesture(resp.text)
    extracted = extract_text_from_known_paths(parsed)
    if extracted:
        print(f"[Vertex] primary text: {extracted!r}")
        return normalize_gesture(extracted)
    found = find_first_string_in_json(parsed)
    if found:
        print(f"[Vertex] fallback text: {found!r}")
        return normalize_gesture(found)
    dumped = json.dumps(parsed)
    print(f"[Vertex] dump length={len(dumped)}")
    return normalize_gesture(dumped)


def _lazy_load_local_dependencies():
    """Carga del modelo MLP entrenado."""

    global LOCAL_MLP_MODEL, LOCAL_MLP_INPUT_DIM, LOCAL_LABELS, MP_HANDS_CTX
    if LOCAL_MLP_MODEL is not None and MP_HANDS_CTX is not None:
        return True
    
    # Cargar modelo
    try:
        load_error = None
        try:
            LOCAL_MLP_MODEL = keras.models.load_model(LOCAL_MODEL_PATH, compile=False)
        except Exception as e_simple:

            # Si el modelo no existe mostrar error
            load_error = e_simple
            print(f"[LocalModel] Error cargando modelo (intento simple): {load_error}")
            LOCAL_MLP_MODEL = None
            return False

        try:
            LOCAL_MLP_INPUT_DIM = LOCAL_MLP_MODEL.input_shape[1]
        except Exception:
            try:
                LOCAL_MLP_INPUT_DIM = LOCAL_MLP_MODEL.layers[0].input_shape[-1]
            except Exception:
                LOCAL_MLP_INPUT_DIM = None
        print(f"[LocalModel] Modelo cargado '{LOCAL_MODEL_PATH}' (input_dim={LOCAL_MLP_INPUT_DIM})")

        LOCAL_LABELS = [
            'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
            'DEL','NOTHING','SPACE'
        ]
        print(f"[LocalModel] Etiquetas hardcoded ({len(LOCAL_LABELS)})")

        try:
            out_units = LOCAL_MLP_MODEL.output_shape[-1]
            if out_units != len(LOCAL_LABELS):
                print(f"[LocalModel][WARN] output_units={out_units} != num_labels={len(LOCAL_LABELS)}")
        except Exception:
            pass
    except Exception as e:
        print(f"[LocalModel] No se pudo cargar el modelo '{LOCAL_MODEL_PATH}': {e}")
        LOCAL_MLP_MODEL = None
        return False
    
    try:
        MP_HANDS_CTX = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1,
                                               min_detection_confidence=0.5, min_tracking_confidence=0.5)
    except Exception as e:
        print(f"[LocalModel] Error iniciando MediaPipe Hands: {e}")
        MP_HANDS_CTX = None
        return False
    return True


def _extract_hand_landmarks(frame_bgr) -> Optional[np.ndarray]:
    """Extrae vector de características desde frame BGR usando MediaPipe.

    Intenta adaptarse a la dimensión requerida por el MLP:
    - Si input_dim == 63 => usa (x,y,z) * 21 landmarks
    - Si input_dim == 42 => usa (x,y) * 21 landmarks
    - Si input_dim >= 63 y no es múltiplo exacto, rellena con ceros
    """
    global MP_HANDS_CTX, LOCAL_MLP_INPUT_DIM
    if MP_HANDS_CTX is None:
        return None
    
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = MP_HANDS_CTX.process(rgb)
    if not res.multi_hand_landmarks:
        return None
    hand = res.multi_hand_landmarks[0]
    coords = []
    for lm in hand.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    vec = np.array(coords, dtype='float32')
    if LOCAL_MLP_INPUT_DIM is None:
        return None
    if LOCAL_MLP_INPUT_DIM == vec.shape[0]:
        return vec
    if LOCAL_MLP_INPUT_DIM == 42:
        xy = []
        for i in range(0, len(coords), 3):
            xy.extend(coords[i:i + 2])
        return np.array(xy, dtype='float32')
    
    # Si mayor, rellenar
    if LOCAL_MLP_INPUT_DIM > vec.shape[0]:
        pad = np.zeros((LOCAL_MLP_INPUT_DIM - vec.shape[0],), dtype='float32')
        return np.concatenate([vec, pad])

    # Si menor, recortar
    return vec[:LOCAL_MLP_INPUT_DIM]


def call_local_model_from_frame(frame_bgr: np.ndarray) -> Optional[str]:
    """Pipeline completo de inferencia local desde frame BGR."""
    global LOCAL_MLP_MODEL, LOCAL_LABELS, _local_pred_buffer

    if frame_bgr is None:
        return None
    
    if not _lazy_load_local_dependencies():
        return None
    feats = _extract_hand_landmarks(frame_bgr)

    if feats is None:
        return "NOTHING"

    try:
        feats = feats.reshape(1, -1)
        preds = LOCAL_MLP_MODEL.predict(feats, verbose=0)[0]
        idx = int(np.argmax(preds))
        prob = float(preds[idx])
        if idx >= len(LOCAL_LABELS):
            return "NOTHING"
        raw_label = LOCAL_LABELS[idx]

        if raw_label == 'del':
            token = 'DEL'
        elif raw_label == 'space':
            token = 'SPACE'
        elif raw_label == 'nothing':
            token = 'NOTHING'
        else:
            token = raw_label.upper()
        if prob < LOCAL_MIN_CONF:
            token = 'NOTHING'

        _local_pred_buffer.append(token)
        if len(_local_pred_buffer) > LOCAL_SMOOTHING:
            _local_pred_buffer = _local_pred_buffer[-LOCAL_SMOOTHING:]
        vals, counts = np.unique(_local_pred_buffer, return_counts=True)
        smooth_token = vals[int(np.argmax(counts))]
        return smooth_token
    except Exception as e:
        print(f"[LocalModel] Error durante la inferencia: {e}")
        return None


class GestureLoginApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("GestureID")
        try:
            self.root.attributes('-fullscreen', True)
            self.root.bind('<Escape>', lambda e: self.root.attributes('-fullscreen', False))
        except Exception:
            try:
                self.root.state('zoomed')
            except Exception:
                self.root.geometry('1200x800')
        self.root.minsize(900, 600)
        self.root.configure(bg="#f0f2f5")

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame = None
        self.running = False
        self.mode = 'GESTOS'

        self.username_var = tk.StringVar()
        self.password_var = tk.StringVar()
        self.last_pred_var = tk.StringVar(value='-')
        self.dev_mode = False
        self.selected_model = MODEL_NAME

        self.cam_size = (640, 480)

        # ---- Estado de registro ----
        self.registering = False
        self.reg_images: List[Image.Image] = []
        self.reg_preds: List[str] = []

        self.build_ui()
        self._bind_space()
        self.start_camera()
        try:
            self.username_entry.focus_set()
        except Exception:
            pass

    def build_ui(self):
        # Título
        tk.Label(self.root, text="GestureID", font=("Helvetica", 28, 'bold'), bg=self.root.cget('bg')).pack(pady=(12, 6))

        # Contenedor principal
        self.container = tk.Frame(self.root, bg=self.root.cget('bg'))
        self.container.pack(fill=tk.BOTH, expand=True, padx=20, pady=8)
        for i in range(3):
            self.container.grid_columnconfigure(i, weight=1)

        # Tarjeta de login
        self.card = tk.Frame(self.container, bg='white', bd=1, relief='groove', width=420, height=300)
        self.card.grid_propagate(False)
        self.card.grid(row=0, column=1, sticky='n', padx=6, pady=(10, 0))

        ttk.Label(self.card, text='Usuario:', font=('Arial', 12)).pack(anchor='w', padx=12, pady=(10, 0))
        self.username_entry = ttk.Entry(self.card, textvariable=self.username_var, width=34)
        self.username_entry.pack(padx=12, pady=(6, 8))
        self.pass_label = ttk.Label(self.card, text='Contraseña gestual:', font=('Arial', 12))
        self.pass_label.pack(anchor='w', padx=12, pady=(2, 0))
        self.password_entry = ttk.Entry(self.card, textvariable=self.password_var, width=34, show='*', state='readonly')
        self.password_entry.pack(padx=12, pady=(6, 8))

        mode_row = tk.Frame(self.card, bg='white')
        mode_row.pack(pady=(4, 4))
        self.gesture_btn = tk.Button(mode_row, text='Gestos', width=12, command=lambda: self.set_mode('GESTOS'))
        self.pass_btn = tk.Button(mode_row, text='Contraseña', width=12, command=lambda: self.set_mode('CONTRASEÑA'))
        self.gesture_btn.pack(side=tk.LEFT, padx=6)
        self.pass_btn.pack(side=tk.LEFT, padx=6)

        actions = tk.Frame(self.card, bg='white')
        actions.pack(pady=(4, 4))
        self.capture_btn = tk.Button(actions, text='Capturar (Espacio)', command=self.capture)
        self.capture_btn.pack(side=tk.LEFT, padx=(0, 8))

        # Botón borrar última letra de contraseña gestual
        self.delete_last_btn = tk.Button(actions, text='Borrar', command=self.delete_last_gesture_char)
        self.delete_last_btn.pack(side=tk.LEFT, padx=4)
        tk.Button(actions, text='Login', command=self.login).pack(side=tk.LEFT, padx=4)
        tk.Button(actions, text='REGISTRO', command=self.open_register_card).pack(side=tk.LEFT, padx=4)
        tk.Button(actions, text='Salir', command=self.quit).pack(side=tk.LEFT, padx=4)

        # Tarjeta de registro
        self.reg_card = tk.Frame(self.container, bg='white', bd=1, relief='groove', width=760, height=640)
        self.reg_card.grid_propagate(False)

        # Columna derecha
        self.right_col = tk.Frame(self.container, bg=self.root.cget('bg'))
        self.right_col.grid(row=0, column=2, rowspan=2, sticky='n', padx=(12, 20), pady=10)

        # Webcam
        self.webcam_holder = tk.Frame(self.right_col, width=self.cam_size[0], height=self.cam_size[1], bg='#000')
        self.webcam_holder.pack_propagate(False)
        self.webcam_holder.pack(side=tk.TOP)
        self.webcam_label = tk.Label(self.webcam_holder, text='NO CAM', fg='white', bg='#000', font=('Helvetica', 14))
        self.webcam_label.pack(fill=tk.BOTH, expand=True)

        # Preview (última captura / dev)
        self.preview_holder = tk.Frame(self.right_col, width=self.cam_size[0], height=self.cam_size[1], bg='#222')
        self.preview_holder.pack_propagate(False)

        # Inicialmente oculto hasta activar DEV Mode
        self.preview_label = tk.Label(self.preview_holder, text='Sin captura', fg='white', bg='#222')
        self.preview_label.place(relx=0.5, rely=0.5, anchor='center')

        # Botón toggle dev
        self.dev_toggle_btn = tk.Button(self.root, text='DEV Mode', width=12, command=self.toggle_dev_mode)
        self.dev_toggle_btn.place(relx=0.01, rely=0.98, anchor='sw')

        # Tarjeta DEV
        self.dev_card = tk.Frame(self.container, bg='white', bd=1, relief='groove', width=420, height=200)
        self.dev_card.grid_propagate(False)
        ttk.Label(self.dev_card, text='Dev Tools', font=('Arial', 10, 'bold')).pack(anchor='w', padx=8, pady=(6, 0))
        pred_row = ttk.Frame(self.dev_card)
        pred_row.pack(padx=8, pady=(6, 4), fill=tk.X)
        ttk.Label(pred_row, text='Último gesto:').pack(side=tk.LEFT)
        self.dev_pred_label = ttk.Label(pred_row, textvariable=self.last_pred_var, font=('Arial', 12, 'bold'))
        self.dev_pred_label.pack(side=tk.LEFT, padx=6)
        ttk.Label(self.dev_card, text='Modelo:', font=('Arial', 10)).pack(anchor='w', padx=8)
        self.model_buttons = {}
        models_row = ttk.Frame(self.dev_card)
        models_row.pack(padx=8, pady=(4, 6))
        for key in ["gemini-2.5-pro", "gemini-2.5-pro-simple", "gemini-2.5-flash", "local"]:
            b = tk.Button(models_row, text=key.split('-')[-1], width=12, command=lambda k=key: self.select_model(k))
            b.pack(side=tk.LEFT, padx=4)
            self.model_buttons[key] = b
        self._refresh_model_buttons()

        # Estilos iniciales
        self._style_mode_buttons()
        self._refresh_mode_buttons()

        try:
            self.dev_card.grid_forget()
        except Exception:
            pass

    def delete_last_gesture_char(self):
        if self.mode != 'GESTOS':
            return
        cur = self.password_var.get()
        if cur:
            self.password_var.set(cur[:-1])

    def _set_busy(self, busy: bool):
        try:
            if busy:
                self.root.config(cursor='watch')
            else:
                self.root.config(cursor='')
            self.root.update_idletasks()
        except Exception:
            pass

    def _style_mode_buttons(self):
        for btn in (self.gesture_btn, self.pass_btn):
            try:
                btn.configure(bg='#e6e6e6', activebackground='#d0d0d0', bd=1, relief=tk.RAISED)
            except Exception:
                pass

    def _refresh_mode_buttons(self):
        active = {'bg': ACCENT, 'fg': 'white', 'relief': tk.SUNKEN}
        idle = {'bg': '#e6e6e6', 'fg': 'black', 'relief': tk.RAISED}
        if self.mode == 'GESTOS':
            self.gesture_btn.config(**active)
            self.pass_btn.config(**idle)
            self.password_entry.configure(state='readonly')
            self.pass_label.config(text='Contraseña gestual:')
            try:
                self.delete_last_btn.config(state='normal')
            except Exception:
                pass
        else:
            self.pass_btn.config(**active)
            self.gesture_btn.config(**idle)
            self.password_entry.configure(state='normal')
            self.pass_label.config(text='Contraseña:')
            try:
                self.delete_last_btn.config(state='disabled')
            except Exception:
                pass

    def set_mode(self, mode: str):
        if mode not in ('GESTOS', 'CONTRASEÑA'):
            return
        if self.mode == mode:
            return
        self.mode = mode
        try:
            self.password_var.set('')
        except Exception:
            pass
        if mode == 'CONTRASEÑA':
            try:
                self.card.grid_configure(column=0, columnspan=3)
            except Exception:
                pass
            try:
                self.right_col.grid_remove()
            except Exception:
                pass
            self.password_entry.configure(state='normal')
            self.stop_camera()
            try:
                self.capture_btn.config(state='disabled')
            except Exception:
                pass
            if self.dev_mode:
                self.toggle_dev_mode(force=False)
            try:
                self.dev_toggle_btn.place_forget()
            except Exception:
                pass
        else:
            try:
                self.card.grid_configure(column=1, columnspan=1)
            except Exception:
                pass
            try:
                self.right_col.grid()
            except Exception:
                pass
            self.password_entry.configure(state='readonly')
            if not self.running:
                self.start_camera()
            try:
                self.capture_btn.config(state='normal')
            except Exception:
                pass
            try:
                self.dev_toggle_btn.place(relx=0.01, rely=0.98, anchor='sw')
            except Exception:
                pass
        self._refresh_mode_buttons()

    def _on_space(self, event=None):
        if self.registering:
            self.registration_capture()
            return 'break'
        if self.mode == 'GESTOS':
            self.capture()
            return 'break'
        return None

    def _bind_space(self):
        try:
            self.root.bind('<space>', self._on_space)
        except Exception:
            pass

    def start_camera(self):
        if self.cap is None:
            attempts = [(0, cv2.CAP_MSMF), (0, cv2.CAP_DSHOW), (0, cv2.CAP_VFW), (0, cv2.CAP_ANY), (1, cv2.CAP_ANY), (0, 0)]
            for idx, backend in attempts:
                try:
                    cap = cv2.VideoCapture(idx, backend) if backend else cv2.VideoCapture(idx)
                    time.sleep(0.25)
                    if cap.isOpened():
                        self.cap = cap
                        print(f"[Camera] abierta (idx={idx}, backend={backend})")
                        break
                    cap.release()
                except Exception:
                    pass
            if self.cap is None:
                self.root.after(300, lambda: (messagebox.showerror('Cámara', 'No se pudo abrir la cámara'), self._refocus_user()))
                return
        if not self.running:
            self.running = True
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running and self.cap and self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                break
            self.frame = frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(pil.resize(self.cam_size))
            def upd():
                try:
                    self.webcam_label.config(image=imgtk, text='')
                    self.webcam_label.imgtk = imgtk
                except Exception:
                    pass
            self.root.after(0, upd)
            time.sleep(1 / 30)

    def stop_camera(self):
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def capture(self):
        if self.registering:
            self.registration_capture()
            return
        if self.mode != 'GESTOS' or self.frame is None:
            return
        letterboxed = letterbox_to_512_from_bgr(self.frame)
        png_bytes = pil_to_png_bytes(letterboxed)
        self.last_pred_var.set('...')
        self._set_busy(True)

        if self.dev_mode:
            # Si es modelo local, intentar dibujar landmarks
            if self.selected_model == 'local':
                pil_with_lm = self._make_landmark_preview(self.frame) or Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
                self._update_preview(pil_with_lm)
            else:
                try:
                    raw_pil = Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
                except Exception:
                    raw_pil = letterboxed
                self._update_preview(raw_pil)
        threading.Thread(target=self._infer_thread, args=(png_bytes,), daemon=True).start()

    def _infer_thread(self, png_bytes: bytes):
        key = self.selected_model
        if key == 'local':
            frame_copy = None
            try:
                if self.frame is not None:
                    frame_copy = self.frame.copy()
            except Exception:
                pass
            pred = call_local_model_from_frame(frame_copy) if frame_copy is not None else 'NOTHING'
        else:
            if key == 'gemini-2.5-pro-simple':
                model_name = 'gemini-2.5-pro'
                system_prompt = SIMPLE_PROMPT
            elif key == 'gemini-2.5-flash':
                model_name = 'gemini-2.5-flash'
                system_prompt = PROMPT
            else:
                model_name = 'gemini-2.5-pro'
                system_prompt = PROMPT
            pred = call_vertex_generate(png_bytes, model_name=model_name, system_prompt=system_prompt)
        if pred is None:
            pred = 'NOTHING'
        if pred and pred != 'NOTHING':
            cur = self.password_var.get()
            if pred == 'DEL':
                cur = cur[:-1]
            elif pred == 'SPACE':
                cur += ' '
            else:
                cur += pred
            self.password_var.set(cur)
        self.root.after(0, lambda: self.last_pred_var.set(pred))
        self.root.after(0, lambda: self._set_busy(False))

    def toggle_dev_mode(self, force: Optional[bool] = None):
        new_state = (not self.dev_mode) if force is None else force
        if new_state == self.dev_mode:
            return
        self.dev_mode = new_state
        if self.dev_mode:
            if not self.preview_holder.winfo_manager():
                self.preview_holder.pack(side=tk.TOP, pady=(18, 0))
            self.dev_card.grid(row=1, column=1, sticky='n', padx=6, pady=(25, 10))
            self.dev_toggle_btn.config(relief=tk.SUNKEN, bg=ACCENT, fg='white')
        else:
            self.dev_card.grid_forget()
            try:
                if self.preview_holder.winfo_manager():
                    self.preview_holder.pack_forget()
            except Exception:
                pass
            self.dev_toggle_btn.config(relief=tk.RAISED, bg=self.root.cget('bg'), fg='black')

    def _update_preview(self, pil_img: Image.Image):
        try:
            target_w, target_h = self.cam_size
            im = pil_img.copy()

            # Llenar completamente recortando sobrante
            ratio = max(target_w / im.width, target_h / im.height)
            new_w, new_h = int(im.width * ratio), int(im.height * ratio)
            if (new_w, new_h) != (im.width, im.height):
                im = im.resize((new_w, new_h), Image.LANCZOS)
            left = max(0, (new_w - target_w) // 2)
            top = max(0, (new_h - target_h) // 2)
            im = im.crop((left, top, left + target_w, top + target_h))
            imgtk = ImageTk.PhotoImage(im)
            self.preview_label.config(image=imgtk, text='')
            self.preview_label.imgtk = imgtk
        except Exception:
            pass

    def _make_landmark_preview(self, frame_bgr) -> Optional[Image.Image]:
        """Genera una imagen PIL con landmarks dibujados (si MediaPipe está disponible)."""
        try:
            if frame_bgr is None:
                return None
            # Asegurar dependencias cargadas
            if not _lazy_load_local_dependencies():
                return None
            
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = MP_HANDS_CTX.process(rgb)
            if not res.multi_hand_landmarks:
                return Image.fromarray(rgb)
            annotated = rgb.copy()

            try:
                
                for hand_landmarks in res.multi_hand_landmarks:
                    du.draw_landmarks(
                        annotated,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=du.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                        connection_drawing_spec=du.DrawingSpec(color=(255,140,0), thickness=2, circle_radius=2)
                    )
            except Exception:

                hand = res.multi_hand_landmarks[0]
                h, w = annotated.shape[:2]
                for lm in hand.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(annotated, (x, y), 4, (0, 255, 0), -1)
            return Image.fromarray(annotated)
        except Exception as e:
            print(f"[LocalModel] Error generando preview landmarks: {e}")
            return None

    def select_model(self, key: str):
        self.selected_model = key
        self._refresh_model_buttons()

    def _refresh_model_buttons(self):
        for k, btn in getattr(self, 'model_buttons', {}).items():
            if k == self.selected_model:
                btn.config(bg=ACCENT, fg='white', relief=tk.SUNKEN)
            else:
                btn.config(bg='#e6e6e6', fg='black', relief=tk.RAISED)

    def login(self):
        user = self.username_var.get().strip()
        entered = self.password_var.get()
        if not user:
            messagebox.showerror('Login', 'Introduce un usuario'); return
        if not entered:
            messagebox.showwarning('Login', 'Introduce la contraseña'); return
        
        # cargar metadata
        meta_path = os.path.join(USERS_DIR, user, 'metadata.json')
        if not os.path.exists(meta_path):
            messagebox.showerror('Login', 'Usuario no encontrado'); return
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        except Exception:
            messagebox.showerror('Login', 'Error leyendo metadata'); return
        if self.mode == 'CONTRASEÑA':
            salt = meta.get('password_salt','')
            stored_hash = meta.get('password_hash','')
            h = hashlib.sha256((salt + entered).encode('utf-8')).hexdigest()
            if h != stored_hash:
                messagebox.showerror('Login', 'Contraseña incorrecta')
                return
            messagebox.showinfo('Login', f'Bienvenido {user}! (login contraseña)')
        else:
            # Gestos
            stored_joined = meta.get('gesture_password_joined','')
            if entered != stored_joined:
                messagebox.showerror('Login', 'Contraseña gestual incorrecta')
                return
            messagebox.showinfo('Login', f'Bienvenido {user}! (login gestual)')

    # ===================== REGISTRO =====================
    def open_register_card(self):
        if self.registering:
            return
        self.registering = True
        self.reg_images.clear()
        self.reg_preds.clear()

        # Ocultar tarjeta principal de login
        try:
            self.card.grid_forget()
        except Exception:
            pass

        # Mostrar u ocultar dev card según estado actual de dev_mode
        if self.dev_mode:
            try:
                self.dev_card.grid(row=0, column=2, sticky='n', padx=6, pady=(10,0))
            except Exception:
                pass
        else:
            try:
                self.dev_card.grid_forget()
            except Exception:
                pass

        # Limpiar contenido anterior del panel de registro
        for w in self.reg_card.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass

        # Construir UI de registro
        tk.Label(self.reg_card, text='Registro de Usuario', font=('Helvetica', 18, 'bold'), bg='white').pack(pady=(10,4))
        form = tk.Frame(self.reg_card, bg='white')
        form.pack(pady=4)
        tk.Label(form, text='Usuario:', bg='white').grid(row=0, column=0, sticky='e', padx=4, pady=2)
        self.reg_user_entry = ttk.Entry(form, width=28)
        self.reg_user_entry.grid(row=0, column=1, padx=4, pady=2)
        tk.Label(form, text='Contraseña:', bg='white').grid(row=1, column=0, sticky='e', padx=4, pady=2)
        self.reg_pass_entry = ttk.Entry(form, width=28, show='*')
        self.reg_pass_entry.grid(row=1, column=1, padx=4, pady=2)
        tk.Label(form, text='Confirmar:', bg='white').grid(row=2, column=0, sticky='e', padx=4, pady=2)
        self.reg_pass2_entry = ttk.Entry(form, width=28, show='*')
        self.reg_pass2_entry.grid(row=2, column=1, padx=4, pady=2)

        tk.Label(self.reg_card, text='Gestos capturados (mínimo 3):', bg='white').pack(anchor='w', padx=12, pady=(8,2))
        self.reg_listbox = tk.Listbox(self.reg_card, height=8, width=56)
        self.reg_listbox.pack(padx=12, pady=(0,6))
        self.reg_listbox.bind('<<ListboxSelect>>', self._on_reg_list_select)

        # Holder fijo para preview grande
        self.reg_preview_holder = tk.Frame(self.reg_card, width=660, height=500, bg='#222')
        self.reg_preview_holder.pack_propagate(False)
        self.reg_preview_holder.pack(padx=12, pady=4)
        self.reg_preview = tk.Label(self.reg_preview_holder, text='(sin gestos)', bg='#222', fg='white')
        self.reg_preview.place(relx=0.5, rely=0.5, anchor='center')

        btns = tk.Frame(self.reg_card, bg='white')
        btns.pack(pady=6)
        tk.Button(btns, text='Capturar (ESPACIO)', command=self.registration_capture).pack(side=tk.LEFT, padx=4)
        tk.Button(btns, text='Borrar', command=self.registration_delete_last).pack(side=tk.LEFT, padx=4)
        tk.Button(btns, text='Finalizar', command=self.finish_registration).pack(side=tk.LEFT, padx=4)
        tk.Button(btns, text='Cancelar', command=self.cancel_registration).pack(side=tk.LEFT, padx=4)

        # Colocar tarjeta de registro en el grid
        self.reg_card.grid(row=0, column=1, sticky='n', padx=6, pady=(10,0))

        # Enfocar campo usuario
        try:
            self.reg_user_entry.focus_set()
        except Exception:
            pass

        # Asegurar cámara encendida
        if not self.running:
            self.start_camera()

    def _validate_registration_fields(self, username: str, pw: str, pw2: str) -> bool:
        # Validaciones básicas de registro
        if not username:
            messagebox.showerror('Registro', 'Nombre de usuario inválido')
            return False
        if any(c in username for c in ('/', '\\', '..')):
            messagebox.showerror('Registro', 'Nombre de usuario inválido')
            return False
        if not pw or not pw2 or pw != pw2:
            messagebox.showerror('Registro', 'Las contraseñas no coinciden')
            return False
        if len(pw) < 5 or not pw.isalnum():
            messagebox.showerror('Registro', 'La contraseña debe ser alfanumérica y de al menos 5 caracteres')
            return False
        user_dir = os.path.join(USERS_DIR, username)
        # Solo comprobar existencia al inicio; permitir capturas subsiguientes aunque ya se haya validado
        if self.reg_preds == [] and os.path.exists(user_dir):
            messagebox.showerror('Registro', 'El usuario ya existe')
            return False
        return True

    def registration_capture(self):
        if not self.registering or self.frame is None:
            return
        username = self.reg_user_entry.get().strip()
        pw = self.reg_pass_entry.get()
        pw2 = self.reg_pass2_entry.get()
        if not self._validate_registration_fields(username, pw, pw2):
            return
        
        # Inferir gesto
        letterboxed = letterbox_to_512_from_bgr(self.frame)
        png_bytes = pil_to_png_bytes(letterboxed)
        raw_pil = Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
        self._set_busy(True)
        def infer():
            key = self.selected_model
            if key == 'local':
                pred = call_local_model_from_frame(self.frame.copy()) if self.frame is not None else 'NOTHING'
            else:
                if key == 'gemini-2.5-pro-simple':
                    mname, sp = 'gemini-2.5-pro', SIMPLE_PROMPT
                elif key == 'gemini-2.5-flash':
                    mname, sp = 'gemini-2.5-flash', PROMPT
                else:
                    mname, sp = 'gemini-2.5-pro', PROMPT
                pred = call_vertex_generate(png_bytes, model_name=mname, system_prompt=sp)
            if pred is None:
                pred = 'NOTHING'
            def after():
                if pred == 'NOTHING':
                    messagebox.showwarning('Registro', 'Gesto NOTHING. Intenta de nuevo.')
                    self._set_busy(False)
                    return
                self.reg_images.append(raw_pil)
                self.reg_preds.append(pred)
                self.reg_listbox.insert(tk.END, f"Gesto_{len(self.reg_preds)}")
                self._update_reg_preview_image(len(self.reg_images)-1)

                # Actualizar DEV card si activo
                if self.dev_mode:
                    try:
                        self.last_pred_var.set(pred)

                        # Reutilizar preview dev (panel derecho) con imagen capturada
                        self._update_preview(raw_pil)
                    except Exception:
                        pass
                self._set_busy(False)
            self.root.after(0, after)
        threading.Thread(target=infer, daemon=True).start()

    def registration_delete_last(self):
        if not self.registering or not self.reg_preds:
            return
        self.reg_images.pop()
        self.reg_preds.pop()
        try:
            self.reg_listbox.delete(tk.END)
        except Exception:
            pass
        if not self.reg_preds:
            self.reg_preview.config(image='', text='(sin gestos)')
        else:

            # mostrar ahora el último restante
            self._update_reg_preview_image(len(self.reg_images)-1)

    def _on_reg_list_select(self, event):
        if not self.registering:
            return
        try:
            sel = self.reg_listbox.curselection()
            if not sel:
                return
            idx = sel[0]
            if 0 <= idx < len(self.reg_images):
                self._update_reg_preview_image(idx)
        except Exception:
            pass

    def _update_reg_preview_image(self, idx: int):
        try:
            img = self.reg_images[idx]
            big = img.copy()
            # tamaño mayor
            big.thumbnail((480, 360), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(big)
            self.reg_preview.config(image=imgtk, text='')
            self.reg_preview.imgtk = imgtk
        except Exception:
            pass

    def finish_registration(self):
        if not self.registering:
            return
        username = self.reg_user_entry.get().strip()
        pw = self.reg_pass_entry.get()
        pw2 = self.reg_pass2_entry.get()
        if not self._validate_registration_fields(username, pw, pw2):
            return
        if len(self.reg_preds) < 3:
            messagebox.showerror('Registro', 'Debes capturar al menos 3 gestos válidos')
            return
        
        # Crear carpeta usuario
        try:
            os.makedirs(USERS_DIR, exist_ok=True)
            user_dir = os.path.join(USERS_DIR, username)
            os.makedirs(user_dir, exist_ok=False)
        except FileExistsError:
            messagebox.showerror('Registro', 'El usuario ya existe')
            return
        except Exception as e:
            messagebox.showerror('Registro', f'No se pudo crear carpeta: {e}')
            return
        
        # Guardar imágenes
        gesture_filenames = []
        for idx, img in enumerate(self.reg_images, start=1):
            fname = f'gesture_{idx}.png'
            path = os.path.join(user_dir, fname)
            try:
                img.save(path)
                gesture_filenames.append(fname)
            except Exception as e:
                messagebox.showerror('Registro', f'Error guardando {fname}: {e}')
                return
        salt = os.urandom(16).hex()
        pwd_hash = hashlib.sha256((salt + pw).encode('utf-8')).hexdigest()
        metadata = {
            'username': username,
            'password_salt': salt,
            'password_hash': pwd_hash,
            'gestures': gesture_filenames,
            'gesture_password': self.reg_preds,
            'gesture_password_joined': ''.join(self.reg_preds)
        }
        try:
            with open(os.path.join(user_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            messagebox.showerror('Registro', f'Error guardando metadata: {e}')
            return
        messagebox.showinfo('Registro', f'Usuario "{username}" registrado correctamente')
        self.cancel_registration(after_success=True)

    def cancel_registration(self, after_success: bool = False):
        if not self.registering:
            return
        self.registering = False
        self.reg_images.clear()
        self.reg_preds.clear()
        try:
            self.reg_card.grid_forget()
        except Exception:
            pass
        
        # mostrar card principal nuevamente
        try:
            self.card.grid(row=0, column=1, sticky='n', padx=6, pady=(10,0))
        except Exception:
            pass
        if self.dev_mode:
            self.dev_card.grid(row=1, column=1, sticky='n', padx=6, pady=(25, 10))
        if not after_success:
            messagebox.showinfo('Registro', 'Registro cancelado')
        try:
            self.username_entry.focus_set()
        except Exception:
            pass

    def _refocus_user(self):
        try:
            self.username_entry.focus_set()
        except Exception:
            pass

    def quit(self):
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        try:
            self.root.destroy()
        except Exception:
            pass

if __name__ == "__main__":
    root = tk.Tk()
    app = GestureLoginApp(root)
    root.mainloop()
