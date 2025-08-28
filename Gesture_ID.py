import base64
import io
import json
import threading
import time
import os
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox

import requests
import google.auth
from google.auth.transport.requests import Request


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
    """Carga perezosa TensorFlow, MediaPipe y el modelo MLP entrenado."""
    global LOCAL_MLP_MODEL, LOCAL_MLP_INPUT_DIM, LOCAL_LABELS, MP_HANDS_CTX
    if LOCAL_MLP_MODEL is not None and MP_HANDS_CTX is not None:
        return True
    try:
        import tensorflow as tf  # noqa: F401
    except Exception as e:
        print(f"[LocalModel] TensorFlow no disponible: {e}")
        return False
    try:
        import mediapipe as mp  # noqa: F401
    except Exception as e:
        print(f"[LocalModel] MediaPipe no disponible: {e}")
        return False
    # Cargar modelo
    try:
        from tensorflow import keras  # type: ignore
        load_error = None
        # 1. Intentar carga simple (MLP estándar)
        try:
            LOCAL_MLP_MODEL = keras.models.load_model(LOCAL_MODEL_PATH, compile=False)
        except Exception as e_simple:
            load_error = e_simple
            # 2. Fallback: registrar capas GCN solo si realmente se requiere
            try:
                import tensorflow as tf
                NUM_LANDMARKS = 21
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (0, 9), (9, 10), (10, 11), (11, 12),
                    (0, 13), (13, 14), (14, 15), (15, 16),
                    (0, 17), (17, 18), (18, 19), (19, 20),
                    (5, 9), (9, 13), (13, 17)
                ]
                A = np.zeros((NUM_LANDMARKS, NUM_LANDMARKS), dtype=np.float32)
                for i, j in connections:
                    A[i, j] = 1; A[j, i] = 1
                A += np.eye(NUM_LANDMARKS, dtype=np.float32)
                D = np.diag(np.sum(A, axis=1)); D_inv_sqrt = np.linalg.inv(np.sqrt(D)); norm_adj = D_inv_sqrt @ A @ D_inv_sqrt
                NORM_ADJ_TF = tf.constant(norm_adj, dtype=tf.float32)
                class GraphConvLayer(keras.layers.Layer):
                    def __init__(self, units, **kwargs):
                        super().__init__(**kwargs); self.units = units
                    def build(self, input_shape):
                        in_dim = int(input_shape[-1])
                        self.w = self.add_weight(name='w', shape=(in_dim, self.units), initializer='glorot_uniform', trainable=True)
                        super().build(input_shape)
                    def call(self, inputs):
                        xw = tf.matmul(inputs, self.w)
                        return tf.matmul(tf.expand_dims(NORM_ADJ_TF, 0), xw)
                    def get_config(self):
                        c = super().get_config(); c.update({'units': self.units}); return c
                class AttentionLayer(keras.layers.Layer):
                    def __init__(self, units, **kwargs):
                        super().__init__(**kwargs); self.units = units
                    def build(self, input_shape):
                        self.q_dense = keras.layers.Dense(self.units, activation='tanh')
                        self.k_dense = keras.layers.Dense(self.units, activation='softmax')
                        super().build(input_shape)
                    def call(self, inputs):
                        q = self.q_dense(inputs); att = self.k_dense(q)
                        if att.shape[-1] != inputs.shape[-1]:
                            if not hasattr(self, 'proj'):
                                self.proj = keras.layers.Dense(inputs.shape[-1], activation=None, use_bias=False)
                            att = self.proj(att)
                        return inputs * att
                    def get_config(self):
                        c = super().get_config(); c.update({'units': self.units}); return c
                LOCAL_MLP_MODEL = keras.models.load_model(LOCAL_MODEL_PATH, custom_objects={
                    'GraphConvLayer': GraphConvLayer,
                    'AttentionLayer': AttentionLayer
                }, compile=False)
                print(f"[LocalModel] Cargado con capas GCN tras fallback. Error previo: {load_error}")
            except Exception as e_fallback:
                print(f"[LocalModel] No se pudo cargar el modelo (intentado simple y fallback): {e_fallback}\nOriginal: {load_error}")
                LOCAL_MLP_MODEL = None
                return False
        # Determinar input dim (primera dimensión de features)
        try:
            LOCAL_MLP_INPUT_DIM = LOCAL_MLP_MODEL.input_shape[1]
        except Exception:
            # fallback: intentar inferir desde capas
            try:
                LOCAL_MLP_INPUT_DIM = LOCAL_MLP_MODEL.layers[0].input_shape[-1]
            except Exception:
                LOCAL_MLP_INPUT_DIM = None
        print(f"[LocalModel] Modelo cargado '{LOCAL_MODEL_PATH}' (input_dim={LOCAL_MLP_INPUT_DIM})")
        # Fijar orden exacto de etiquetas proporcionado por el usuario
        LOCAL_LABELS = [
            'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
            'DEL','NOTHING','SPACE'
        ]
        print(f"[LocalModel] Etiquetas hardcoded ({len(LOCAL_LABELS)})")
        # Validar output vs etiquetas
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
    # Crear contexto de manos
    try:
        import mediapipe as mp
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
    vec = np.array(coords, dtype='float32')  # 63
    if LOCAL_MLP_INPUT_DIM is None:
        return None
    if LOCAL_MLP_INPUT_DIM == vec.shape[0]:
        return vec
    if LOCAL_MLP_INPUT_DIM == 42:  # solo x,y
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
        # Map a tokens estandarizados
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
        # Suavizado simple por voto mayoritario en ventana
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

        self.build_ui()
        self._bind_space()
        self.start_camera()
        try:
            self.username_entry.focus_set()
        except Exception:
            pass

    def build_ui(self):
        tk.Label(self.root, text="GestureID", font=("Helvetica", 28, 'bold'), bg=self.root.cget('bg')).pack(pady=(12, 6))
        self.container = tk.Frame(self.root, bg=self.root.cget('bg'))
        self.container.pack(fill=tk.BOTH, expand=True, padx=20, pady=8)
        for i in range(3):
            self.container.grid_columnconfigure(i, weight=1)

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
        tk.Button(actions, text='Login', command=self.login).pack(side=tk.LEFT, padx=4)
        tk.Button(actions, text='Salir', command=self.quit).pack(side=tk.LEFT, padx=4)

        self.right_col = tk.Frame(self.container, bg=self.root.cget('bg'))
        self.right_col.grid(row=0, column=2, rowspan=2, sticky='n', padx=(12, 20), pady=10)

        self.webcam_holder = tk.Frame(self.right_col, width=self.cam_size[0], height=self.cam_size[1], bg='#000')
        self.webcam_holder.pack_propagate(False)
        self.webcam_holder.pack(side=tk.TOP)
        self.webcam_label = tk.Label(self.webcam_holder, text='NO CAM', fg='white', bg='#000', font=('Helvetica', 14))
        self.webcam_label.pack(fill=tk.BOTH, expand=True)

        self.preview_holder = tk.Frame(self.right_col, width=self.cam_size[0], height=self.cam_size[1], bg='#222')
        self.preview_holder.pack_propagate(False)
        self.preview_label = tk.Label(self.preview_holder, text='Sin captura', fg='white', bg='#222')
        self.preview_label.place(relx=0.5, rely=0.5, anchor='center')

        self.dev_toggle_btn = tk.Button(self.root, text='DEV Mode', width=12, command=self.toggle_dev_mode)
        self.dev_toggle_btn.place(relx=0.01, rely=0.98, anchor='sw')

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

        self._style_mode_buttons()
        self._refresh_mode_buttons()

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
        else:
            self.pass_btn.config(**active)
            self.gesture_btn.config(**idle)
            self.password_entry.configure(state='normal')
            self.pass_label.config(text='Contraseña:')

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
        if self.mode != 'GESTOS' or self.frame is None:
            return
        letterboxed = letterbox_to_512_from_bgr(self.frame)
        png_bytes = pil_to_png_bytes(letterboxed)
        self.last_pred_var.set('...')
        if self.dev_mode:
            # Si es modelo local, intentar dibujar landmarks
            if self.selected_model == 'local':
                pil_with_lm = self._make_landmark_preview(self.frame) or Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
                self._update_preview(pil_with_lm)
            else:
                # Usar imagen RAW para preview (sin letterbox) y rellenar panel completo
                try:
                    raw_pil = Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
                except Exception:
                    raw_pil = letterboxed
                self._update_preview(raw_pil)
        threading.Thread(target=self._infer_thread, args=(png_bytes,), daemon=True).start()

    def _infer_thread(self, png_bytes: bytes):
        key = self.selected_model
        if key == 'local':
            # Usar frame original para landmarks
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
            import mediapipe as mp  # import local para no fallar si falta
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = MP_HANDS_CTX.process(rgb)
            if not res.multi_hand_landmarks:
                return Image.fromarray(rgb)
            annotated = rgb.copy()
            # Intentar usar drawing_utils si existe
            try:
                from mediapipe.python.solutions import drawing_utils as du
                for hand_landmarks in res.multi_hand_landmarks:
                    du.draw_landmarks(
                        annotated,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=du.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                        connection_drawing_spec=du.DrawingSpec(color=(255,140,0), thickness=2, circle_radius=2)
                    )
            except Exception:
                # Fallback simple
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
        pwd = self.password_var.get()
        if not user:
            messagebox.showerror('Login', 'Introduce un usuario'); return
        if not pwd:
            messagebox.showwarning('Login', 'No hay contraseña'); return
        origen = 'gestos' if self.mode == 'GESTOS' else 'texto'
        stars = '*' * len(pwd)
        messagebox.showinfo('Login', f'Usuario: {user}\nPassword: {stars}\n(Método: {origen} – simulación)')

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
