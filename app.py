"""
Virtual Paint — Air Canvas (Streamlit Cloud Edition)
====================================================
Versao web do Virtual Paint usando streamlit-webrtc.
A camera vem do navegador do usuario (WebRTC), nao do servidor.

Deploy:
  1. Suba este repo no GitHub (com requirements.txt e packages.txt)
  2. Em https://share.streamlit.io conecte o repo e aponte para app.py
  3. Acesse o link publico (HTTPS) — qualquer pessoa pode usar

Stack: Streamlit + streamlit-webrtc + OpenCV + MediaPipe Tasks (HandLandmarker)
"""

import os
import time
import threading
import urllib.request
from datetime import datetime

import av
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration


# =====================================================================
# Configuracao da pagina
# =====================================================================

st.set_page_config(
    page_title="Virtual Paint — Air Canvas",
    page_icon="🎨",
    layout="wide",
)


# =====================================================================
# Constantes visuais (paleta + dimensoes da toolbar)
# Valores ligeiramente menores que no original porque o frame WebRTC
# costuma chegar em 640x480 (nao 1280x720) na maioria dos navegadores.
# =====================================================================

TOOLS = [
    {"name": "Vermelho", "type": "color",  "color": (60, 76, 231)},
    {"name": "Laranja",  "type": "color",  "color": (0, 140, 255)},
    {"name": "Amarelo",  "type": "color",  "color": (0, 220, 240)},
    {"name": "Verde",    "type": "color",  "color": (113, 204, 46)},
    {"name": "Ciano",    "type": "color",  "color": (242, 195, 0)},
    {"name": "Azul",     "type": "color",  "color": (242, 99, 39)},
    {"name": "Roxo",     "type": "color",  "color": (175, 82, 156)},
    {"name": "Branco",   "type": "color",  "color": (245, 245, 245)},
    {"name": "Borracha", "type": "eraser", "color": (45, 45, 50)},
    {"name": "Limpar",   "type": "clear",  "color": (70, 70, 80)},
    {"name": "Salvar",   "type": "save",   "color": (80, 175, 80)},
]

TOOLBAR_HEIGHT = 72
BUTTON_SIZE = 48
BUTTON_PADDING = 10
TOOLBAR_BG = (28, 28, 34)
TOOLBAR_BORDER = (55, 55, 65)
TEXT_COLOR = (235, 235, 240)
SELECTED_BORDER = (255, 255, 255)
HOVER_ACCENT = (0, 255, 180)

BRUSH_THICKNESS = 7
ERASER_THICKNESS = 50
SELECTION_HOVER_FRAMES = 6  # frames sobre o botao para "clicar"


# =====================================================================
# Modelo MediaPipe — baixado no startup do app
# =====================================================================

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = "hand_landmarker.task"


def ensure_model():
    """Baixa o modelo apenas uma vez por sessao do servidor."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Baixando modelo MediaPipe HandLandmarker (~5 MB)..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


# =====================================================================
# Detector de maos (MediaPipe Tasks API, modo VIDEO)
# =====================================================================

class HandDetector:
    TIP_IDS = [4, 8, 12, 16, 20]

    def __init__(self, max_hands=1, det_conf=0.6, track_conf=0.5):
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=det_conf,
            min_hand_presence_confidence=det_conf,
            min_tracking_confidence=track_conf,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._t0 = time.time()

    def find_hand(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.time() - self._t0) * 1000)
        result = self.landmarker.detect_for_video(mp_image, ts_ms)

        if not result.hand_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        return [(int(p.x * w), int(p.y * h)) for p in result.hand_landmarks[0]]

    def fingers_up(self, points):
        if not points:
            return [0, 0, 0, 0, 0]
        fingers = []
        fingers.append(1 if points[self.TIP_IDS[0]][0] > points[self.TIP_IDS[0] - 1][0] else 0)
        for i in range(1, 5):
            tip_y = points[self.TIP_IDS[i]][1]
            pip_y = points[self.TIP_IDS[i] - 2][1]
            fingers.append(1 if tip_y < pip_y else 0)
        return fingers


# =====================================================================
# UI helpers — retangulos arredondados e toolbar
# =====================================================================

def rounded_rect(img, pt1, pt2, color, thickness=-1, radius=8):
    x1, y1 = pt1
    x2, y2 = pt2
    if thickness < 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for cx, cy in [(x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                       (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)]:
            cv2.circle(img, (cx, cy), radius, color, -1)
    else:
        cv2.rectangle(img, pt1, pt2, color, thickness)


class Toolbar:
    def __init__(self, width):
        self.width = width
        self.height = TOOLBAR_HEIGHT
        self.buttons = self._layout_buttons()

    def _layout_buttons(self):
        n = len(TOOLS)
        pad = BUTTON_PADDING
        total_w = n * BUTTON_SIZE + (n - 1) * pad
        # Se nao couber, reduz padding
        if total_w > self.width - 16:
            available = self.width - 16 - n * BUTTON_SIZE
            pad = max(2, available // max(1, n - 1))
            total_w = n * BUTTON_SIZE + (n - 1) * pad
        start_x = (self.width - total_w) // 2
        y = (self.height - BUTTON_SIZE) // 2
        buttons = []
        for i, tool in enumerate(TOOLS):
            x = start_x + i * (BUTTON_SIZE + pad)
            buttons.append({"rect": (x, y, x + BUTTON_SIZE, y + BUTTON_SIZE), "tool": tool})
        return buttons

    def draw(self, frame, selected_index, hover_index, hover_progress):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), TOOLBAR_BG, -1)
        cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)
        cv2.line(frame, (0, self.height), (self.width, self.height), TOOLBAR_BORDER, 2)

        for i, btn in enumerate(self.buttons):
            x1, y1, x2, y2 = btn["rect"]
            tool = btn["tool"]
            rounded_rect(frame, (x1, y1), (x2, y2), tool["color"], -1, radius=8)

            label = {"eraser": "BOR", "clear": "LIM", "save": "SAL"}.get(tool["type"])
            if label:
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                tx = x1 + (BUTTON_SIZE - tw) // 2
                ty = y1 + (BUTTON_SIZE + th) // 2
                cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, TEXT_COLOR, 1, cv2.LINE_AA)

            if i == selected_index:
                cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3),
                              SELECTED_BORDER, 2)

            if i == hover_index and hover_progress > 0:
                progress = min(1.0, hover_progress / SELECTION_HOVER_FRAMES)
                bar_x2 = int(x1 + (x2 - x1) * progress)
                cv2.rectangle(frame, (x1, y2 + 3), (bar_x2, y2 + 7), HOVER_ACCENT, -1)

    def get_button_at(self, x, y):
        for i, btn in enumerate(self.buttons):
            x1, y1, x2, y2 = btn["rect"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                return i
        return -1


# =====================================================================
# Video processor — instanciado pelo streamlit-webrtc, recv() roda
# para cada frame que chega do navegador
# =====================================================================

class VirtualPaintProcessor:
    def __init__(self):
        # Modelo ja foi baixado pelo main thread antes do streamer iniciar
        self.detector = HandDetector(max_hands=1)
        self.lock = threading.Lock()

        # Inicializados no primeiro frame (precisamos saber w/h reais)
        self.toolbar = None
        self.canvas = None
        self.last_size = None

        # Estado de desenho
        self.selected_index = 0
        self.hover_index = -1
        self.hover_progress = 0
        self.prev_point = None

        # HUD
        self.flash_msg = ""
        self.flash_time = 0
        self.fps = 0.0
        self._last_t = time.time()

        # Comandos vindos do sidebar (thread-safe via lock)
        self.cmd_clear = False
        self.cmd_select_index = None

    # ----------------- API para o sidebar -----------------

    def request_clear(self):
        with self.lock:
            self.cmd_clear = True

    def request_select(self, index):
        with self.lock:
            self.cmd_select_index = index

    def get_canvas_snapshot(self):
        """Retorna copia do canvas atual (para o botao de download)."""
        with self.lock:
            if self.canvas is None:
                return None
            return self.canvas.copy()

    # ----------------- ferramentas -----------------

    def _apply_tool(self, idx):
        tool = TOOLS[idx]
        if tool["type"] == "clear":
            self.canvas[:] = 0
            self._flash("Canvas limpo")
        elif tool["type"] == "save":
            # No servidor nao ha disco persistente — instruimos o usuario
            # a usar o botao de download na barra lateral.
            self._flash("Use 'Baixar PNG' na barra lateral")
        else:
            self.selected_index = idx

    def _flash(self, text):
        self.flash_msg = text
        self.flash_time = time.time()

    # ----------------- loop de frame -----------------

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # espelho natural
        h, w = img.shape[:2]

        with self.lock:
            # (Re)inicializa canvas e toolbar se o tamanho mudou
            if self.canvas is None or self.last_size != (h, w):
                self.canvas = np.zeros((h, w, 3), dtype=np.uint8)
                self.toolbar = Toolbar(w)
                self.last_size = (h, w)

            # Processa comandos do sidebar
            if self.cmd_clear:
                self.canvas[:] = 0
                self.cmd_clear = False
                self._flash("Canvas limpo")
            if self.cmd_select_index is not None:
                self._apply_tool(self.cmd_select_index)
                self.cmd_select_index = None

            # Deteccao de mao
            points = self.detector.find_hand(img)
            cursor, mode = None, "idle"

            if points:
                fingers = self.detector.fingers_up(points)
                index_tip = points[8]
                middle_tip = points[12]
                cursor = index_tip

                # Modo selecao: indicador + medio
                if fingers[1] and fingers[2]:
                    mode = "select"
                    self.prev_point = None
                    mid = ((index_tip[0] + middle_tip[0]) // 2,
                           (index_tip[1] + middle_tip[1]) // 2)

                    if mid[1] < TOOLBAR_HEIGHT:
                        hover = self.toolbar.get_button_at(*mid)
                        if hover == self.hover_index and hover != -1:
                            self.hover_progress += 1
                            if self.hover_progress >= SELECTION_HOVER_FRAMES:
                                self._apply_tool(hover)
                                self.hover_progress = 0
                                self.hover_index = -1
                        else:
                            self.hover_index = hover
                            self.hover_progress = 1 if hover != -1 else 0
                    else:
                        self.hover_index = -1
                        self.hover_progress = 0

                # Modo desenho: apenas indicador
                elif fingers[1] and not fingers[2]:
                    mode = "draw"
                    self.hover_index = -1
                    self.hover_progress = 0

                    if index_tip[1] > TOOLBAR_HEIGHT:
                        tool = TOOLS[self.selected_index]
                        if tool["type"] == "eraser":
                            color, thick = (0, 0, 0), ERASER_THICKNESS
                        else:
                            color, thick = tool["color"], BRUSH_THICKNESS
                        if self.prev_point is not None:
                            cv2.line(self.canvas, self.prev_point, index_tip,
                                     color, thick, cv2.LINE_AA)
                        self.prev_point = index_tip
                    else:
                        self.prev_point = None
                else:
                    self.prev_point = None
                    self.hover_index = -1
                    self.hover_progress = 0
            else:
                self.prev_point = None
                self.hover_index = -1

            # Compoe canvas sobre o frame (mascara binaria)
            gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            bg = cv2.bitwise_and(img, img, mask=mask_inv)
            fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
            img = cv2.add(bg, fg)

            # Toolbar
            self.toolbar.draw(img, self.selected_index, self.hover_index, self.hover_progress)

            # Cursor
            if cursor is not None:
                cx, cy = cursor
                if mode == "draw":
                    tool = TOOLS[self.selected_index]
                    if tool["type"] == "eraser":
                        cv2.circle(img, (cx, cy), ERASER_THICKNESS // 2, (200, 200, 200), 2)
                    else:
                        cv2.circle(img, (cx, cy), BRUSH_THICKNESS, tool["color"], -1)
                        cv2.circle(img, (cx, cy), BRUSH_THICKNESS + 2, (255, 255, 255), 2)
                elif mode == "select":
                    cv2.circle(img, (cx, cy), 11, HOVER_ACCENT, 2)
                    cv2.circle(img, (cx, cy), 3, HOVER_ACCENT, -1)

            # FPS suavizado
            now = time.time()
            dt = max(now - self._last_t, 1e-6)
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
            self._last_t = now

            # Status bar inferior
            bar_h = 30
            overlay = img.copy()
            cv2.rectangle(overlay, (0, h - bar_h), (w, h), (20, 20, 25), -1)
            cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
            tool = TOOLS[self.selected_index]
            label = {"draw": "DESENHANDO", "select": "SELECIONANDO", "idle": "PAUSADO"}[mode]
            status = f"{label}  |  {tool['name']}  |  {self.fps:.0f} FPS"
            cv2.putText(img, status, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, TEXT_COLOR, 1, cv2.LINE_AA)

            if self.flash_msg and time.time() - self.flash_time < 3:
                (tw, _), _ = cv2.getTextSize(self.flash_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                x = w - tw - 12
                cv2.putText(img, self.flash_msg, (x, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, HOVER_ACCENT, 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# =====================================================================
# UI Streamlit
# =====================================================================

st.title("🎨 Virtual Paint — Air Canvas")
st.caption(
    "Desenhe no ar usando o dedo indicador. "
    "Levante indicador + medio para selecionar ferramentas na barra superior."
)

with st.expander("Como usar", expanded=False):
    st.markdown(
        """
        **Gestos**
        - ☝️ **Indicador levantado** → desenhar
        - ✌️ **Indicador + medio levantados** → modo selecao (paire sobre os botoes ~0,5 s)
        - ✋ Outros gestos → pausar

        **Ferramentas (barra superior, da esquerda para direita)**
        8 cores → Borracha (BOR) → Limpar (LIM) → Salvar (SAL).

        **Dicas**
        - Use **Chrome ou Edge** (melhor compatibilidade WebRTC).
        - Permita o acesso a camera quando o navegador pedir.
        - Iluminacao boa e mao a ~50 cm da camera ajudam muito.
        - Em mobile, segure o aparelho na horizontal para ter mais area.
        - Se o FPS cair muito, feche outras abas pesadas.
        """
    )

# Garante que o modelo esta baixado antes de iniciar o streamer
ensure_model()

# STUN publico do Google — ajuda na travessia de NAT em redes domesticas
RTC_CONFIG = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

ctx = webrtc_streamer(
    key="virtual-paint",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=VirtualPaintProcessor,
    media_stream_constraints={
        "video": {"width": {"ideal": 960}, "height": {"ideal": 540}},
        "audio": False,
    },
    async_processing=True,
)

# Sidebar: controles redundantes (caso o gesto nao funcione) + download
st.sidebar.header("Controles")

if ctx.video_processor:
    color_tools = [(i, t) for i, t in enumerate(TOOLS) if t["type"] == "color"]
    names = [t["name"] for _, t in color_tools]
    chosen = st.sidebar.selectbox("Cor do pincel", names, index=0)
    if st.sidebar.button("Aplicar cor"):
        idx = next(i for i, t in color_tools if t["name"] == chosen)
        ctx.video_processor.request_select(idx)
        st.sidebar.success(f"Cor: {chosen}")

    st.sidebar.divider()

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🩹 Borracha"):
            idx = next(i for i, t in enumerate(TOOLS) if t["type"] == "eraser")
            ctx.video_processor.request_select(idx)
    with col2:
        if st.button("🗑️ Limpar"):
            ctx.video_processor.request_clear()

    st.sidebar.divider()
    st.sidebar.subheader("Salvar arte")

    if st.sidebar.button("📸 Capturar canvas"):
        snap = ctx.video_processor.get_canvas_snapshot()
        if snap is not None and snap.any():
            ok, buf = cv2.imencode(".png", snap)
            if ok:
                st.session_state["last_snapshot"] = buf.tobytes()
                st.sidebar.success("Pronto para download abaixo.")
        else:
            st.sidebar.warning("Canvas vazio — desenhe algo primeiro.")

    if "last_snapshot" in st.session_state:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.sidebar.download_button(
            "⬇️ Baixar PNG",
            data=st.session_state["last_snapshot"],
            file_name=f"virtual_paint_{ts}.png",
            mime="image/png",
        )
else:
    st.sidebar.info("Clique em **START** acima para iniciar a camera.")

st.markdown(
    "<hr><div style='text-align:center;color:#888;font-size:0.85em;'>"
    "Streamlit + streamlit-webrtc + MediaPipe HandLandmarker"
    "</div>",
    unsafe_allow_html=True,
)
