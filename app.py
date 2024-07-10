# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import cv2
# import mediapipe as mp
# import numpy as np
# import base64
# from io import BytesIO
# from PIL import Image
# import uuid
#
# app = Flask(__name__)
# CORS(app)
#
# # Initialize Mediapipe Face Mesh and Hands
# mp_face_mesh = mp.solutions.face_mesh
# mp_hands = mp.solutions.hands
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
# hands = mp_hands.Hands()
#
# # Dictionaries to store jewelry images
# necklace_images = {}
# ring_images = {}
# earring_images = {}
# bracelet_images = {}
#
# @app.route("/")
# def home():
#     return "Flask server is running."
#
# @app.route("/upload_necklace", methods=["POST"])
# def upload_necklace():
#     file = request.files.get("necklace")
#     if not file:
#         return jsonify({"error": "Necklace image not provided"}), 400
#
#     image = Image.open(file.stream)
#     necklace_id = str(uuid.uuid4())
#     necklace_images[necklace_id] = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
#
#     return jsonify({"necklace_id": necklace_id})
#
# @app.route("/upload_ring", methods=["POST"])
# def upload_ring():
#     file = request.files.get("ring")
#     if not file:
#         return jsonify({"error": "Ring image not provided"}), 400
#
#     image = Image.open(file.stream)
#     ring_id = str(uuid.uuid4())
#     ring_images[ring_id] = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
#
#     return jsonify({"ring_id": ring_id})
#
# @app.route("/upload_earring", methods=["POST"])
# def upload_earring():
#     file = request.files.get("earring")
#     if not file:
#         return jsonify({"error": "Earring image not provided"}), 400
#
#     image = Image.open(file.stream)
#     earring_id = str(uuid.uuid4())
#     earring_images[earring_id] = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
#
#     return jsonify({"earring_id": earring_id})
#
# @app.route("/upload_bracelet", methods=["POST"])
# def upload_bracelet():
#     file = request.files.get("bracelet")
#     if not file:
#         return jsonify({"error": "Bracelet image not provided"}), 400
#
#     image = Image.open(file.stream)
#     bracelet_id = str(uuid.uuid4())
#     bracelet_images[bracelet_id] = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
#
#     return jsonify({"bracelet_id": bracelet_id})
#
# @app.route("/process_frame", methods=["POST"])
# def process_frame():
#     data = request.json
#     if not data or 'frame' not in data:
#         return jsonify({"error": "Frame not provided"}), 400
#
#     frame_data = data["frame"].split(",")[1]
#     necklace_id = data.get("necklace_id")
#     ring_id = data.get("ring_id")
#     earring_id = data.get("earring_id")
#     bracelet_id = data.get("bracelet_id")
#
#     frame = Image.open(BytesIO(base64.b64decode(frame_data)))
#     frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
#
#     if necklace_id and necklace_id in necklace_images:
#         frame = overlay_necklace(frame, necklace_images[necklace_id])
#     if ring_id and ring_id in ring_images:
#         frame = overlay_ring_on_hand(frame, ring_images[ring_id])
#     if earring_id and earring_id in earring_images:
#         frame = overlay_earring(frame, earring_images[earring_id])
#     if bracelet_id and bracelet_id in bracelet_images:
#         frame = add_bracelet_overlay(frame, bracelet_images[bracelet_id])
#
#     _, buffer = cv2.imencode(".jpg", frame)
#     jpg_as_text = base64.b64encode(buffer).decode("utf-8")
#
#     return jsonify({"image": jpg_as_text})
#
# def overlay_necklace(frame, necklace_img):
#     # Offset for positioning the necklace below the chin
#     offset_y = 30
#
#     # Process the frame to find face landmarks
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb_frame)
#     frame1 = frame.copy()
#
#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             landmark_points = face_landmarks.landmark
#
#             # Use landmarks 152 (lower chin) and 9 (neck area)
#             x_chin, y_chin = int(landmark_points[152].x * frame.shape[1]), int(landmark_points[152].y * frame.shape[0])
#             x_neck, y_neck = int(landmark_points[9].x * frame.shape[1]), int(landmark_points[9].y * frame.shape[0])
#             y_chin += offset_y  # Add an offset to position the necklace slightly below the chin
#
#             # Calculate the distance between the landmarks
#             distance = np.sqrt((x_neck - x_chin) ** 2 + (y_neck - y_chin) ** 2)
#
#             # Resize the necklace image based on the distance
#             scale = distance / necklace_img.shape[1]
#             new_w = int(necklace_img.shape[1] * scale)
#             new_h = int(necklace_img.shape[0] * scale)
#             overlay = cv2.resize(necklace_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
#
#             # Calculate position for overlay
#             x_center = x_chin
#             y_center = y_chin + int(distance / 3)  # Position slightly above the chest
#             h, w = frame.shape[:2]
#             x_start = max(0, x_center - new_w // 2)
#             y_start = max(0, y_center - new_h // 2)
#             x_end = min(w, x_start + overlay.shape[1])
#             y_end = min(h, y_start + overlay.shape[0])
#
#             overlay_resized = overlay[:(y_end - y_start), :(x_end - x_start)]
#
#             alpha_s = overlay_resized[:, :, 3] / 255.0
#             alpha_l = 1.0 - alpha_s
#
#             # Apply overlay to the frame
#             for c in range(0, 3):
#                 frame1[y_start:y_end, x_start:x_end, c] = (alpha_s * overlay_resized[:, :, c] +
#                                                            alpha_l * frame[y_start:y_end, x_start:x_end, c])
#     return frame1
#
# def overlay_ring_on_hand(frame, ring_img):
#     frame1 = frame.copy()
#     # Convert the image to RGB
#     image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # Process using Mediapipe
#     results = hands.process(image_rgb)
#
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Overlay ring image between landmarks 13 and 14 (assuming landmark indices)
#             if len(hand_landmarks.landmark) >= 14:  # Ensure there are enough landmarks
#                 # Get the coordinates of landmarks 13 and 14
#                 x1 = int(hand_landmarks.landmark[13].x * frame.shape[1])
#                 y1 = int(hand_landmarks.landmark[13].y * frame.shape[0])
#                 x2 = int(hand_landmarks.landmark[14].x * frame.shape[1])
#                 y2 = int(hand_landmarks.landmark[14].y * frame.shape[0])
#
#                 # Calculate midpoint between landmarks 13 and 14
#                 x_mid = (x1 + x2) // 2
#                 y_mid = (y1 + y2) // 2
#                 reference_size = 50  # Example size in pixels
#                 # Calculate size of the ring image based on a reference size
#                 size_factor = reference_size / ring_img.shape[1]  # Assuming ring_img is square
#
#                 # Resize ring image
#                 ring_resized = cv2.resize(ring_img, None, fx=size_factor, fy=size_factor)
#
#                 # Calculate position to place the ring image centered on the midpoint
#                 x_offset = x_mid - ring_resized.shape[1] // 2
#                 y_offset = y_mid - ring_resized.shape[0] // 2
#
#                 # Ensure overlay and ring_resized have the same shape
#                 overlay = frame1[y_offset:y_offset + ring_resized.shape[0], x_offset:x_offset + ring_resized.shape[1]]
#                 if overlay.shape[0] == ring_resized.shape[0] and overlay.shape[1] == ring_resized.shape[1]:
#                     # Overlay the resized ring image onto the frame
#                     alpha_s = ring_resized[:, :, 3] / 255.0
#                     alpha_l = 1.0 - alpha_s
#                     for c in range(0, 3):
#                         overlay[:, :, c] = (alpha_s * ring_resized[:, :, c] +
#                                             alpha_l * overlay[:, :, c])
#
#                     # Update the frame with the overlay
#                     frame1[y_offset:y_offset + ring_resized.shape[0],
#                     x_offset:x_offset + ring_resized.shape[1]] = overlay
#
#     return frame1
#
# def overlay_earring(frame, earring_img):
#     # Process the frame to find face landmarks
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb_frame)
#     frame1 = frame.copy()
#
#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             landmark_points = face_landmarks.landmark
#
#             # Get z-coordinates to determine visibility
#             z_ear_left = landmark_points[234].z
#             z_ear_right = landmark_points[454].z
#
#             # Apply earring to left ear if visible
#             if z_ear_left <= z_ear_right + 0.1:
#                 x_ear_left = int(landmark_points[234].x * frame.shape[1])
#                 y_ear_left = int(landmark_points[234].y * frame.shape[0])
#                 apply_earring(frame1, earring_img, x_ear_left, y_ear_left, is_left=True)
#
#             # Apply earring to right ear if visible
#             if z_ear_right <= z_ear_left + 0.1:
#                 x_ear_right = int(landmark_points[454].x * frame.shape[1])
#                 y_ear_right = int(landmark_points[454].y * frame.shape[0])
#                 apply_earring(frame1, earring_img, x_ear_right, y_ear_right, is_left=False)
#
#     return frame1
#
# def apply_earring(frame, earring_img, x_ear, y_ear, is_left):
#     # Calculate the distance to determine the size of the earring
#     distance = frame.shape[1] * 0.15  # Adjust the size based on your image and preference
#
#     # Resize the earring image based on the distance
#     scale = distance / earring_img.shape[1]
#     new_w = int(earring_img.shape[1] * scale)
#     new_h = int(earring_img.shape[0] * scale)
#
#     # Flip horizontally for left ear
#     if is_left:
#         overlay = cv2.flip(earring_img, 1)
#     else:
#         overlay = earring_img.copy()
#
#     overlay = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)
#
#     # Calculate position for overlay
#     x_center = x_ear
#     y_center = y_ear + int(distance * 0.5)  # Position slightly below the ear landmark
#     h, w = frame.shape[:2]
#     x_start = max(0, x_center - new_w // 2)
#     y_start = max(0, y_center - new_h // 2)
#     x_end = min(w, x_start + new_w)
#     y_end = min(h, y_start + new_h)
#
#     overlay_resized = overlay[:(y_end - y_start), :(x_end - x_start)]
#
#     alpha_s = overlay_resized[:, :, 3] / 255.0
#     alpha_l = 1.0 - alpha_s
#
#     # Apply overlay to the frame
#     for c in range(0, 3):
#         frame[y_start:y_end, x_start:x_end, c] = (alpha_s * overlay_resized[:, :, c] +
#                                                   alpha_l * frame[y_start:y_end, x_start:x_end, c])
#     print(f"Earring applied at: x_start: {x_start}, y_start: {y_start}, x_end: {x_end}, y_end: {y_end}")
#
# def add_bracelet_overlay(frame, bracelet_img):
#     frame1 = frame.copy()
#     # Convert the image to RGB
#     image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # Process using Mediapipe
#     results = hands.process(image_rgb)
#
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Overlay bracelet image near the wrist area (landmark 0)
#             if len(hand_landmarks.landmark) >= 2:  # Ensure there are enough landmarks
#                 # Get the coordinates of landmark 0 (wrist coordinates)
#                 x1 = int(hand_landmarks.landmark[0].x * frame.shape[1])
#                 y1 = int(hand_landmarks.landmark[0].y * frame.shape[0])
#
#                 # Calculate size of the bracelet image based on a reference size
#                 reference_size = 100  # Example size in pixels
#                 scaling_factor = 1.2  # Scale bracelet size by 20%
#                 size_factor = (reference_size / bracelet_img.shape[1]) * scaling_factor
#
#                 # Resize bracelet image
#                 bracelet_resized = cv2.resize(bracelet_img, None, fx=size_factor, fy=size_factor)
#
#                 # Adjust offsets to move the bracelet closer to landmark 0
#                 x_offset = x1 - bracelet_resized.shape[1] // 2
#                 y_offset = y1 - bracelet_resized.shape[0] // 2 + 30  # Move it 30 pixels below the landmark 0
#
#                 # Ensure overlay and bracelet_resized have the same shape
#                 if 0 <= y_offset < frame.shape[0] - bracelet_resized.shape[0] and 0 <= x_offset < frame.shape[1] - bracelet_resized.shape[1]:
#                     overlay = frame[y_offset:y_offset + bracelet_resized.shape[0], x_offset:x_offset + bracelet_resized.shape[1]]
#                     if overlay.shape[0] == bracelet_resized.shape[0] and overlay.shape[1] == bracelet_resized.shape[1]:
#                         # Overlay the resized bracelet image onto the frame
#                         alpha_s = bracelet_resized[:, :, 3] / 255.0
#                         alpha_l = 1.0 - alpha_s
#                         for c in range(0, 3):
#                             overlay[:, :, c] = (alpha_s * bracelet_resized[:, :, c] +
#                                                 alpha_l * overlay[:, :, c])
#
#                         # Update the frame with the overlay
#                         frame1[y_offset:y_offset + bracelet_resized.shape[0], x_offset:x_offset + bracelet_resized.shape[1]] = overlay
#     frame1 = cv2.flip(frame1, 1)
#     return frame1
#
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)



from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import uuid
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG)

# Initialize Mediapipe Face Mesh and Hands
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
hands = mp_hands.Hands()

# Dictionaries to store jewelry images
necklace_images = {}
ring_images = {}
earring_images = {}
bracelet_images = {}

@app.route("/")
def home():
    return "Flask server is running."

@app.route("/upload_necklace", methods=["POST"])
def upload_necklace():
    file = request.files.get("necklace")
    if not file:
        return jsonify({"error": "Necklace image not provided"}), 400

    image = Image.open(file.stream)
    necklace_id = str(uuid.uuid4())
    necklace_images[necklace_id] = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
    logging.debug(f"Uploaded necklace image with ID: {necklace_id}, Shape: {necklace_images[necklace_id].shape}")
    return jsonify({"necklace_id": necklace_id})

@app.route("/upload_ring", methods=["POST"])
def upload_ring():
    file = request.files.get("ring")
    if not file:
        return jsonify({"error": "Ring image not provided"}), 400

    image = Image.open(file.stream)
    ring_id = str(uuid.uuid4())
    ring_images[ring_id] = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
    logging.debug(f"Uploaded ring image with ID: {ring_id}, Shape: {ring_images[ring_id].shape}")
    return jsonify({"ring_id": ring_id})

@app.route("/upload_earring", methods=["POST"])
def upload_earring():
    file = request.files.get("earring")
    if not file:
        return jsonify({"error": "Earring image not provided"}), 400

    image = Image.open(file.stream)
    earring_id = str(uuid.uuid4())
    earring_images[earring_id] = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
    logging.debug(f"Uploaded earring image with ID: {earring_id}, Shape: {earring_images[earring_id].shape}")
    return jsonify({"earring_id": earring_id})

@app.route("/upload_bracelet", methods=["POST"])
def upload_bracelet():
    file = request.files.get("bracelet")
    if not file:
        return jsonify({"error": "Bracelet image not provided"}), 400

    image = Image.open(file.stream)
    bracelet_id = str(uuid.uuid4())
    bracelet_images[bracelet_id] = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
    logging.debug(f"Uploaded bracelet image with ID: {bracelet_id}, Shape: {bracelet_images[bracelet_id].shape}")
    return jsonify({"bracelet_id": bracelet_id})

@app.route("/process_frame", methods=["POST"])
def process_frame():
    data = request.json
    if not data or 'frame' not in data:
        return jsonify({"error": "Frame not provided"}), 400

    frame_data = data["frame"].split(",")[1]
    necklace_id = data.get("necklace_id")
    ring_id = data.get("ring_id")
    earring_id = data.get("earring_id")
    bracelet_id = data.get("bracelet_id")

    frame = Image.open(BytesIO(base64.b64decode(frame_data)))
    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

    logging.debug("Processing frame")

    if necklace_id and necklace_id in necklace_images:
        logging.debug(f"Overlaying necklace with ID: {necklace_id}")
        frame = overlay_necklace(frame, necklace_images[necklace_id])
    if ring_id and ring_id in ring_images:
        logging.debug(f"Overlaying ring with ID: {ring_id}")
        frame = overlay_ring_on_hand(frame, ring_images[ring_id])
    if earring_id and earring_id in earring_images:
        logging.debug(f"Overlaying earring with ID: {earring_id}")
        frame = overlay_earring(frame, earring_images[earring_id])
    if bracelet_id and bracelet_id in bracelet_images:
        logging.debug(f"Overlaying bracelet with ID: {bracelet_id}")
        frame = add_bracelet_overlay(frame, bracelet_images[bracelet_id])

    _, buffer = cv2.imencode(".jpg", frame)
    jpg_as_text = base64.b64encode(buffer).decode("utf-8")

    return jsonify({"image": jpg_as_text})

def overlay_necklace(frame, necklace_img):
    offset_y = 30
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    frame1 = frame.copy()

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmark_points = face_landmarks.landmark
            x_chin, y_chin = int(landmark_points[152].x * frame.shape[1]), int(landmark_points[152].y * frame.shape[0])
            x_neck, y_neck = int(landmark_points[9].x * frame.shape[1]), int(landmark_points[9].y * frame.shape[0])
            y_chin += offset_y

            distance = np.sqrt((x_neck - x_chin) ** 2 + (y_neck - y_chin) ** 2)
            scale = distance / necklace_img.shape[1]
            new_w = int(necklace_img.shape[1] * scale)
            new_h = int(necklace_img.shape[0] * scale)
            overlay = cv2.resize(necklace_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            x_center = x_chin
            y_center = y_chin + int(distance / 3)
            h, w = frame.shape[:2]
            x_start = max(0, x_center - new_w // 2)
            y_start = max(0, y_center - new_h // 2)
            x_end = min(w, x_start + overlay.shape[1])
            y_end = min(h, y_start + overlay.shape[0])

            overlay_resized = overlay[:(y_end - y_start), :(x_end - x_start)]

            alpha_s = overlay_resized[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(0, 3):
                frame1[y_start:y_end, x_start:x_end, c] = (alpha_s * overlay_resized[:, :, c] +
                                                           alpha_l * frame[y_start:y_end, x_start:x_end, c])
            logging.debug(f"Necklace applied at: x_start: {x_start}, y_start: {y_start}, x_end: {x_end}, y_end: {y_end}")
    return frame1

def overlay_ring_on_hand(frame, ring_img):
    frame1 = frame.copy()
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if len(hand_landmarks.landmark) >= 14:
                x1 = int(hand_landmarks.landmark[13].x * frame.shape[1])
                y1 = int(hand_landmarks.landmark[13].y * frame.shape[0])
                x2 = int(hand_landmarks.landmark[14].x * frame.shape[1])
                y2 = int(hand_landmarks.landmark[14].y * frame.shape[0])

                x_mid = (x1 + x2) // 2
                y_mid = (y1 + y2) // 2
                reference_size = 50
                size_factor = reference_size / ring_img.shape[1]
                ring_resized = cv2.resize(ring_img, None, fx=size_factor, fy=size_factor)

                x_offset = x_mid - ring_resized.shape[1] // 2
                y_offset = y_mid - ring_resized.shape[0] // 2
                overlay = frame1[y_offset:y_offset + ring_resized.shape[0], x_offset:x_offset + ring_resized.shape[1]]
                if overlay.shape[0] == ring_resized.shape[0] and overlay.shape[1] == ring_resized.shape[1]:
                    alpha_s = ring_resized[:, :, 3] / 255.0
                    alpha_l = 1.0 - alpha_s
                    for c in range(0, 3):
                        overlay[:, :, c] = (alpha_s * ring_resized[:, :, c] + alpha_l * overlay[:, :, c])
                    frame1[y_offset:y_offset + ring_resized.shape[0], x_offset:x_offset + ring_resized.shape[1]] = overlay
                logging.debug(f"Ring applied at: x_start: {x_offset}, y_start: {y_offset}, x_end: {x_offset + ring_resized.shape[1]}, y_end: {y_offset + ring_resized.shape[0]}")
    return frame1

def overlay_earring(frame, earring_img):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    frame1 = frame.copy()

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmark_points = face_landmarks.landmark
            z_ear_left = landmark_points[234].z
            z_ear_right = landmark_points[454].z

            if z_ear_left <= z_ear_right + 0.1:
                x_ear_left = int(landmark_points[234].x * frame.shape[1])
                y_ear_left = int(landmark_points[234].y * frame.shape[0])
                apply_earring(frame1, earring_img, x_ear_left, y_ear_left, is_left=True)

            if z_ear_right <= z_ear_left + 0.1:
                x_ear_right = int(landmark_points[454].x * frame.shape[1])
                y_ear_right = int(landmark_points[454].y * frame.shape[0])
                apply_earring(frame1, earring_img, x_ear_right, y_ear_right, is_left=False)

    return frame1

def apply_earring(frame, earring_img, x_ear, y_ear, is_left):
    distance = frame.shape[1] * 0.15
    scale = distance / earring_img.shape[1]
    new_w = int(earring_img.shape[1] * scale)
    new_h = int(earring_img.shape[0] * scale)

    if is_left:
        overlay = cv2.flip(earring_img, 1)
    else:
        overlay = earring_img.copy()

    overlay = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)

    x_center = x_ear
    y_center = y_ear + int(distance * 0.5)
    h, w = frame.shape[:2]
    x_start = max(0, x_center - new_w // 2)
    y_start = max(0, y_center - new_h // 2)
    x_end = min(w, x_start + new_w)
    y_end = min(h, y_start + new_h)

    overlay_resized = overlay[:(y_end - y_start), :(x_end - x_start)]

    alpha_s = overlay_resized[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        frame[y_start:y_end, x_start:x_end, c] = (alpha_s * overlay_resized[:, :, c] +
                                                  alpha_l * frame[y_start:y_end, x_start:x_end, c])
    logging.debug(f"Earring applied at: x_start: {x_start}, y_start: {y_start}, x_end: {x_end}, y_end: {y_end}")

def add_bracelet_overlay(frame, bracelet_img):
    frame1 = frame.copy()
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if len(hand_landmarks.landmark) >= 2:
                x1 = int(hand_landmarks.landmark[0].x * frame.shape[1])
                y1 = int(hand_landmarks.landmark[0].y * frame.shape[0])

                reference_size = 100
                scaling_factor = 1.2
                size_factor = (reference_size / bracelet_img.shape[1]) * scaling_factor
                bracelet_resized = cv2.resize(bracelet_img, None, fx=size_factor, fy=size_factor)

                x_offset = x1 - bracelet_resized.shape[1] // 2
                y_offset = y1 - bracelet_resized.shape[0] // 2 + 30

                if 0 <= y_offset < frame.shape[0] - bracelet_resized.shape[0] and 0 <= x_offset < frame.shape[1] - bracelet_resized.shape[1]:
                    overlay = frame[y_offset:y_offset + bracelet_resized.shape[0], x_offset:x_offset + bracelet_resized.shape[1]]
                    if overlay.shape[0] == bracelet_resized.shape[0] and overlay.shape[1] == bracelet_resized.shape[1]:
                        alpha_s = bracelet_resized[:, :, 3] / 255.0
                        alpha_l = 1.0 - alpha_s
                        for c in range(0, 3):
                            overlay[:, :, c] = (alpha_s * bracelet_resized[:, :, c] +
                                                alpha_l * overlay[:, :, c])
                        frame1[y_offset:y_offset + bracelet_resized.shape[0], x_offset:x_offset + bracelet_resized.shape[1]] = overlay
                        logging.debug(f"Bracelet applied at: x_start: {x_offset}, y_start: {y_offset}, x_end: {x_offset + bracelet_resized.shape[1]}, y_end: {y_offset + bracelet_resized.shape[0]}")
    return frame1

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

