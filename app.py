# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import cv2
# import mediapipe as mp
# import numpy as np
# import base64
# from io import BytesIO
# from PIL import Image
# import uuid
# import logging
#
# app = Flask(__name__)
# CORS(app)
#
# logging.basicConfig(level=logging.DEBUG)
#
# # Initialize Mediapipe Face Mesh and Hands
# mp_face_mesh = mp.solutions.face_mesh
# mp_hands = mp.solutions.hands
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
# hands = mp_hands.Hands()
#
# # Temporary storage for uploaded jewelry images
# temp_storage = {
#     "necklace": None,
#     "ring": None,
#     "earring": None,
#     "bracelet": None
# }
#
# @app.route("/")
# def home():
#     return "Flask server is running here."
#
# @app.route("/upload_<jewelry_type>", methods=["POST"])
# def upload_jewelry(jewelry_type):
#     jewelry_type = jewelry_type.lower()
#     if jewelry_type not in temp_storage:
#         return jsonify({"error": "Invalid jewelry type"}), 400
#
#     data = request.json
#     image_data = data.get("image")
#     if not image_data:
#         return jsonify({"error": f"{jewelry_type.capitalize()} image data not provided"}), 400
#
#     image = Image.open(BytesIO(base64.b64decode(image_data.split(",")[1])))
#     jewelry_id = str(uuid.uuid4())
#     temp_storage[jewelry_type] = {
#         "id": jewelry_id,
#         "image": cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
#     }
#     logging.debug(f"Uploaded {jewelry_type} image with ID: {jewelry_id}, Shape: {temp_storage[jewelry_type]['image'].shape}")
#     return jsonify({f"{jewelry_type}_id": jewelry_id})
#
# # @app.route("/process_frame", methods=["POST"])
# # def process_frame():
# #     data = request.json
# #     if not data or 'frame' not in data:
# #         return jsonify({"error": "Frame not provided"}), 400
#
# #     frame_data = data["frame"].split(",")[1]
# #     frame = Image.open(BytesIO(base64.b64decode(frame_data)))
# #     frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
#
# #     logging.debug("Processing frame")
#
# #     jewelry_types = data.get("jewelry_types", {})
#
# #     logging.debug(f"Jewelry types: {jewelry_types}")
# #     logging.debug(f"Available keys in temp_storage: {list(temp_storage.keys())}")
#
# #     for jewelry_type, jewelry_id in jewelry_types.items():
# #         if jewelry_type in temp_storage and temp_storage[jewelry_type] is not None and jewelry_id == temp_storage[jewelry_type]["id"]:
# #             logging.debug(f"Overlaying {jewelry_type} with ID: {jewelry_id}")
# #             if jewelry_type == "necklace":
# #                 frame = overlay_necklace(frame, temp_storage[jewelry_type]["image"])
# #             elif jewelry_type == "ring":
# #                 frame = overlay_ring_on_hand(frame, temp_storage[jewelry_type]["image"])
# #             elif jewelry_type == "earring":
# #                 frame = overlay_earring(frame, temp_storage[jewelry_type]["image"])
# #             elif jewelry_type == "bracelet":
# #                 frame = add_bracelet_overlay(frame, temp_storage[jewelry_type]["image"])
#
# #     _, buffer = cv2.imencode(".jpg", frame)
# #     jpg_as_text = base64.b64encode(buffer).decode("utf-8")
#
# #     return jsonify({"image": jpg_as_text})
# @app.route("/process_frame", methods=["POST"])
# def process_frame():
#     try:
#         data = request.json
#         if not data or 'frame' not in data:
#             logging.error("Frame not provided")
#             return jsonify({"error": "Frame not provided"}), 400
#
#         frame_data = data["frame"].split(",")[1]
#
#         # Ensure proper padding
#         missing_padding = len(frame_data) % 4
#         if missing_padding:
#             frame_data += '=' * (4 - missing_padding)
#
#         frame = Image.open(BytesIO(base64.b64decode(frame_data)))
#         frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
#
#         logging.debug("Processing frame")
#
#         jewelry_types = data.get("jewelry_types", {})
#
#         logging.debug(f"Jewelry types: {jewelry_types}")
#         logging.debug(f"Available keys in temp_storage: {list(temp_storage.keys())}")
#
#         for jewelry_type, jewelry_id in jewelry_types.items():
#             if jewelry_type in temp_storage and temp_storage[jewelry_type] is not None and jewelry_id == temp_storage[jewelry_type]["id"]:
#                 logging.debug(f"Overlaying {jewelry_type} with ID: {jewelry_id}")
#                 if jewelry_type == "necklace":
#                     frame = overlay_necklace(frame, temp_storage[jewelry_type]["image"])
#                 elif jewelry_type == "ring":
#                     frame = overlay_ring_on_hand(frame, temp_storage[jewelry_type]["image"])
#                 elif jewelry_type == "earring":
#                     frame = overlay_earring(frame, temp_storage[jewelry_type]["image"])
#                 elif jewelry_type == "bracelet":
#                     frame = add_bracelet_overlay(frame, temp_storage[jewelry_type]["image"])
#
#         _, buffer = cv2.imencode(".jpg", frame)
#         jpg_as_text = base64.b64encode(buffer).decode("utf-8")
#
#         return jsonify({"image": jpg_as_text})
#     except Exception as e:
#         logging.error(f"Error processing frame: {e}")
#         return jsonify({"error": "Internal Server Error"}), 500
#
#
# # Existing overlay functions remain the same
# def overlay_necklace(frame, necklace_img):
#     offset_y = 30
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb_frame)
#     frame1 = frame.copy()
#
#     if results.multi_face_landmarks:
#         logging.debug("Found face landmarks")
#         for face_landmarks in results.multi_face_landmarks:
#             landmark_points = face_landmarks.landmark
#             x_chin, y_chin = int(landmark_points[152].x * frame.shape[1]), int(landmark_points[152].y * frame.shape[0])
#             x_neck, y_neck = int(landmark_points[9].x * frame.shape[1]), int(landmark_points[9].y * frame.shape[0])
#             y_chin += offset_y
#
#             distance = np.sqrt((x_neck - x_chin) ** 2 + (y_neck - y_chin) ** 2)
#             scale = distance / necklace_img.shape[1]
#             new_w = int(necklace_img.shape[1] * scale)
#             new_h = int(necklace_img.shape[0] * scale)
#             overlay = cv2.resize(necklace_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
#
#             x_center = x_chin
#             y_center = y_chin + int(distance / 3)
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
#             for c in range(0, 3):
#                 frame1[y_start:y_end, x_start:x_end, c] = (alpha_s * overlay_resized[:, :, c] +
#                                                            alpha_l * frame[y_start:y_end, x_start:x_end, c])
#             logging.debug(f"Necklace applied at: x_start: {x_start}, y_start: {y_start}, x_end: {x_end}, y_end: {y_end}")
#     else:
#         logging.debug("No face landmarks found")
#     return frame1
#
# def overlay_ring_on_hand(frame, ring_img):
#     frame1 = frame.copy()
#     image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(image_rgb)
#
#     if results.multi_hand_landmarks:
#         logging.debug("Found hand landmarks")
#         for hand_landmarks in results.multi_hand_landmarks:
#             if len(hand_landmarks.landmark) >= 14:
#                 x1 = int(hand_landmarks.landmark[13].x * frame.shape[1])
#                 y1 = int(hand_landmarks.landmark[13].y * frame.shape[0])
#                 x2 = int(hand_landmarks.landmark[14].x * frame.shape[1])
#                 y2 = int(hand_landmarks.landmark[14].y * frame.shape[0])
#
#                 x_mid = (x1 + x2) // 2
#                 y_mid = (y1 + y2) // 2
#                 reference_size = 50
#                 size_factor = reference_size / ring_img.shape[1]
#                 ring_resized = cv2.resize(ring_img, None, fx=size_factor, fy=size_factor)
#
#                 x_offset = x_mid - ring_resized.shape[1] // 2
#                 y_offset = y_mid - ring_resized.shape[0] // 2
#                 overlay = frame1[y_offset:y_offset + ring_resized.shape[0], x_offset:x_offset + ring_resized.shape[1]]
#                 if overlay.shape[0] == ring_resized.shape[0] and overlay.shape[1] == ring_resized.shape[1]:
#                     alpha_s = ring_resized[:, :, 3] / 255.0
#                     alpha_l = 1.0 - alpha_s
#                     for c in range(0, 3):
#                         overlay[:, :, c] = (alpha_s * ring_resized[:, :, c] + alpha_l * overlay[:, :, c])
#                     frame1[y_offset:y_offset + ring_resized.shape[0], x_offset:x_offset + ring_resized.shape[1]] = overlay
#                 logging.debug(f"Ring applied at: x_start: {x_offset}, y_start: {y_offset}, x_end: {x_offset + ring_resized.shape[1]}, y_end: {y_offset + ring_resized.shape[0]}")
#     else:
#         logging.debug("No hand landmarks found")
#     return frame1
#
# def overlay_earring(frame, earring_img):
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb_frame)
#     frame1 = frame.copy()
#
#     if results.multi_face_landmarks:
#         logging.debug("Found face landmarks")
#         for face_landmarks in results.multi_face_landmarks:
#             landmark_points = face_landmarks.landmark
#             z_ear_left = landmark_points[234].z
#             z_ear_right = landmark_points[454].z
#
#             if z_ear_left <= z_ear_right + 0.1:
#                 x_ear_left = int(landmark_points[234].x * frame.shape[1])
#                 y_ear_left = int(landmark_points[234].y * frame.shape[0])
#                 apply_earring(frame1, earring_img, x_ear_left, y_ear_left, is_left=True)
#
#             if z_ear_right <= z_ear_left + 0.1:
#                 x_ear_right = int(landmark_points[454].x * frame.shape[1])
#                 y_ear_right = int(landmark_points[454].y * frame.shape[0])
#                 apply_earring(frame1, earring_img, x_ear_right, y_ear_right, is_left=False)
#     else:
#         logging.debug("No face landmarks found")
#     return frame1
#
# def apply_earring(frame, earring_img, x_ear, y_ear, is_left):
#     distance = frame.shape[1] * 0.15
#     scale = distance / earring_img.shape[1]
#     new_w = int(earring_img.shape[1] * scale)
#     new_h = int(earring_img.shape[0] * scale)
#
#     if is_left:
#         overlay = cv2.flip(earring_img, 1)
#     else:
#         overlay = earring_img.copy()
#
#     overlay = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)
#
#     x_center = x_ear
#     y_center = y_ear + int(distance * 0.5)
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
#     for c in range(0, 3):
#         frame[y_start:y_end, x_start:x_end, c] = (alpha_s * overlay_resized[:, :, c] +
#                                                   alpha_l * frame[y_start:y_end, x_start:x_end, c])
#     logging.debug(f"Earring applied at: x_start: {x_start}, y_start: {y_start}, x_end: {x_end}, y_end: {y_end}")
#
# def add_bracelet_overlay(frame, bracelet_img):
#     frame1 = frame.copy()
#     image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(image_rgb)
#
#     if results.multi_hand_landmarks:
#         logging.debug("Found hand landmarks")
#         for hand_landmarks in results.multi_hand_landmarks:
#             if len(hand_landmarks.landmark) >= 2:
#                 x1 = int(hand_landmarks.landmark[0].x * frame.shape[1])
#                 y1 = int(hand_landmarks.landmark[0].y * frame.shape[0])
#
#                 reference_size = 100
#                 scaling_factor = 1.2
#                 size_factor = (reference_size / bracelet_img.shape[1]) * scaling_factor
#                 bracelet_resized = cv2.resize(bracelet_img, None, fx=size_factor, fy=size_factor)
#
#                 x_offset = x1 - bracelet_resized.shape[1] // 2
#                 y_offset = y1 - bracelet_resized.shape[0] // 2 + 30
#
#                 if 0 <= y_offset < frame.shape[0] - bracelet_resized.shape[0] and 0 <= x_offset < frame.shape[1] - bracelet_resized.shape[1]:
#                     overlay = frame[y_offset:y_offset + bracelet_resized.shape[0], x_offset:x_offset + bracelet_resized.shape[1]]
#                     if overlay.shape[0] == bracelet_resized.shape[0] and overlay.shape[1] == bracelet_resized.shape[1]:
#                         alpha_s = bracelet_resized[:, :, 3] / 255.0
#                         alpha_l = 1.0 - alpha_s
#                         for c in range(0, 3):
#                             overlay[:, :, c] = (alpha_s * bracelet_resized[:, :, c] +
#                                                 alpha_l * overlay[:, :, c])
#                         frame1[y_offset:y_offset + bracelet_resized.shape[0], x_offset:x_offset + bracelet_resized.shape[1]] = overlay
#                         logging.debug(f"Bracelet applied at: x_start: {x_offset}, y_start: {y_offset}, x_end: {x_offset + bracelet_resized.shape[1]}, y_end: {y_offset + bracelet_resized.shape[0]}")
#     else:
#         logging.debug("No hand landmarks found")
#     return frame1
#
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)
#
#
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

# Temporary storage for uploaded jewelry images
temp_storage = {
    "necklace": None,
    "ring": None,
    "earring": None,
    "bracelet": None
}

@app.route("/")
def home():
    return "Flask server is running here."

@app.route("/upload_<jewelry_type>", methods=["POST"])
def upload_jewelry(jewelry_type):
    jewelry_type = jewelry_type.lower()
    if jewelry_type not in temp_storage:
        return jsonify({"error": "Invalid jewelry type"}), 400

    data = request.json
    image_data = data.get("image")
    if not image_data:
        return jsonify({"error": f"{jewelry_type.capitalize()} image data not provided"}), 400

    image = Image.open(BytesIO(base64.b64decode(image_data.split(",")[1])))
    jewelry_id = str(uuid.uuid4())
    temp_storage[jewelry_type] = {
        "id": jewelry_id,
        "image": cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
    }
    logging.debug(f"Uploaded {jewelry_type} image with ID: {jewelry_id}, Shape: {temp_storage[jewelry_type]['image'].shape}")
    return jsonify({f"{jewelry_type}_id": jewelry_id})

# @app.route("/process_frame", methods=["POST"])
# def process_frame():
#     data = request.json
#     if not data or 'frame' not in data:
#         return jsonify({"error": "Frame not provided"}), 400

#     frame_data = data["frame"].split(",")[1]
#     frame = Image.open(BytesIO(base64.b64decode(frame_data)))
#     frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

#     logging.debug("Processing frame")

#     jewelry_types = data.get("jewelry_types", {})

#     logging.debug(f"Jewelry types: {jewelry_types}")
#     logging.debug(f"Available keys in temp_storage: {list(temp_storage.keys())}")

#     for jewelry_type, jewelry_id in jewelry_types.items():
#         if jewelry_type in temp_storage and temp_storage[jewelry_type] is not None and jewelry_id == temp_storage[jewelry_type]["id"]:
#             logging.debug(f"Overlaying {jewelry_type} with ID: {jewelry_id}")
#             if jewelry_type == "necklace":
#                 frame = overlay_necklace(frame, temp_storage[jewelry_type]["image"])
#             elif jewelry_type == "ring":
#                 frame = overlay_ring_on_hand(frame, temp_storage[jewelry_type]["image"])
#             elif jewelry_type == "earring":
#                 frame = overlay_earring(frame, temp_storage[jewelry_type]["image"])
#             elif jewelry_type == "bracelet":
#                 frame = add_bracelet_overlay(frame, temp_storage[jewelry_type]["image"])

#     _, buffer = cv2.imencode(".jpg", frame)
#     jpg_as_text = base64.b64encode(buffer).decode("utf-8")

#     return jsonify({"image": jpg_as_text})
@app.route("/process_frame", methods=["POST"])
def process_frame():
    try:
        data = request.json
        if not data or 'frame' not in data:
            logging.error("Frame not provided")
            return jsonify({"error": "Frame not provided"}), 400

        frame_data = data["frame"].split(",")[1]

        # Ensure proper padding
        missing_padding = len(frame_data) % 4
        if missing_padding:
            frame_data += '=' * (4 - missing_padding)

        frame = Image.open(BytesIO(base64.b64decode(frame_data)))
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

        logging.debug("Processing frame")

        jewelry_types = data.get("jewelry_types", {})

        logging.debug(f"Jewelry types: {jewelry_types}")
        logging.debug(f"Available keys in temp_storage: {list(temp_storage.keys())}")

        for jewelry_type, jewelry_id in jewelry_types.items():
            if jewelry_type in temp_storage and temp_storage[jewelry_type] is not None and jewelry_id == temp_storage[jewelry_type]["id"]:
                logging.debug(f"Overlaying {jewelry_type} with ID: {jewelry_id}")
                if jewelry_type == "necklace":
                    frame = overlay_necklace(frame, temp_storage[jewelry_type]["image"])
                elif jewelry_type == "ring":
                    frame = overlay_ring_on_hand(frame, temp_storage[jewelry_type]["image"])
                elif jewelry_type == "earring":
                    frame = overlay_earring(frame, temp_storage[jewelry_type]["image"])
                elif jewelry_type == "bracelet":
                    frame = add_bracelet_overlay(frame, temp_storage[jewelry_type]["image"])

        _, buffer = cv2.imencode(".jpg", frame)
        jpg_as_text = base64.b64encode(buffer).decode("utf-8")

        return jsonify({"image": jpg_as_text})
    except Exception as e:
        logging.error(f"Error processing frame: {e}")
        return jsonify({"error": "Internal Server Error"}), 500


# Existing overlay functions remain the same
def overlay_necklace(frame, necklace_img):
    offset_y = 30
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    frame1 = frame.copy()

    if results.multi_face_landmarks:
        logging.debug("Found face landmarks")
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
    else:
        logging.debug("No face landmarks found")
    return frame1

# def overlay_ring_on_hand(frame, ring_img):
#     frame1 = frame.copy()
#     image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(image_rgb)
#
#     if results.multi_hand_landmarks:
#         logging.debug("Found hand landmarks")
#         for hand_landmarks in results.multi_hand_landmarks:
#             if len(hand_landmarks.landmark) >= 14:
#                 x1 = int(hand_landmarks.landmark[13].x * frame.shape[1])
#                 y1 = int(hand_landmarks.landmark[13].y * frame.shape[0])
#                 x2 = int(hand_landmarks.landmark[14].x * frame.shape[1])
#                 y2 = int(hand_landmarks.landmark[14].y * frame.shape[0])
#
#                 x_mid = (x1 + x2) // 2
#                 y_mid = (y1 + y2) // 2
#                 reference_size = 50
#                 size_factor = reference_size / ring_img.shape[1]
#                 ring_resized = cv2.resize(ring_img, None, fx=size_factor, fy=size_factor)
#
#                 x_offset = x_mid - ring_resized.shape[1] // 2
#                 y_offset = y_mid - ring_resized.shape[0] // 2
#                 overlay = frame1[y_offset:y_offset + ring_resized.shape[0], x_offset:x_offset + ring_resized.shape[1]]
#                 if overlay.shape[0] == ring_resized.shape[0] and overlay.shape[1] == ring_resized.shape[1]:
#                     alpha_s = ring_resized[:, :, 3] / 255.0
#                     alpha_l = 1.0 - alpha_s
#                     for c in range(0, 3):
#                         overlay[:, :, c] = (alpha_s * ring_resized[:, :, c] + alpha_l * overlay[:, :, c])
#                     frame1[y_offset:y_offset + ring_resized.shape[0], x_offset:x_offset + ring_resized.shape[1]] = overlay
#                 logging.debug(f"Ring applied at: x_start: {x_offset}, y_start: {y_offset}, x_end: {x_offset + ring_resized.shape[1]}, y_end: {y_offset + ring_resized.shape[0]}")
#     else:
#         logging.debug("No hand landmarks found")
#     return frame1
def overlay_ring_on_hand(frame, ring_img):
    frame1 = frame.copy()
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        logging.debug("Found hand landmarks")
        for hand_landmarks in results.multi_hand_landmarks:
            if len(hand_landmarks.landmark) >= 14:
                x1 = int(hand_landmarks.landmark[13].x * frame.shape[1])
                y1 = int(hand_landmarks.landmark[13].y * frame.shape[0])
                x2 = int(hand_landmarks.landmark[14].x * frame.shape[1])
                y2 = int(hand_landmarks.landmark[14].y * frame.shape[0])

                distance = int(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

                # Adjust ring size to fit between landmarks 13 and 14
                scale_factor = distance / ring_img.shape[1]
                ring_resized = cv2.resize(ring_img, None, fx=scale_factor, fy=scale_factor,
                                          interpolation=cv2.INTER_AREA)

                x_mid = (x1 + x2) // 2
                y_mid = (y1 + y2) // 2

                x_offset = x_mid - ring_resized.shape[1] // 2
                y_offset = y_mid - ring_resized.shape[0] // 2
                overlay = frame1[y_offset:y_offset + ring_resized.shape[0], x_offset:x_offset + ring_resized.shape[1]]

                if overlay.shape[0] == ring_resized.shape[0] and overlay.shape[1] == ring_resized.shape[1]:
                    alpha_s = ring_resized[:, :, 3] / 255.0
                    alpha_l = 1.0 - alpha_s
                    for c in range(0, 3):
                        overlay[:, :, c] = (alpha_s * ring_resized[:, :, c] + alpha_l * overlay[:, :, c])
                    frame1[y_offset:y_offset + ring_resized.shape[0],
                    x_offset:x_offset + ring_resized.shape[1]] = overlay
                logging.debug(
                    f"Ring applied at: x_start: {x_offset}, y_start: {y_offset}, x_end: {x_offset + ring_resized.shape[1]}, y_end: {y_offset + ring_resized.shape[0]}")
    else:
        logging.debug("No hand landmarks found")
    return frame1
def overlay_earring(frame, earring_img):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    frame1 = frame.copy()

    if results.multi_face_landmarks:
        logging.debug("Found face landmarks")
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
    else:
        logging.debug("No face landmarks found")
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
        logging.debug("Found hand landmarks")
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
    else:
        logging.debug("No hand landmarks found")
    return frame1

if __name__ == "__main__":
    app.run()
