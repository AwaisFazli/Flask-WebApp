import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
        np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return int(angle)


def calculate_distance(a, b):
    x1, y1 = a
    x2, y2 = b

    x1 = x1*640
    x2 = x2*640

    y1 = y1*480
    y2 = y2*480

    dist = m.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist


font = cv2.FONT_HERSHEY_SIMPLEX
light_green = (127, 233, 100)

count = 0
pos = None


class Video(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
            ref, frame = self.video.read()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Calculate angle
                angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                # dist = calculate_distance(shoulder, wrist)
                # print(dist)

                # Visualize angle
                cv2.putText(image, str(angle),
                            tuple(np.multiply(
                                l_elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (
                                255, 255, 255), 2, cv2.LINE_AA
                            )

                if (angle >= 165):
                    pos = "up"

                if (angle <= 85 and pos == "up"):
                    pos = "down"
                    count += 1

                cv2.putText(image, str(int(count)), (50, 50),
                            font, 0.9, light_green, 2)

            except:
                pass

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(
                                          color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(
                                          color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            # if (angle >= 165):
            #     pos = "up"

            # if (angle <= 85 and pos == "up"):
            #     pos = "down"
            #     count += 1

            # cv2.putText(image, str(int(count)), (50, 50),
            #             font, 0.9, light_green, 2)

            ret, jpg = cv2.imencode('.jpg', image)
            return jpg.tobytes()
