import cv2
import mediapipe as mp
import numpy as np
import time
import camDetectionFunctions
import nithFunctions

# Initialize the face mesh model.
mp_face_mesh = mp.solutions.face_mesh

# Create a face mesh object with minimum detection confidence and minimum tracking confidence thresholds.
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.1,  # Minimum confidence value for the face detection model
    min_tracking_confidence=0.1,  # Minimum confidence value for the landmark tracking model)
)
# Import the drawing utilities module from Mediapipe.
mp_drawing = mp.solutions.drawing_utils

# Define the specifications for drawing the face mesh annotations.
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Create a video capture object to access the webcam feed (by default, the webcam with index 0 is used).
cap = cv2.VideoCapture(0)


def main_loop():
    while cap.isOpened():
        start = time.time()
        success, image = cap.read()

        # Flip image horizontally
        # Convert space from BGR to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance
        image.flags.writeable = False

        # Get result
        results = face_mesh.process(image)

        # Improve performance again
        image.flags.writeable = True

        # Convert the color space from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape

        # If there are results
        if results.multi_face_landmarks:
            # Get first face
            face_landmarks = results.multi_face_landmarks[0]

            # Get orientation
            x, y, z = camDetectionFunctions.get_face_orientation(
                face_landmarks, img_w, img_h
            )

            # Get eyes aperture ratio
            # using eye aperture ratio
            # (
            #     left_eye_aperture_ratio,
            #     right_eye_aperture_ratio,
            # ) = camDetectionFunctions.get_eye_aperture_ratio_SEGMENTSMETHOD(
            #     face_landmarks
            # )

            # using areas
            # (
            #     left_eye_aperture_ratio,
            #     right_eye_aperture_ratio,
            # ) = camDetectionFunctions.get_eye_aperture_ratio_SEGMENTSMETHOD(
            #     face_landmarks
            # )

            # using twosegmentsmethod
            (
                left_eye_aperture_ratio,
                right_eye_aperture_ratio,
            ) = camDetectionFunctions.get_eye_aperture_ratio_TWOSEGMENTSMETHOD(
                face_landmarks
            )

            # Get mouth aperture ratio
            mouth_aperture_ratio = camDetectionFunctions.get_mouth_aperture_ratio(
                face_landmarks
            )

            head_roll = camDetectionFunctions.get_head_roll(face_landmarks)

            # region Put head rotation text
            cv2.putText(
                image,
                "Y: " + str(np.round(x, 2)),
                (500, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                image,
                "P: " + str(np.round(y, 2)),
                (500, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                image,
                "R: " + str(np.round(head_roll, 2)),
                (500, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            # endregion

            # region Put eyes text
            cv2.putText(
                image,
                "LE_AR: " + str(np.round(left_eye_aperture_ratio, 2)),
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                image,
                "RE_AR: " + str(np.round(right_eye_aperture_ratio, 2)),
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            # endregion

            # region Put mouth aperture text
            cv2.putText(
                image,
                "Mouth aperture: " + str(np.round(mouth_aperture_ratio, 2)),
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            # endregion

            # region Put FPS text
            end = time.time()
            totalTime = end - start

            fps = 1 / totalTime

            cv2.putText(
                image,
                f"FPS: {int(fps)}",
                (20, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                landmark_drawing_spec=drawing_spec,
            )
            # endregion

            nithFunctions.send_data(
                x * 3,
                y * 3,
                head_roll,
                mouth_aperture_ratio,
                left_eye_aperture_ratio,
                right_eye_aperture_ratio,
            )

        cv2.imshow("Head Pose Estimation", image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
