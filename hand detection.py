import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 520)  
cap.set(4, 520)  

detector = HandDetector(detectionCon=0.7, maxHands=2)

while True:
    
    success, img = cap.read()
    if not success:
        break

    flipped_frame = cv2.flip(img, 1)

    hands, flipped_frame = detector.findHands(flipped_frame)

    if hands:
        for hand in hands:
            lmList = hand["lmList"]  
            bbox = hand["bbox"]      
            handType = hand["type"]   

            fingers = detector.fingersUp(hand)

            if handType == "Left":
                display_handType = "Right"
            elif handType == "Right":
                display_handType = "Left"

            if display_handType == "Left":
                left_thumb, left_index, left_middle, left_ring, left_pinky = fingers

                if left_thumb == 1 and left_index == 0 and left_middle == 0 and left_ring == 0 and left_pinky == 0:
                    cv2.putText(flipped_frame, "Thumbs up",
                                (bbox[0], bbox[1] - 30),  
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, (255, 0, 0), 2)
                elif left_thumb == 1 and left_index == 1 and left_middle == 0 and left_ring == 0 and left_pinky == 1:
                    cv2.putText(flipped_frame, "Rock", 
                                (bbox[0], bbox[1] - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, (255, 0, 0), 2)
                elif left_thumb == 0 and left_index == 1 and left_middle == 1 and left_ring == 0 and left_pinky == 0:
                    cv2.putText(flipped_frame, "Victory", 
                                (bbox[0], bbox[1] - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, (255, 0, 0), 2)

            elif display_handType == "Right":
                right_thumb, right_index, right_middle, right_ring, right_pinky = fingers

                if right_thumb == 1 and right_index == 0 and right_middle == 0 and right_ring == 0 and right_pinky == 0:
                    cv2.putText(flipped_frame, "Thumbs up", 
                                (bbox[0], bbox[1] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, (0, 255, 0), 2)
                elif right_thumb == 1 and right_index == 1 and right_middle == 0 and right_ring == 0 and right_pinky == 1:
                    cv2.putText(flipped_frame, "Rock", 
                                (bbox[0], bbox[1] - 30),  
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, (0, 255, 0), 2)
                elif right_thumb == 0 and right_index == 1 and right_middle == 1 and right_ring == 0 and right_pinky == 0:
                    cv2.putText(flipped_frame, "victory", 
                                (bbox[0], bbox[1] - 30),  
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, (0, 255, 0), 2)

    cv2.imshow("Smart Camera", flipped_frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
