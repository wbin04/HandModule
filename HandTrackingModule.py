import cv2
from HandDectectionModule import HandDetector

class HandTracking:
    def __init__(self, detection_confidence=0.8, max_hands=2):
        self.cap = cv2.VideoCapture(0)
        self.detector = HandDetector(detection_confidence, max_hands)

    def detect_hands(self):
        success, img = self.cap.read()
        if not success:
            return None, None
        
        hands, img = self.detector.find_hands(img)
        return hands, img

    def process_hands(self, hands, img):
        if hands:
            for hand in hands:
                bbox = hand["bbox"]
                hand_type = hand["type"]
                
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 255), 2)
                
                cv2.putText(img, hand_type, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                                
                if len(hands) == 2:
                    hand1, hand2 = hands[0], hands[1]
                    centerPoint1, centerPoint2 = hand1["center"], hand2["center"]
                    
                    length = ((centerPoint1[0] - centerPoint2[0]) ** 2 + (centerPoint1[1] - centerPoint2[1]) ** 2) ** 0.5
                    cv2.line(img, centerPoint1, centerPoint2, (0, 255, 0), 2)
                    midPoint = ((centerPoint1[0] + centerPoint2[0]) // 2, (centerPoint1[1] + centerPoint2[1]) // 2)
                    cv2.putText(img, f"Dist: {int(length)} pixels", midPoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img

    def run(self):
        while True:
            hands, img = self.detect_hands()
            if img is not None:
                img = self.process_hands(hands, img)
                cv2.imshow("Hand Tracking", img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()