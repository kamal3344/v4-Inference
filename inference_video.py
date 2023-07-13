# importing the required packages
import cv2
import numpy as np
import time
class Yolov4:
    def __init__(self):
        self.weights = 'D:/Computer vision/Models/YOLO v4/pretrained_coco_files/yolov4.weights'  # loading weights
        self.cfg = 'D:/Computer vision/Models/YOLO v4/pretrained_coco_files/yolov4.cfg'  # loading cfg file
        self.classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
                        'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                        'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
                        'toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard',
                        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                        'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self.Neural_Network = cv2.dnn.readNetFromDarknet(self.cfg, self.weights)
        self.outputs = self.Neural_Network.getUnconnectedOutLayersNames()
        self.COLORS = np.random.randint(0, 255, size=(len(self.classes), 3), dtype="uint8")
        self.image_size = 320

    def bounding_box(self, detections):
        confidence_score = []
        ids = []
        cordinates = []
        Threshold = 0.5
        for i in detections:
            for j in i:
                probs_values = j[5:]
                class_ = np.argmax(probs_values)
                confidence_ = probs_values[class_]

                if confidence_ > Threshold:
                    w , h = int(j[2] * self.image_size) , int(j[3] * self.image_size)
                    x , y = int(j[0] * self.image_size - w / 2) , int(j[1] * self.image_size - h / 2)
                    cordinates.append([x,y,w,h])
                    ids.append(class_)
                    confidence_score.append(float(confidence_))
        final_box = cv2.dnn.NMSBoxes(cordinates , confidence_score , Threshold , .6)
        return final_box , cordinates , confidence_score , ids


    def predictions(self,prediction_box, bounding_box, confidence, class_labels, width_ratio, height_ratio,end_time,image):
        for j in prediction_box.flatten():
            x, y, w, h = bounding_box[j]
            x = int(x * width_ratio)
            y = int(y * height_ratio)
            w = int(w * width_ratio)
            h = int(h * height_ratio)
            label = str(self.classes[class_labels[j]])
            conf_ = str(round(confidence[j], 2))
            color = [int(c) for c in self.COLORS[class_labels[j]]]
            cv2.rectangle(image, (x, y), (x + w, y + h),color, 2)
            cv2.putText(image, label + ' ' + conf_, (x, y - 2), cv2.FONT_HERSHEY_COMPLEX, .5, color, 2)
            time=f"Inference time: {end_time:.3f}"
            cv2.putText(image, time ,(10,13), cv2.FONT_HERSHEY_COMPLEX, .5, (156,0,166), 1)
        return image

    def Inference(self, image,original_width,original_height):
        blob = cv2.dnn.blobFromImage(image, 1 / 255, (320, 320), True, crop=False)
        self.Neural_Network.setInput(blob)
        start_time=time.time()
        output_data = self.Neural_Network.forward(self.outputs)
        end_time=time.time()-start_time
        final_box, cordinates, confidence_score, ids = self.bounding_box(output_data)
        outcome=self.predictions(final_box , cordinates , confidence_score , ids ,original_width / 320,original_height / 320,end_time,image)
        return outcome

if __name__ == "__main__":
    obj = Yolov4()  # constructor called and executed
    cap = cv2.VideoCapture('traffic_signs.mp4')
    fps = cv2.CAP_PROP_FPS
    width = cap.get(3)
    height = cap.get(4)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter("demo.avi", fourcc, fps, (int(width), int(height)))
    while cap.isOpened():
        res, frame = cap.read()
        if res == True:
            img = frame
            outcome = obj.Inference(image=frame,original_width=width,original_height=height)
            cv2.imshow("demo", outcome)
            output.write(outcome)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
