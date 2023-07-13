# importing the required packages
import cv2
import numpy as np
import time
class Yolov4:
    def __init__(self):
        self.weights = 'D:/Computer vision/Models/YOLO v4/custom/yolov4-tiny_best.weights'  # loading weights
        self.cfg = 'D:/Computer vision/Models/YOLO v4/custom/yolov4-tiny.cfg'  # loading cfg file
        self.classes = ['Person','Helmet','Reflective_Jacket','Shoe','Bend','Truck','Fork_Lift']
        self.Neural_Network = cv2.dnn.readNetFromDarknet(self.cfg, self.weights)
        self.outputs = self.Neural_Network.getUnconnectedOutLayersNames()
        self.COLORS = np.random.randint(0, 255, size=(len(self.classes), 3), dtype="uint8")
        self.image_size = 608

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


    def predictions(self,prediction_box, bounding_box, confidence, class_labels, width_ratio, height_ratio,end_time):
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
            cv2.putText(image, label + ' ' + conf_, (x, y - 2), cv2.FONT_HERSHEY_COMPLEX, .9, color, 2)
            time=f"Inference time: {end_time:.3f}"
            cv2.putText(image, time ,(10,13), cv2.FONT_HERSHEY_COMPLEX, .5, (156,0,166), 1)

    def Inference(self, image,original_width,original_height):
        blob = cv2.dnn.blobFromImage(image, 1 / 255, (608, 608), True, crop=False)
        self.Neural_Network.setInput(blob)
        start_time=time.time()
        output_data = self.Neural_Network.forward(self.outputs)
        end_time=time.time()-start_time
        self.bounding_box(output_data)
        final_box, cordinates, confidence_score, ids = self.bounding_box(output_data)
        self.predictions(final_box , cordinates , confidence_score , ids ,original_width / 608,original_height / 608,end_time)

if __name__ == "__main__":
    obj = Yolov4()  # constructor called and executed
    image = 'custom.jpg'
    image = cv2.imread(image, 1)
    original_width , original_height = image.shape[1] , image.shape[0]
    obj.Inference(image=image,original_width=original_width,original_height=original_height)
    cv2.imshow('Inference ',image)
    cv2.waitKey()
    cv2.destroyAllWindows()