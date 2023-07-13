import cv2
import numpy as np 
import matplotlib.pyplot as plt 


Threshold = 0.5
image_size = 320

b = []

def predictions(final_box , cordinates , confidence_score , ids ,width_ratio,height_ratio):
    count1 = 0
    count2 = 0
    k = []
    c = []
    
    for i in final_box.flatten():
        if classes_names[ids[i]] == 'motorcycle':
            count1+=1
            k.append(count1)
            x , y , w , h = cordinates[i]
            x = int(x * width_ratio)
            y = int(y * height_ratio)
            w = int(w * width_ratio)
            h = int(h * height_ratio)
            font = cv2.FONT_HERSHEY_PLAIN
            cnf = str(round(confidence_score[i] , 2))
            text = str(classes_names[ids[i]])+'-'+cnf
            cv2.rectangle(image , (x,y) , (x+w,y+h) , (0,255,0),2,cv2.LINE_AA)
            cv2.putText(image,text , (x ,y-2) , font,1,(255,255,255),1,cv2.LINE_AA)
        elif classes_names[ids[i]] == 'person':
            count2+=1
            c.append(count2)
            x , y , w , h = cordinates[i]
            x = int(x * width_ratio)
            y = int(y * height_ratio)
            w = int(w * width_ratio)
            h = int(h * height_ratio)
            font = cv2.FONT_HERSHEY_PLAIN
            cnf = str(round(confidence_score[i] , 2))
            text = str(classes_names[ids[i]])+'-'+cnf
            cv2.rectangle(image , (x,y) , (x+w,y+h) , (0,255,0),2,cv2.LINE_AA)
            cv2.putText(image,text , (x ,y-2) , font,1,(255,255,255),1,cv2.LINE_AA)
            
    if len(k) > 0:
        font = cv2.FONT_HERSHEY_PLAIN
        text1 = 'Total Bikes : {}'.format(k[-1])
        cv2.putText(image , text1 , (10,25) , font,1,(255,255,0),1,cv2.LINE_4) 
    else:
        font = cv2.FONT_HERSHEY_PLAIN
        k.append(0)
        text1 = 'Total Bikes : {}'.format(k[-1])
        cv2.putText(image , text1 , (10,25) , font,1,(255,255,0),1,cv2.LINE_4) 
        
    if len(c) > 0:
        font = cv2.FONT_HERSHEY_PLAIN
        text1 = 'Total Persons : {}'.format(c[-1])
        cv2.putText(image , text1 , (200,25) , font,1,(0,0,255),1,cv2.LINE_4) 
    else:
        c.append(0)
        font = cv2.FONT_HERSHEY_PLAIN
        text1 = 'Total Persons : {}'.format(c[-1])
        cv2.putText(image , text1 , (200,25) , font,1,(0,0,255),1,cv2.LINE_4) 

def bounding_box(detections):
    confidence_score = []
    ids = []
    cordinates = []
    
    
    for i in detections:
        for j in i:
            probs_values = j[5:]
            class_ = np.argmax(probs_values)
            confidence_ = probs_values[class_]
            
            if confidence_ > Threshold:
                w , h = int(j[2] * image_size) , int(j[3] * image_size)
                x , y = int(j[0] * image_size - w / 2) , int(j[1] * image_size - h / 2)
                cordinates.append([x,y,w,h])
                ids.append(class_)
                confidence_score.append(confidence_)
    final_box = cv2.dnn.NMSBoxes(cordinates , confidence_score , Threshold , .6)
    return final_box , cordinates , confidence_score , ids


image = cv2.imread('./testing images/test_2.jpg')
#cv2.imshow('image',image)
#cv2.waitKey()
#cv2.destroyAllWindows()
original_width , original_height = image.shape[1] , image.shape[0]

Neural_Network = cv2.dnn.readNetFromDarknet('./Files/yolov4.cfg','./Files/yolov4.weights')
classes_names = []
k = open('./Files/class_names','r')
for i in k.readlines():
    classes_names.append(i.strip())
#print(classes_names)
blob = cv2.dnn.blobFromImage(image , 1/255 , (320,320) , True , crop = False)
#print(blob.shape)
Neural_Network.setInput(blob)
cfg_data = Neural_Network.getLayerNames()
#print(cfg_data)
layer_names = Neural_Network.getUnconnectedOutLayers()
outputs = [cfg_data[i-1] for i in layer_names]
#print(outputs)
output_data = Neural_Network.forward(outputs)

                
        
    
final_box , cordinates , confidence_score , ids = bounding_box(output_data)   
predictions(final_box , cordinates , confidence_score , ids ,original_width / 320,original_height / 320 )    
    
cv2.imshow('img',image)
cv2.waitKey()
cv2.destroyAllWindows()