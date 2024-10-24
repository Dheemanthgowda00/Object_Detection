import cv2
import pyttsx3
import threading

cap= cv2.VideoCapture(1)
cap.set(3,640)
cap.set(4,480)


classNames= []
classFile= r'D:\\coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = r'D:\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = r'D:\\frozen_inference_graph.pb'


net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

# Initialize the text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1)  # Volume level

# Function to handle text-to-speech asynchronously
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Initial speech when the program starts
initial_speech_thread = threading.Thread(target=speak, args=("Hello everyone, welcome to Robomanthan",))
initial_speech_thread.start()

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=0.55)
    print(classIds,bbox)
    if len(classIds)!=0:
        for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=3)
            className = classNames[classId - 1].upper()  # Get the object name
            cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            cv2.putText(img,str(round(confidence*100,2)) , (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # Start a new thread for speaking without blocking
            speech_thread = threading.Thread(target=speak, args=(f"Detected {className}",))
            speech_thread.start()

    cv2.imshow("Output:", img)
    cv2.waitKey(1)
    # engine = pyttsx3.init()
    # engine.say(img)
    # engine.runAndWait()
