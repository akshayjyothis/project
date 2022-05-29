import face_recognition
import cv2
import numpy as np
import subprocess




# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('Rose.mp4')

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video file")


#Device normal print
device_def="""USB:

    USB 3.0 Bus:

      Host Controller Driver: AppleUSBXHCISPTLP
      PCI Device ID: 0x9d2f 
      PCI Revision ID: 0x0021 
      PCI Vendor ID: 0x8086 

    USB 3.1 Bus:

      Host Controller Driver: AppleUSBXHCIAR
      PCI Device ID: 0x15d4 
      PCI Revision ID: 0x0002 
      PCI Vendor ID: 0x8086 
      Bus Number: 0x00 

"""




# This is a super simple (but slow) example of running face recognition on live video from your webcam.
# There's a second example that's a little more complicated but runs faster.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
anu_image = face_recognition.load_image_file("anu.jpg")
anu_face_encoding = face_recognition.face_encodings(anu_image)[0]

akshay_image = face_recognition.load_image_file("akshay.jpg")
akshay_face_encoding = face_recognition.face_encodings(akshay_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    anu_face_encoding,akshay_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Anu","Akshay"
]

#FACE DETECTION FLAG
face_det=0

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        


        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        #FACE DETECTION FLAG CHANGE
        if name=="Akshay":
            face_det=1
        else:
            face_det=0    

    ret, vid = cap.read()
    font = cv2.FONT_HERSHEY_SIMPLEX

    if ret == True:

        #DEVICE
        result = subprocess.run(["system_profiler", "SPUSBDataType"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        device_details=result.stdout
        device_text="NO EXTERNAL DEVICE CONNECTED"  
        if device_details==device_def:
            device_text="NO EXTERNAL DEVICE CONNECTED" 
            cv2.putText(vid,device_text, (50,80), font, 1, (0, 255,0), 2, cv2.LINE_4)
        else:
            device_text="EXTERNAL DEVICE DETECTED"
            cv2.putText(vid,device_text, (50,80), font, 1, (0,0,255), 2, cv2.LINE_4)     
        
        

        #Audio Device 
        result = subprocess.run(["system_profiler", "SPAudioDataType"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        audio_devices=result.stdout 
        target_device="realme Buds Air 2"
        if (audio_devices.find(target_device) == -1):
             cv2.putText(vid,'AUDIO DEVICE NOT CONNECTED', (50,110), font, 1, (0,0,255), 2, cv2.LINE_4)
        else:
             cv2.putText(vid,'AUDIO DEVICE CONNECTED', (50,110), font, 1, (0, 255,0), 2, cv2.LINE_4)


        position = ((int) (vid.shape[1]/2 - 268/2), (int) (vid.shape[0]/2 - 36/2))
        # Display the resulting frame
        if face_det==1:
             cv2.putText(vid, 'FACE DETECTED', (50, 50), font, 1, (0, 255,0), 2, cv2.LINE_4)
        else:
             cv2.putText(vid, 'FACE NOT DETECTED', (50, 50), font, 1, (0, 0,255), 2, cv2.LINE_4)
        cv2.imshow('OTT', vid)
    cv2.putText(vid, 'No external device detected', (50, 50), font, 1, (255, 0,0), 2, cv2.LINE_4)
    # Display the resulting image
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera", 400, 400)
    cv2.imshow("Camera", frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
