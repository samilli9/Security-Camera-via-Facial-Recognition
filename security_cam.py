###########
# Imports #
###########
import cv2
import time
import datetime




#########################################
# Access your security camera and video #
#########################################

# Declare how many video devices are present as an index, I am only using one (my laptop webcam)
cap = cv2.VideoCapture(0)

# Face detection classifier (using Haar cascades for its speed)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

# Initialize necessary variables
detection = False
detection_stopped_time = None
timer_started = False
SECONDS_TO_RECORD_AFTER_DETECTION = 5

# Establish the video capture's frame size
frame_size = (int(cap.get(3)), int(cap.get(4)))

# Output the video as an mp4
fourcc = cv2.VideoWriter_fourcc(*"mp4v")




########################################
# Begin logic for the security camera! #
########################################
while True:
	# Read frame by frame from the video capture device
    _, frame = cap.read()

    # For classification of faces/bodies the frame must be greyscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Returns a list of faces/bodies that exist
    # 1.3 is the scale factor that determines speed and accuracy of the algorithm
    # 5 is the minimum number of neighbors (Establishing that a face/body should be detected 5 times to classify it as a face. The more neighbors the more accuracy)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    bodies = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw a green box where the algorithm locates a face/body
    for (x,y,width,height) in faces:
        cv2.rectangle(frame, (x,y), (x + width, y + height), (0,255,0), 3)
        

        ###############################################################################
        # Logic for recording a video clip when a subject appears in the video frame! #
        # If something appears in the video frame then start recording.      		  #
        # Keep recording UNLESS the subject is gone for more than 5 seconds. 		  #
        # Store the video with a timestamp.                                 		  #
        ###############################################################################
        if len(faces) + len(bodies) > 0:
            if detection:
                timer_started = False 
            else:
                detection = True
                current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
                out = cv2.VideoWriter(f"{current_time}.mp4", fourcc, 20, frame_size)
                print("Started Recording!")
        elif detection:
            if timer_started:
                if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                    detection = False
                    timer_started = False
                    out.release()
                    print('Stop Recording!')
            else:
                timer_started = True
                detection_stopped_time = time.time()

    if detection:
        out.write(frame)

    cv2.imshow("Camera", frame)

    # Exit the video camera by hitting the q key
    if cv2.waitKey(1) == ord('q'):
        break



###################
# Clear resources #
###################
out.release()
cap.release()
cv2.destroyAllWindows()