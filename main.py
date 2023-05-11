import cv2

# Load the video file
cap = cv2.VideoCapture("1.mp4")

# Initialize the first frame
ret, frame1 = cap.read()
frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
frame1_gray = cv2.GaussianBlur(frame1_gray, (21, 21), 0)

# Initialize the number of moment
moment_detected = 0

while True:
    # Read the next frame
    ret, frame2 = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale and apply blur
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.GaussianBlur(frame2_gray, (21, 21), 0)

    # Calculate the difference between the two frames
    frame_diff = cv2.absdiff(frame1_gray, frame2_gray)

    # Apply thresholding to the difference frame
    thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]

    # Dilate the thresholded frame to fill in holes
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find the contours in the thresholded frame
    contours, _ = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours
    for contour in contours:
        # Ignore small contours
        if cv2.contourArea(contour) < 1000:
            continue

        # Draw a rectangle around the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Increment the moment detected
        moment_detected += 1

    # Update the previous frame
    frame1_gray = frame2_gray

    # Display the frame with the bounding boxes and the moment detected
    cv2.putText(frame2, f"moment detected: {moment_detected}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("moment detected", frame2)
    key = cv2.waitKey(1)

    # Break the loop if the 'q' key is pressed
    if key == ord('q'):
        break

# Release the video file and close all windows
cap.release()
cv2.destroyAllWindows()
