import cv2
import sys

def main():
    # 1. Load the pre-trained Haar Cascade face detection model from OpenCV's data pool
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        print("Error: Could not load the Haar Cascade XML classifier file.")
        sys.exit(1)

    # 2. Open a direct connection to your physical local webcam hardware (0 = Default)
    # Using CAP_DSHOW on Windows dramatically speeds up camera initialization
    if sys.platform.startswith('win'):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)

    # Performance optimization: Set lower frame capture resolution if stream lags
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open the local webcam hardware component.")
        print("Ensure no other application (Zoom, Teams, Colab browser) is using it.")
        sys.exit(1)

    print("Live tracking running! Click onto the video window and press 'q' to exit.")

    try:
        while True:
            # Capture the newest real-time video frame
            ret, frame = cap.read()
            
            # Guard clause to handle unexpected camera drops
            if not ret or frame is None:
                print("Warning: Dropped frame detected. Retrying...")
                continue
                
            # Convert to Grayscale (Haar Cascades evaluate faster on single-channel shapes)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Optimize image matrix scales to filter out background noise artifacts
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=6, 
                minSize=(40, 40)
            )
            
            # Render tracing indicators directly into the active matrix array
            for (x, y, w, h) in faces:
                # Blue bounding rectangle (BGR format: 255 Blue, 0 Green, 0 Red)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Text overlay tag pinned above the tracking zone
                cv2.putText(
                    frame, "Face Detected", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
                )
            
            # Render the updated graphics array within a dedicated UI window panel
            cv2.imshow("Real-Time Face Tracker", frame)
            
            # Check if the user pressed 'q' on their keyboard to close the execution
            # waitKey(1) introduces a mandatory 1ms delay for the UI frame to draw
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nProcess interrupted by console interface.")
        
    finally:
        # 3. Securely free up hardware registries to prevent device freezing bugs
        cap.release()
        cv2.destroyAllWindows()
        print("Camera system successfully closed and resources released.")

if __name__ == "__main__":
    main()
