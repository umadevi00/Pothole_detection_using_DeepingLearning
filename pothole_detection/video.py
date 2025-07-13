import cv2
import base64
from inference_sdk import InferenceHTTPClient

# Initialize Roboflow Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="TE5iDtuSU8YSdfPmmDSX"
)

# Load video
video_path = r"C:\Users\umadevi\OneDrive\Desktop\Miniproject\pothole-detection\test_video.mp4"
cap = cv2.VideoCapture(video_path)

# Video writer (optional if you want to save output)
save_output = True
output_path = "pothole_output.avi"
if save_output:
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Ensure the video is opened successfully
if not cap.isOpened():
    print("❌ Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ End of video reached or error while reading.")
        break

    # Resize if needed (Roboflow may have image size limits)
    resized_frame = cv2.resize(frame, (640, 480))

    # Encode to base64
    _, buffer = cv2.imencode('.jpg', resized_frame)
    encoded_image = base64.b64encode(buffer).decode("utf-8")

    try:
        result = CLIENT.infer(encoded_image, model_id="pothole_detection-vuxge/1")
        predictions = result.get("predictions", [])

        # Draw boxes
        for pred in predictions:
            x, y = int(pred['x']), int(pred['y'])
            w, h = int(pred['width']), int(pred['height'])
            confidence = pred['confidence']
            class_name = pred['class']

            x1, y1 = x - w // 2, y - h // 2
            x2, y2 = x + w // 2, y + h // 2

            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{class_name}: {confidence*100:.1f}%"
            cv2.putText(resized_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    except Exception as e:
        print(f"❌ Inference error: {e}")

    # Show frame
    cv2.imshow("Pothole Detection", resized_frame)

    if save_output:
        out.write(resized_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit the video loop
        break

cap.release()  # Release the video capture object
if save_output:
    out.release()  # Release the video writer if saving output
cv2.destroyAllWindows()  # Close all OpenCV windows
