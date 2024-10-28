# Import the necessary modules.
import os
import csv
import math
import datetime
import imutils
import cv2 as cv2
import numpy as np
import mediapipe as mp
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import numpy as np
import cv2

# Mapping landmark indices to body parts
BODY_PARTS = {
    0: 'Nose',
    1: 'Left Eye Inner',
    2: 'Left Eye',
    3: 'Left Eye Outer',
    4: 'Right Eye Inner',
    5: 'Right Eye',
    6: 'Right Eye Outer',
    7: 'Left Ear',
    8: 'Right Ear',
    9: 'Mouth Left',
    10: 'Mouth Right',
    11: 'Left Shoulder',
    12: 'Right Shoulder',
    13: 'Left Elbow',
    14: 'Right Elbow',
    15: 'Left Wrist',
    16: 'Right Wrist',
    17: 'Left Pinky',
    18: 'Right Pinky',
    19: 'Left Index',
    20: 'Right Index',
    21: 'Left Thumb',
    22: 'Right Thumb',
    23: 'Left Hip',
    24: 'Right Hip',
    25: 'Left Knee',
    26: 'Right Knee',
    27: 'Left Ankle',
    28: 'Right Ankle',
    29: 'Left Heel',
    30: 'Right Heel',
    31: 'Left Foot Index',
    32: 'Right Foot Index'
}

def write_pose_landmarks_to_csv(pose_landmarks_list, csv_filename):
    with open(csv_filename, mode='a', newline='') as csvfile:  # Open the file in append mode
        fieldnames = ['Pose', 'X', 'Y', 'Z']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # If the file is empty, write the header
        if csvfile.tell() == 0:
            writer.writeheader()

        # Write pose landmarks to the file
        for idx, pose_landmarks in enumerate(pose_landmarks_list):
            for i, landmark in enumerate(pose_landmarks):
                body_part = BODY_PARTS.get(i, 'Unknown')
                writer.writerow({'Pose': f'Pose {idx + 1} - {body_part}',
                                 'X': landmark.x,
                                 'Y': landmark.y,
                                 'Z': landmark.z})
            # Insert an empty row between poses
            writer.writerow({})  # Empty row

def calculate_distance(point1, point2):
    # Calculate Euclidean distance between two points
    return np.linalg.norm(np.array(point1) - np.array(point2))

def calculate_midpoint(point1, point2):
    # Calculate midpoint between two points
    return ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)

def calculate_perpendicular_distance(point, line_start, line_end):
    # Check if the line is vertical (x coordinates are the same)
    if line_end[0] == line_start[0]:
        # If the line is vertical, the perpendicular distance is simply the absolute difference in x coordinates
        return abs(point[0] - line_start[0])
    else:
        # Calculate the equation of the line and then the perpendicular distance
        line_slope = (line_end[1] - line_start[1]) / (line_end[0] - line_start[0])
        line_intercept = line_start[1] - line_slope * line_start[0]
        perpendicular_distance = abs(line_slope * point[0] - point[1] + line_intercept) / (line_slope ** 2 + 1) ** 0.5
        return perpendicular_distance

def calculate_height(landmarks, image_width, image_height):
    left_ankle = landmarks[27]
    left_heel = landmarks[29]
    left_foot_index = landmarks[31]
    left_knee = landmarks[25]
    right_hip = landmarks[24]
    left_hip = landmarks[23]
    right_shoulder = landmarks[12]
    left_shoulder = landmarks[11]
    right_mouth = landmarks[10]
    left_mouth = landmarks[9]
    nose = landmarks[0]

    # Convert normalized landmark coordinates to pixel coordinates
    pixel_left_ankle = (int(left_ankle.x * image_width), int(left_ankle.y * image_height))
    pixel_left_heel = (int(left_heel.x * image_width), int(left_heel.y * image_height))
    pixel_left_foot_index = (int(left_foot_index.x * image_width), int(left_foot_index.y * image_height))
    pixel_left_knee = (int(left_knee.x * image_width), int(left_knee.y * image_height))
    pixel_right_hip = (int(right_hip.x * image_width), int(right_hip.y * image_height))
    pixel_left_hip = (int(left_hip.x * image_width), int(left_hip.y * image_height))
    pixel_right_shoulder = (int(right_shoulder.x * image_width), int(right_shoulder.y * image_height))
    pixel_left_shoulder = (int(left_shoulder.x * image_width), int(left_shoulder.y * image_height))
    pixel_right_mouth = (int(right_mouth.x * image_width), int(right_mouth.y * image_height))
    pixel_left_mouth = (int(left_mouth.x * image_width), int(left_mouth.y * image_height))
    pixel_nose = (int(nose.x * image_width), int(nose.y * image_height))

    # Calculate distances using pixel coordinates
    distance_ankle_heel_foot_index =  calculate_perpendicular_distance(pixel_left_ankle, pixel_left_heel,
                                                                             pixel_left_foot_index)
    distance_knee_ankle = calculate_distance(pixel_left_knee, pixel_left_ankle)
    distance_hip_knee = calculate_distance(pixel_left_hip, pixel_left_knee)
    distance_midpoint_shoulder_hip = calculate_distance(calculate_midpoint(pixel_left_shoulder, pixel_right_shoulder),
                                                        calculate_midpoint(pixel_left_hip, pixel_right_hip))
    # distance_midpoint_shoulder_nose = calculate_distance(calculate_midpoint(pixel_left_shoulder, pixel_right_shoulder), pixel_nose)
    distance_midpoint_mouth_shoulder = calculate_distance(calculate_midpoint(pixel_left_mouth, pixel_right_mouth),
                                                          calculate_midpoint(pixel_left_shoulder, pixel_right_shoulder))
    distance_nose_mouth = calculate_perpendicular_distance(pixel_nose, pixel_left_mouth, pixel_right_mouth)
    distance_nose_top_of_head = 3.236 * distance_nose_mouth  # 0.5  # Assuming 0.5 as the ratio for simplicity, adjust as needed
    # Sum up to calculate the height
    heightf = (
            distance_ankle_heel_foot_index +
            distance_knee_ankle +
            distance_hip_knee +
            distance_midpoint_shoulder_hip +
            distance_midpoint_mouth_shoulder +
            distance_nose_mouth +
            distance_nose_top_of_head
    )

    # Store distances in a list
    distances = [
        distance_ankle_heel_foot_index,
        distance_knee_ankle,
        distance_hip_knee,
        distance_midpoint_shoulder_hip,
        distance_midpoint_mouth_shoulder,
        distance_nose_mouth,
        distance_nose_top_of_head
    ]

    return heightf, distances

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)
  image_height, image_width, _ = annotated_image.shape

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())

     # Print the coordinates of each landmark
    """print(f"Pose {idx + 1} Landmarks:")"""
    for i, landmark in enumerate(pose_landmarks):
      body_part = BODY_PARTS.get(i, 'Unknown')

      # Convert normalized coordinates to pixel coordinates
      pixel_x = int(landmark.x * image_width)
      pixel_y = int(landmark.y * image_height)

     

    heightf, distances = calculate_height(pose_landmarks,image_width, image_height)
    """print(f"Estimated Height: {height:.2f} pixels")"""
    global height_in_cm
    height_in_cm = heightf/pixel_per_cm


  return pose_landmarks_list, annotated_image, distances

# Path to the folder containing images
data_folder = "pose_3"

# Get a list of all image files in the data folder
image_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.jpg')]
# Read the image
# Loop over each image file
for img_path in image_files:
    print("Processing image:", img_path)

    # Load the image
    image = cv2.imread(img_path)
    dist_in_cm = 30.5
    from PIL import Image
    from ultralytics import YOLO

    model = YOLO('best.pt')
    results = model(img_path)

    # Extract bounding box dimensions
    boxes = results[0].boxes.xywh.cpu()
    for box in boxes:
        x, y, w, h = box
        
    # Visualize the results
    for i, r in enumerate(results):
        r.save(filename=f'results{i}.jpg')
    global pixel_per_cm
    pixel_per_cm = h.item() / dist_in_cm
    print(pixel_per_cm)

    model_path = 'pose_landmarker_heavy.task'

    # Create an PoseLandmarker object.
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)

    img_path_temp = "results0.jpg"
    # Load the input image.
    image = mp.Image.create_from_file(img_path_temp)

    # Detect pose landmarks from the input image.
    detection_result = detector.detect(image)

    # Process the detection result. In this case, visualize it.
    pose_landmarks_list, annotated_image, distances = draw_landmarks_on_image(image.numpy_view(), detection_result)

    import csv

    # Initialize a list to store distances for each pose
    all_distances = []

    # Store distances for each pose
    all_distances.append(distances)

    # Define CSV filename for distances
    csv_distances_filename = "2D_pose3_test/2D_pose3.csv"

    # Write distances to a CSV file
    # Write distances to a CSV file
    with open(csv_distances_filename, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if csvfile.tell() == 0:
            # Write header if file is empty
            writer.writerow(
                ['Pose'] + [f'Distance{i + 1} (cm)' for i in range(len(distances))] + ['Height (cm)'])
        # Write distances for each pose
        for idx, pose_distances in enumerate(all_distances):
            # Convert distances from pixels to centimeters
            distances_cm = [distance / pixel_per_cm for distance in pose_distances]
            # Calculate height in centimeters
            height_cm = sum(distances_cm)
            height_in_cm = height_cm
            writer.writerow([f'Pose {idx + 1}'] + distances_cm + [height_cm])

    # Write pose landmarks to CSV
    csv_filename = "2D_pose3_test/pose_landmarks4_rotate_raw.csv"
    write_pose_landmarks_to_csv(pose_landmarks_list, csv_filename)
    #print("Pose landmarks saved to:", csv_filename)

    # Calculate the position for the text
    text = "Estimated Height: {:.2f} cm".format(height_in_cm)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = (annotated_image.shape[1] - text_size[0]) // 2
    text_y = 100  # Distance from the top edge

    # Draw the text on the image
    cv2.putText(annotated_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
    # Define the path where you want to save the final image
    # Generate a unique filename based on current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = f"2D_pose3_test/ouput3/final_image_{timestamp}.jpg"

    # Save the final annotated image
    cv2.imwrite(output_path, annotated_image)

    # Resize the annotated image to fit within the screen resolution
    screen_res = 1280, 720  # Example screen resolution (width, height)
    scale_percent = min(screen_res[0] / annotated_image.shape[1], screen_res[1] / annotated_image.shape[0]) * 100
    width = int(annotated_image.shape[1] * scale_percent / 100)
    height = int(annotated_image.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(annotated_image, (width, height))

    # Display the resized image
    cv2.imshow("Final Result", resized_image)
    cv2.waitKey(10)
    cv2.destroyAllWindows()









