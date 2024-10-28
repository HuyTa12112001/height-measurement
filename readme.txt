# Pose Landmarker Script

This script processes images to detect pose landmarks using the MediaPipe library and calculates distances and estimated heights based on the detected landmarks.

## Prerequisites

Before running the script, ensure you have the following dependencies installed:

- Python 3.x
- OpenCV
- MediaPipe
- NumPy
- SciPy
- imutils
- Pillow
- YOLO (Ultralytics)

You can install the necessary Python packages using pip:

pip install opencv-python mediapipe numpy scipy imutils pillow ultralytics

## Directory Structure

Ensure your project directory has the following structure:

project_directory/
|-- pose_landmarker_heavy.task
|-- best.pt
|-- data_folder/
| |-- image1.jpg
| |-- image2.jpg
| ...
|-- outputx/
|-- script.py
|-- readme.txt

- `pose_landmarker_heavy.task`: MediaPipe Pose Landmarker model file.
- `best.pt`: YOLO model file.
- `data_folder/`: Folder containing input images (JPEG format).
- `outputx/`: Folder where the output images will be saved.
- `script.py`: The Python script provided above.
- `readme.txt`: This readme file.

## Running the Script

1. Place the input images in the `data_folder/`.
2. Ensure the `pose_landmarker_heavy.task` and `best.pt` files are in the project directory.
3. Run the script using the following command:

python script.py

## Script Explanation

1. **Importing Libraries**: The necessary libraries are imported.
2. **Mapping Landmark Indices to Body Parts**: A dictionary maps the landmark indices to corresponding body parts.
3. **Writing Pose Landmarks to CSV**: The `write_pose_landmarks_to_csv` function writes pose landmarks to a CSV file.
4. **Calculating Distances and Heights**: Functions to calculate distances, midpoints, and heights based on the landmarks.
5. **Drawing Landmarks on Images**: The `draw_landmarks_on_image` function visualizes pose landmarks on images.
6. **Processing Images**: The script loops through each image in the `data_folder`, processes it to detect pose landmarks, and calculates distances and heights.
7. **Saving Results**: The annotated images and distances are saved to the specified output folder.

## Output

- The processed images with annotated landmarks and estimated height will be saved in the `outputx/` folder.
- A CSV file containing pose landmarks and distances will be generated.

## Notes

- Ensure the `pose_landmarker_heavy.task` and `best.pt` model files are compatible with the MediaPipe and YOLO versions you are using.
- Adjust the distance in centimeters (`dist_in_cm`) as needed for accurate height estimation.

## Troubleshooting

- If you encounter any errors related to missing dependencies, ensure all required packages are installed.
- Verify the paths to the model files and input images are correct.