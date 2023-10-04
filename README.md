# Sky Detection and Segmentation

This program performs sky detection in a dataset of images and evaluates the accuracy of the detected sky regions. It processes images to identify sky pixels, generate predicted masks, and evaluate the accuracy of the predictions using ground truth masks. The program also visualizes the results by saving the processed images, predicted sky masks, predicted sky regions, and skyline images.

## Prerequisites

- Python 3.9
- OpenCV (cv2) library
- NumPy library

## Getting Started

1. Clone the repository or download the source code.
2. Place your dataset of images in a folder named dataset.
3. Place the corresponding ground truth masks in a folder named mask. The mask images should have the same names as the image groups in the dataset folder.
4. Run the sky_detection_algorithm.py script.
5. The program will process each image group, detect sky regions, evaluate accuracy, and visualize the results in separate folders.

## Folder Structure

- 'dataset/': Folder containing subfolders for each image group.
- 'mask/': Ground truth mask images corresponding to each image group.
- 'predicted_mask/': Predicted mask images for each image group.
- 'predicted_sky/': Predicted sky region images for each image group.
- 'skyline/': Skyline images for each image group.
- 'Computer Vision Project Part II Report': Report of the algorithm.
- 'console_output.txt': Text file containing console output during program execution.
- 'sky_detection_algorithm.py': Main script to execute the program.