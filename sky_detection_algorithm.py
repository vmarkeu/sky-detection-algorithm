# Computer Vision Project Part II
# Sky Detection and Segmentation Algorithm
# Vivien Mak Wing Yan 20012134

import cv2
import numpy as np
import os
import shutil
import time
from scipy.signal import medfilt
import sys

class ConsoleToFile:
    def __init__(self, filename, show_stdout=True):
        self.original_stdout = sys.stdout
        self.log_file = open(filename, "w")
        self.show_stdout = show_stdout

    def redirect(self):
        sys.stdout = self

    def restore(self):
        sys.stdout = self.original_stdout
        self.log_file.close()

    def write(self, text):
        if self.show_stdout:
            self.original_stdout.write(text)
        self.log_file.write(text)
        self.log_file.flush()  # Flush to ensure immediate writing to file

    def flush(self):
        if self.show_stdout:
            self.original_stdout.flush()

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    return image

def detect_sky(image_filenames, group_folder, threshold=0.4):
    avg_sky_percentage = 0
    
    for image_filename in image_filenames:
        image_path = os.path.join(group_folder, image_filename)
        image = preprocess_image(image_path)

        # Convert the image to the HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the ranges of colors corresponding to the sky (blue, white, and gray tones)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([140, 255, 255])
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([255, 50, 255])
        lower_gray = np.array([0, 0, 100])
        upper_gray = np.array([255, 50, 200])

        # Create masks using the color ranges
        mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
        mask_white = cv2.inRange(hsv_image, lower_white, upper_white)
        mask_gray = cv2.inRange(hsv_image, lower_gray, upper_gray)

        # Combine the masks using logical OR
        combined_mask = mask_blue | mask_white | mask_gray

        # Calculate the percentage of sky pixels in the image
        total_pixels = combined_mask.shape[0] * combined_mask.shape[1]
        sky_pixels = np.count_nonzero(combined_mask)
        sky_percentage = sky_pixels / total_pixels

        avg_sky_percentage += sky_percentage
    
    # Calculate the average sky percentage
    avg_sky_percentage /= len(image_filenames)  
    # print("Average Sky Percentage:", avg_sky_percentage)
    
    # Check if average sky percentage is above the threshold
    if avg_sky_percentage >= threshold:
        return True
    else:
        return False

def save_output(image_filename, folder_name, group_folder, output_image, image_type):
    # Save the image in the folder for the respective image group
    filename = image_filename.replace('.jpg', f'_{folder_name}_{image_type}.jpg')
    path = os.path.join(group_folder, filename)
    cv2.imwrite(path, output_image)

def is_daytime(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate the average pixel intensity of the grayscale image
    avg_intensity = np.mean(gray_image)
    # Set a threshold value to differentiate between daytime and nighttime
    threshold = 100
    # If the average intensity is above the threshold -> daytime
    is_daytime = avg_intensity > threshold
    return is_daytime

def remove_skyline_gaps(mask):
    # Get the height and width of the mask
    h, w = mask.shape  
    # Loop through each column in the mask
    for i in range(w):
        # Extract the column at index i
        raw = mask[:, i]  
        # Apply median filtering to the column to reduce noise
        after_median = medfilt(raw, 5)
        try:
            # Find the index of the first zero value (non-sky pixel)
            first_zero_index = np.where(after_median == 0)[0][0]
            # Find the index of the first one value (sky pixel)
            first_one_index = np.where(after_median == 1)[0][0]
            
            # Check if the first zero index is far enough from the top (skyline)
            if first_zero_index > 5:
                # Set the region between the first one index and the first zero index to sky pixels
                mask[first_one_index:first_zero_index, i] = 1
                # Set the region after the first zero index to non-sky pixels
                mask[first_zero_index:, i] = 0
                # Set the region before the first one index to non-sky pixels
                mask[:first_one_index, i] = 0
        except:
            # Continue to the next column if an exception occurs (no zero or one values found)
            continue
    return mask  

def detect_sky_region(img):
    # Get the height and width of the image
    h, w, _ = img.shape  
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Reduce noise
    img_gray = cv2.blur(img_gray, (5, 5))
    img_gray = cv2.medianBlur(img_gray, 3)
    
    # Apply Laplacian edge detection to highlight edges in the image
    lap = cv2.Laplacian(img_gray, cv2.CV_8U)
    # Create a binary mask based on the Laplacian gradient
    gradient_mask = (lap < 8).astype(np.uint8)
    
    # Define a morphological structuring element for erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    # Perform erosion to remove small noise
    mask = cv2.morphologyEx(gradient_mask, cv2.MORPH_ERODE, kernel)
    # Remove gaps in the detected skyline
    mask = remove_skyline_gaps(mask)
    
    # Find connected components in the binary mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    
    # Find the index of the largest connected component (excluding the background)
    largest_component_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    
    # Create a new mask containing only the largest connected component (sky)
    sky_mask = np.zeros_like(labels, dtype=np.uint8)
    sky_mask[labels == largest_component_idx] = 255
    
    # Apply morphological operations to refine the mask
    kernel_fill = np.ones((15, 15), np.uint8)
    # Closing to fill small holes
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel_fill) 
    # Opening to remove small noise
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel_fill)
    # Dilation to expand regions
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_DILATE, kernel_fill)
    
    # Apply the final mask to the original image
    sky_img = cv2.bitwise_and(img, img, mask=sky_mask)
    return sky_mask, sky_img

def post_processing(predicted_mask):
    kernel = np.ones((75, 75), np.uint8) * 255
    # Invert the predicted sky mask to work with non-sky regions
    inverted_mask = cv2.bitwise_not(predicted_mask)
    # Perform closing to fill small gaps in non-sky regions
    closed_mask = cv2.morphologyEx(inverted_mask, cv2.MORPH_CLOSE, kernel)
    # Invert the closed mask back to the original orientation
    postprocessed_sky = cv2.bitwise_not(closed_mask)
    return postprocessed_sky

def evaluate_performance(predicted_mask, ground_truth, image_filename, image_group, image_type):
    # Confusion matrix
    true_positives = np.sum(np.logical_and(predicted_mask, ground_truth))
    true_negatives = np.sum(np.logical_and(np.logical_not(predicted_mask), np.logical_not(ground_truth)))
    false_positives = np.sum(np.logical_and(predicted_mask, np.logical_not(ground_truth)))
    false_negatives = np.sum(np.logical_and(np.logical_not(predicted_mask), ground_truth))
    accuracy = ((true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)) * 100 
    print(f"Accuracy for {image_type} image {image_filename} in group {image_group}: {accuracy:.2f}%")
    return accuracy

def find_skyline(predicted_mask, mask):
    # Find contours in the binary image
    contours, _ = cv2.findContours(predicted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create empty image to draw the skyline
    skyline_img = np.zeros_like(predicted_mask)
    # Apply the mask to the skyline image
    skyline_img = skyline_img * mask
    # Draw the contours on the skyline image
    cv2.drawContours(skyline_img, contours, -1, (255, 255, 255), 2)
    return skyline_img

def process_daytime_images(image_group, image_filenames, group_folder, predicted_mask_group_folder, predicted_sky_group_folder, mask, skyline_group_folder):
    # List to store accuracy values for the images in the current group
    image_accuracies = []
    best_accuracy = 0.0
    best_mask_path = None
    image_type = 'daytime'

    for image_filename in image_filenames:
        image_path = os.path.join(group_folder, image_filename)
        image = preprocess_image(image_path)

        if is_daytime(image):
            # Detect the sky region in the image.
            predicted_mask, predicted_sky = detect_sky_region(image)
            predicted_mask = post_processing(predicted_mask)

            # Save the processed image
            predicted_mask_filename = image_filename.replace('.jpg', f'_predicted_mask_{image_type}.jpg')
            predicted_mask_path = os.path.join(predicted_mask_group_folder, predicted_mask_filename)
            cv2.imwrite(predicted_mask_path, predicted_mask)
            save_output(image_filename, 'predicted_sky', predicted_sky_group_folder, predicted_sky, image_type)

            # Extract the skyline coordinates from the postprocessed sky mask
            skyline = find_skyline(predicted_mask, mask)
            # Save the skyline coordinates as an image in the 'skyline' folder
            save_output(image_filename, 'skyline', skyline_group_folder, skyline, image_type)

            # Evaluate the accuracy of the output images using the mask
            accuracy = evaluate_performance(predicted_mask, mask, image_filename, image_group, image_type)
            image_accuracies.append(accuracy)

            # Check if the current mask has the best accuracy so far
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_mask_path = predicted_mask_path
    return image_accuracies, best_mask_path

def process_nighttime_images(image_group, image_filenames, group_folder, predicted_mask_group_folder, predicted_sky_group_folder, mask, best_mask_dict, skyline_group_folder):
    # List to store accuracy values for the images in the current group
    image_accuracies = []
    image_type = 'nighttime'

    for image_filename in image_filenames:
        image_path = os.path.join(group_folder, image_filename)
        image = preprocess_image(image_path)

        if not is_daytime(image) and image_group in best_mask_dict:
            # Retrieve the best mask
            best_mask = best_mask_dict[image_group]
            best_mask_binary = cv2.cvtColor(best_mask, cv2.COLOR_BGR2GRAY)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Apply the best mask to the grayscale image using bitwise AND operation
            predicted_mask = cv2.bitwise_and(gray_image, gray_image, mask=best_mask_binary)
            # Set all non-zero values in the predicted_sky mask to 255 to obtain binary result
            predicted_mask[predicted_mask > 0] = 255
            # Find the predicted sky region
            predicted_sky = cv2.bitwise_and(image, image, mask=predicted_mask)

            # Save the processed image
            save_output(image_filename, 'predicted_mask', predicted_mask_group_folder, predicted_mask, image_type)
            save_output(image_filename, 'predicted_sky', predicted_sky_group_folder, predicted_sky, image_type)

            # Evaluate the accuracy of the output images using the mask
            accuracy = evaluate_performance(predicted_mask, mask, image_filename, image_group, image_type)
            image_accuracies.append(accuracy)

            # Extract the skyline coordinates from the postprocessed sky mask
            skyline = find_skyline(predicted_mask, mask)
            # Save the skyline coordinates as an image in the 'skyline' folder
            save_output(image_filename, 'skyline', skyline_group_folder, skyline, image_type)
    return image_accuracies

def process_no_sky_images(image_group, image_filenames, group_folder, predicted_mask_group_folder, predicted_sky_group_folder, mask, skyline_group_folder):
    # List to store accuracy values for the images in the current group
    image_accuracies = []

    for image_filename in image_filenames:
        image_path = os.path.join(group_folder, image_filename)
        image = preprocess_image(image_path)
        
        is_day = is_daytime(image)
        image_type = 'no_sky_daytime' if is_day else 'no_sky_nighttime'
        
        # Create empty binary mask as predicted sky because there is no sky in the image
        predicted_mask = np.zeros((image.shape[:2]), dtype=np.uint8)
        # Find the predicted sky region
        predicted_sky = cv2.bitwise_and(image, image, mask=predicted_mask)
        
        # Save the processed image
        save_output(image_filename, 'predicted_mask', predicted_mask_group_folder, predicted_mask, image_type)
        save_output(image_filename, 'predicted_sky', predicted_sky_group_folder, predicted_sky, image_type)

        # Evaluate the accuracy of the output images using the mask
        accuracy = evaluate_performance(predicted_mask, mask, image_filename, image_group, image_type)
        image_accuracies.append(accuracy)
    
        # Extract the skyline coordinates from the postprocessed sky mask
        skyline = find_skyline(predicted_mask, mask)
        # Save the skyline coordinates as an image in the 'skyline' folder
        save_output(image_filename, 'skyline', skyline_group_folder, skyline, image_type)
    return image_accuracies

def main():
    # Folder name
    dataset_folder = 'dataset'
    mask_folder = 'mask'
    predicted_mask_folder = 'predicted_mask'
    predicted_sky_folder = 'predicted_sky'
    skyline_folder = 'skyline'
    
    # Delete the folder if it exists
    if os.path.exists(predicted_mask_folder):
        shutil.rmtree(predicted_mask_folder)
    if os.path.exists(predicted_sky_folder):
        shutil.rmtree(predicted_sky_folder)
    if os.path.exists(skyline_folder):
        shutil.rmtree(skyline_folder)
    # Create the folder if it doesn't exist
    if not os.path.exists(predicted_mask_folder):
        os.makedirs(predicted_mask_folder)
    if not os.path.exists(predicted_sky_folder):
        os.makedirs(predicted_sky_folder)
    if not os.path.exists(skyline_folder):
        os.makedirs(skyline_folder)

    image_groups = [f for f in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, f))]

    # Lists to store accuracy values
    group_accuracies = []
    image_accuracies = []
    # Dictionary to store the best accuracy image mask for each group
    best_mask_dict = {}

    for image_group in image_groups:
        print(f"Currently processing image group {image_group}...")
        group_folder = os.path.join(dataset_folder, image_group)
        predicted_mask_group_folder = os.path.join(predicted_mask_folder, image_group)
        predicted_sky_group_folder = os.path.join(predicted_sky_folder, image_group)
        skyline_group_folder = os.path.join(skyline_folder, image_group)

        # Create the subfolder for the image group if it doesn't exist
        if not os.path.exists(predicted_mask_group_folder):
            os.makedirs(predicted_mask_group_folder)
        if not os.path.exists(predicted_sky_group_folder):
            os.makedirs(predicted_sky_group_folder)
        if not os.path.exists(skyline_group_folder):
            os.makedirs(skyline_group_folder)
        
        # Load the corresponding mask image
        mask_filename = f'{image_group}.png'
        mask_path = os.path.join(mask_folder, mask_filename)
        # Skip evaluation if the mask image is missing
        if not os.path.exists(mask_path):
            print(f"Warning: Mask image {mask_path} not found. Skipping evaluation for image group {image_group}.")
            continue
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Read the images in the image group folder
        image_filenames = [f for f in os.listdir(group_folder) if f.endswith('.jpg')]

        sky_exists = detect_sky(image_filenames, group_folder)
        
        if sky_exists:
            print(f"Sky detected in image group {image_group}.")
            # Process daytime images
            image_accuracies, best_mask_path = process_daytime_images(image_group, image_filenames, group_folder, predicted_mask_group_folder, predicted_sky_group_folder, mask, skyline_group_folder)
    
            # Determine the best mask for the group
            if best_mask_path:
                best_mask = preprocess_image(best_mask_path)
                best_mask_dict[image_group] = best_mask
    
            # Process nighttime images using the best mask
            image_accuracies += process_nighttime_images(image_group, image_filenames, group_folder, predicted_mask_group_folder, predicted_sky_group_folder, mask, best_mask_dict, skyline_group_folder)
        else:
            print(f"No sky detected in image group {image_group}.")
            # Process no sky images
            image_accuracies = process_no_sky_images(image_group, image_filenames, group_folder, predicted_mask_group_folder, predicted_sky_group_folder, mask, skyline_group_folder)

        # Calculate the average accuracy for the current image group
        average_accuracy_group = np.mean(image_accuracies)
        print(f"Average accuracy for image group {image_group}: {average_accuracy_group:.2f}%\n")
        group_accuracies.append(average_accuracy_group)

    print("PERFORMANCE METRICS")
    for item in zip(image_groups, group_accuracies):
        print(f"Average accuracy for {item[0]}: {item[1]:.2f}%")
    # Calculate and print the average accuracy for all image groups
    average_accuracy_all = np.mean(group_accuracies)
    print(f"Average accuracy for all image groups: {average_accuracy_all:.2f}%")

if __name__ == "__main__":
    start_time = time.time()
    print("Start program execution\n")
    output_filename = "console_output.txt"
    console_to_file = ConsoleToFile(output_filename, show_stdout=True)
    console_to_file.redirect()
    try:
        main()
    finally:
        end_time = time.time()
        # Calculate the execution time in seconds
        execution_time = end_time - start_time
        print(f"\nExecution time: {execution_time:.2f} seconds")
        console_to_file.restore()
        print("Execution completed.")