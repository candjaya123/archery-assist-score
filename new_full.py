import os
import argparse
import cv2
import numpy as np
import sys
import glob
from tensorflow.lite.python.interpreter import Interpreter
import importlib.util

MODEL_NAME = 'model_lite'
GRAPH_NAME = 'detect.tflite'
LABELMAP_NAME = 'labelmap.txt'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

IM_NAME = ''
IM_DIR = 'test_images'

save_results = False
show_results = True

num_device = 1

min_conf_threshold = 0.7
prev_frame_time = 0
new_frame_time = 0

def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def load_hsv_ranges(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        min_hue_temp = int(lines[0].split('=')[1])
        max_hue_temp = int(lines[1].split('=')[1])
        min_saturation_temp = int(lines[2].split('=')[1])
        max_saturation_temp = int(lines[3].split('=')[1])
        min_value_temp = int(lines[4].split('=')[1])
        max_value_temp = int(lines[5].split('=')[1])
    return min_hue_temp, max_hue_temp, min_saturation_temp, max_saturation_temp, min_value_temp, max_value_temp

def detect_color(hsv_frame, min_hue, max_hue, min_saturation, max_saturation, min_value, max_value):
    lower_bound = np.array([min_hue, min_saturation, min_value])
    upper_bound = np.array([max_hue, max_saturation, max_value])
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    return mask

def detect_and_draw_contours(main_frame, mask, color, divider ,min_area=1000, min_radius=30, max_radius=350):
    count_circle = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:  # Filter for circular shapes with a minimum area
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            if min_radius <= radius <= max_radius:  # Check if radius is within the specified limits
                count_circle += 1
                # print(f'circle = {count_circle}')
                if(count_circle < 2):
                    cv2.circle(main_frame, center, radius, color, 3)
                    cv2.circle(main_frame, center, int(radius/divider), color, 3)
                    cv2.circle(main_frame, center, 2, color, 3)  # Draw the center of the circle
                    return center, radius, int(radius/divider)
    return 0,0,0

if __name__ == '__main__':
    # Load the label map
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the model
    interpreter = Interpreter(model_path=PATH_TO_CKPT)
    interpreter.allocate_tensors()

    # Define path to images and grab all image filenames
    if IM_DIR:
        PATH_TO_IMAGES = os.path.join(CWD_PATH,IM_DIR)
        images = glob.glob(PATH_TO_IMAGES + '/*.jpg') + glob.glob(PATH_TO_IMAGES + '/*.png') + glob.glob(PATH_TO_IMAGES + '/*.bmp')
        if save_results:
            RESULTS_DIR = IM_DIR + '_results'

    # Create results directory if user wants to save results
    if save_results:
        RESULTS_PATH = os.path.join(CWD_PATH,RESULTS_DIR)
        if not os.path.exists(RESULTS_PATH):
            os.makedirs(RESULTS_PATH)

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    
    floating_model = (input_details[0]['dtype'] == np.float32)
    
    input_mean = 127.5
    input_std = 127.5

    boxes_idx, classes_idx, scores_idx = 1, 3, 0
    video = cv2.VideoCapture(num_device)
    imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    min_hue_temp1, max_hue_temp1, min_saturation_temp1, max_saturation_temp1, min_value_temp1, max_value_temp1 = load_hsv_ranges("param_tes/color1.txt")
    min_hue_temp2, max_hue_temp2, min_saturation_temp2, max_saturation_temp2, min_value_temp2, max_value_temp2 = load_hsv_ranges("param_tes/color2.txt")
    min_hue_temp3, max_hue_temp3, min_saturation_temp3, max_saturation_temp3, min_value_temp3, max_value_temp3 = load_hsv_ranges("param_tes/color3.txt")

    not_see_logo = 0
# Loop over every image and perform detection
for image_path in images:

    # Load image and resize to expected shape [1xHxWx3]
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape 
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Apply morphological opening (erosion followed by dilation)
    image_opened = cv2.morphologyEx(image_resized, cv2.MORPH_OPEN, kernel)
    # Apply morphological closing (dilation followed by erosion)
    image_closed = cv2.morphologyEx(image_opened, cv2.MORPH_CLOSE, kernel)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

    detections = []

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))

            center_x = int((xmin + xmax) / 2)
            center_y = int((ymin + ymax) / 2)
            radius = int((xmax - xmin) / 2)
            cropped_image = image[ymin:ymax, xmin:xmax]

            image = cropped_image  # Replace 'your_image.jpg' with the path to your image file

            # image = cv2.resize(image, (640,480), interpolation= cv2.INTER_AREA)

            # Blur the image
            blurred_image = cv2.GaussianBlur(image, (11, 11), 0)

            # Convert image to HSV
            hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

            # Detect colors
            mask_raw_blue = detect_color(hsv_image, min_hue_temp1, max_hue_temp1, min_saturation_temp1, max_saturation_temp1, min_value_temp1, max_value_temp1)
            mask_raw_red = detect_color(hsv_image, min_hue_temp2, max_hue_temp2, min_saturation_temp2, max_saturation_temp2, min_value_temp2, max_value_temp2)
            mask_raw_yellow = detect_color(hsv_image, min_hue_temp3, max_hue_temp3, min_saturation_temp3, max_saturation_temp3, min_value_temp3, max_value_temp3)
            
            kernel = np.ones((5, 5), np.uint8)

            not_mask_blue = cv2.bitwise_not(mask_raw_blue)
            not_mask_red = cv2.bitwise_not(mask_raw_red)
            not_mask_yellow = cv2.bitwise_not(mask_raw_yellow)

            mask_yellow = cv2.bitwise_and(mask_raw_yellow, not_mask_blue)
            not_yellow = cv2.bitwise_not(mask_yellow)
            mask_red = cv2.bitwise_and(mask_raw_red, cv2.bitwise_and(not_mask_blue, not_yellow))
            not_red = cv2.bitwise_not(mask_red)
            mask_blue = cv2.bitwise_and(mask_raw_blue, cv2.bitwise_and(not_red, not_yellow))
            not_blue = cv2.bitwise_not(mask_blue)

            arrow = cv2.bitwise_and(not_blue, cv2.bitwise_and(not_red, not_yellow))
            arrow = cv2.morphologyEx(arrow, cv2.MORPH_CLOSE, kernel)
            arrow = cv2.dilate(arrow,kernel,iterations = 1)
            # combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

            # arrow = cv2.bitwise_not(combined_mask)

            circle_center_1, radius_2, radius_1 = detect_and_draw_contours(image, mask_yellow, (255, 0, 255), 2)
            circle_center_3, radius_4, radius_3 = detect_and_draw_contours(image, mask_red, (255, 0, 255), 1.3, min_radius=100)
            circle_center_5, radius_6, radius_5 = detect_and_draw_contours(image, mask_blue, (255, 0, 255), 1.2)

            center_x, center_y= circle_center_5
            upperBound = center_y - radius_6
            lowerBound = center_y + radius_6
            leftBound = center_x - radius_6
            rightBound = center_x + radius_6

            circle_center_2 = circle_center_1
            circle_center_4 = circle_center_3
            circle_center_6 = circle_center_5

            # # # Find lines using Hough Line Transform
            lines = cv2.HoughLinesP(arrow, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
            count_line = 0
            if lines is not None:

                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    # cv2.circle(image, (x1, y1), 2, (255, 255, 255), 5)
                    if((y1 > upperBound and y1 < lowerBound and x1 > leftBound and x1 < rightBound) or 
                       (y2 > upperBound and y2 < lowerBound and x2 > leftBound and x2 < rightBound)):
                        count_line += 1
                        print(f'line = {count_line}')
                        if count_line > 1:
                            break
                        tip_pos_1 = euclidean_distance((x1, y1), circle_center_1)
                        tip_pos_1_end = euclidean_distance((x2, y2), circle_center_1)
                        tip_pos_2 = euclidean_distance((x1, y1), circle_center_2)
                        tip_pos_2_end = euclidean_distance((x2, y2), circle_center_2)
                        tip_pos_3 = euclidean_distance((x1, y1), circle_center_3)
                        tip_pos_3_end = euclidean_distance((x2, y2), circle_center_3)
                        tip_pos_4 = euclidean_distance((x1, y1), circle_center_4)
                        tip_pos_4_end = euclidean_distance((x2, y2), circle_center_4)
                        tip_pos_5 = euclidean_distance((x1, y1), circle_center_5)
                        tip_pos_5_end = euclidean_distance((x2, y2), circle_center_5)
                        tip_pos_6 = euclidean_distance((x1, y1), circle_center_6)
                        tip_pos_6_end = euclidean_distance((x2, y2), circle_center_6)

                        is_inside_1 = tip_pos_1 <= radius_1 or tip_pos_1_end <= radius_1
                        is_inside_2 = tip_pos_2 <= radius_2 or tip_pos_2_end <= radius_2
                        is_inside_3 = tip_pos_3 <= radius_3 or tip_pos_3_end <= radius_3
                        is_inside_4 = tip_pos_4 <= radius_4 or tip_pos_4_end <= radius_4
                        is_inside_5 = tip_pos_5 <= radius_5 or tip_pos_5_end <= radius_5
                        is_inside_6 = tip_pos_6 <= radius_6 or tip_pos_6_end <= radius_6

                        if(is_inside_1):
                            print(f'tengah : {is_inside_1}')
                            cv2.putText(image, f'score : 10 point', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        elif(is_inside_2):
                            cv2.putText(image, f'score : 9 point', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                            print(f'9 : {is_inside_2}')
                        elif(is_inside_3):
                            cv2.putText(image, f'score : 8 point', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                            print(f'8 : {is_inside_3}')
                        elif(is_inside_4):
                            cv2.putText(image, f'score : 7 point', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                            print(f'7 : {is_inside_4}')
                        elif(is_inside_5):
                            cv2.putText(image, f'score : 6 point', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                            print(f'6 : {is_inside_5}')
                        elif(is_inside_6):
                            cv2.putText(image, f'score : 5 point', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                            print(f'5 : {is_inside_6}')

                        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 4)
                        cv2.circle(image, (x1, y1), 2, (255, 255, 255), 5)
                        cv2.circle(image, (x2, y2), 2, (0, 0, 255), 5)


            # Display result color
            # cv2.imshow('Putih', mask_white)
            # cv2.imshow('Hitam', mask_black)
            # cv2.imshow('Biru', mask_blue)
            # cv2.imshow('merah', mask_red)
            # cv2.imshow('kuning', mask_yellow)

            # cv2.imshow('not_putih', not_white)
            # cv2.imshow('not_hitam', not_black)
            # cv2.imshow('not_biru', not_blue)
            # cv2.imshow('not_merah', not_red)
            # cv2.imshow('not_kuning', not_yellow)

            # Display result
            # cv2.imshow('Combined Masks', combined_mask)
            cv2.imshow('arrow', arrow)  
            cv2.imshow('Image with Circles', image)
            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

            detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])

    # All the results have been drawn on the image, now display the image
    if show_results:
        # cv2.imshow('Cropped Object', cropped_image)
        # cv2.imshow('Morphological Closing', image_closed)
        # cv2.imshow('Resized Cropped Object', resized_cropped_image)
        # cv2.imshow('Object detector', image)
        
        # Press any key to continue to next image, or press 'q' to quit
        if cv2.waitKey(0) == ord('q'):
            break

    # Save the labeled image to results folder if desired
    if save_results:

        # Get filenames and paths
        image_fn = os.path.basename(image_path)
        image_savepath = os.path.join(CWD_PATH,RESULTS_DIR,image_fn)
        
        base_fn, ext = os.path.splitext(image_fn)
        txt_result_fn = base_fn +'.txt'
        txt_savepath = os.path.join(CWD_PATH,RESULTS_DIR,txt_result_fn)

        # Save image
        cv2.imwrite(image_savepath, image)

        # Write results to text file
        # (Using format defined by https://github.com/Cartucho/mAP, which will make it easy to calculate mAP)
        with open(txt_savepath,'w') as f:
            for detection in detections:
                f.write('%s %.4f %d %d %d %d\n' % (detection[0], detection[1], detection[2], detection[3], detection[4], detection[5]))

# Clean up
cv2.destroyAllWindows()

