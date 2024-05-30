import cv2
import numpy as np

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

def detect_and_draw_contours(main_frame, mask, color, min_radius=30, max_radius=350, exclusion_list=None):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if exclusion_list is None:
        exclusion_list = []
    
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        if min_radius <= radius <= max_radius:
            exclude = False
            for ex_center, ex_radius in exclusion_list:
                if euclidean_distance(center, ex_center) < ex_radius:
                    exclude = True
                    break
            if not exclude:
                cv2.circle(main_frame, center, radius, color, 3)
                exclusion_list.append((center, radius))
    return exclusion_list

def main():
    min_hue_temp1, max_hue_temp1, min_saturation_temp1, max_saturation_temp1, min_value_temp1, max_value_temp1 = load_hsv_ranges("param/color1.txt")
    min_hue_temp2, max_hue_temp2, min_saturation_temp2, max_saturation_temp2, min_value_temp2, max_value_temp2 = load_hsv_ranges("param/color2.txt")
    min_hue_temp3, max_hue_temp3, min_saturation_temp3, max_saturation_temp3, min_value_temp3, max_value_temp3 = load_hsv_ranges("param/color3.txt")
    min_hue_temp4, max_hue_temp4, min_saturation_temp4, max_saturation_temp4, min_value_temp4, max_value_temp4 = load_hsv_ranges("param/color4.txt")
    min_hue_temp5, max_hue_temp5, min_saturation_temp5, max_saturation_temp5, min_value_temp5, max_value_temp5 = load_hsv_ranges("param/color5.txt")

    # Load the image
    image = cv2.imread('target6.png')

    # Blur the image
    blurred_image = cv2.GaussianBlur(image, (11, 11), 0)

    # Convert image to HSV
    hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

    # Detect colors
    mask_raw_white = detect_color(hsv_image, min_hue_temp1, max_hue_temp1, min_saturation_temp1, max_saturation_temp1, min_value_temp1, max_value_temp1)
    mask_raw_black = detect_color(hsv_image, min_hue_temp2, max_hue_temp2, min_saturation_temp2, max_saturation_temp2, min_value_temp2, max_value_temp2)
    mask_raw_blue = detect_color(hsv_image, min_hue_temp3, max_hue_temp3, min_saturation_temp3, max_saturation_temp3, min_value_temp3, max_value_temp3)
    mask_raw_red = detect_color(hsv_image, min_hue_temp4, max_hue_temp4, min_saturation_temp4, max_saturation_temp4, min_value_temp4, max_value_temp4)
    mask_raw_yellow = detect_color(hsv_image, min_hue_temp5, max_hue_temp5, min_saturation_temp5, max_saturation_temp5, min_value_temp5, max_value_temp5)

    # Apply morphology operations to masks
    kernel = np.ones((5, 5), np.uint8)

    not_mask_white = cv2.bitwise_not(mask_raw_white)
    not_mask_black = cv2.bitwise_not(mask_raw_black)
    not_mask_blue = cv2.bitwise_not(mask_raw_blue)
    not_mask_red = cv2.bitwise_not(mask_raw_red)
    not_mask_yellow = cv2.bitwise_not(mask_raw_yellow)

    mask_yellow = cv2.bitwise_and(mask_raw_yellow, cv2.bitwise_and(not_mask_black, cv2.bitwise_and(not_mask_blue, not_mask_white)))
    not_yellow = cv2.bitwise_not(mask_yellow)
    mask_red = cv2.bitwise_and(mask_raw_red, cv2.bitwise_and(not_mask_black, cv2.bitwise_and(not_mask_blue, cv2.bitwise_and(not_mask_white, not_yellow))))
    not_red = cv2.bitwise_not(mask_red)
    mask_blue = cv2.bitwise_and(mask_raw_blue, cv2.bitwise_and(not_mask_black, cv2.bitwise_and(not_mask_white, cv2.bitwise_and(not_red, not_yellow))))
    not_blue = cv2.bitwise_not(mask_blue)
    mask_black = cv2.bitwise_and(mask_raw_black, cv2.bitwise_and(not_mask_white, cv2.bitwise_and(not_blue, cv2.bitwise_and(not_red, not_yellow))))
    mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel)
    mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel)
    not_black = cv2.bitwise_not(mask_black)
    mask_white = cv2.bitwise_and(mask_raw_white, cv2.bitwise_and(not_black, cv2.bitwise_and(not_blue, cv2.bitwise_and(not_red, not_yellow))))
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)

    exclusion_list = []
    exclusion_list = detect_and_draw_contours(image, mask_yellow, (0, 255, 255), min_radius=20, max_radius=80, exclusion_list=exclusion_list)
    exclusion_list = detect_and_draw_contours(image, mask_red, (0, 0, 255), min_radius=50, max_radius=120, exclusion_list=exclusion_list)
    exclusion_list = detect_and_draw_contours(image, mask_blue, (255, 0, 0), min_radius=80, max_radius=200, exclusion_list=exclusion_list)
    exclusion_list = detect_and_draw_contours(image, mask_black, (0, 0, 0), min_radius=150, max_radius=300, exclusion_list=exclusion_list)
    exclusion_list = detect_and_draw_contours(image, mask_white, (255, 255, 255), min_radius=200, max_radius=350, exclusion_list=exclusion_list)

    # Display result
    cv2.imshow('Image with Circles', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
