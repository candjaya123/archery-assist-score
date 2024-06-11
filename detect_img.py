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


def main():
    min_hue_temp1, max_hue_temp1, min_saturation_temp1, max_saturation_temp1, min_value_temp1, max_value_temp1 = load_hsv_ranges("param/color1.txt")
    min_hue_temp2, max_hue_temp2, min_saturation_temp2, max_saturation_temp2, min_value_temp2, max_value_temp2 = load_hsv_ranges("param/color2.txt")
    min_hue_temp3, max_hue_temp3, min_saturation_temp3, max_saturation_temp3, min_value_temp3, max_value_temp3 = load_hsv_ranges("param/color3.txt")
    min_hue_temp4, max_hue_temp4, min_saturation_temp4, max_saturation_temp4, min_value_temp4, max_value_temp4 = load_hsv_ranges("param/color4.txt")
    min_hue_temp5, max_hue_temp5, min_saturation_temp5, max_saturation_temp5, min_value_temp5, max_value_temp5 = load_hsv_ranges("param/color5.txt")

    # Load the image
    image = cv2.imread('test_images/target_5.png')  # Replace 'your_image.jpg' with the path to your image file

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
    not_white = cv2.bitwise_not(mask_white)
    # mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)

    # combined_mask = cv2.bitwise_or(mask_white, cv2.bitwise_or(mask_black, cv2.bitwise_or(mask_blue, cv2.bitwise_or(mask_red, mask_yellow))))
    
    arrow = cv2.bitwise_and(not_white, cv2.bitwise_and(not_blue, cv2.bitwise_and(not_red, cv2.bitwise_and(not_yellow, not_black))))
    arrow = cv2.bitwise_or(arrow, mask_black)
    arrow = cv2.subtract(arrow, mask_black)
    arrow = cv2.morphologyEx(arrow, cv2.MORPH_CLOSE, kernel)
    arrow = cv2.dilate(arrow,kernel,iterations = 1)
    # combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # arrow = cv2.bitwise_not(combined_mask)

    # circle_center_1, radius_2, radius_1 = detect_and_draw_contours(image, mask_yellow, (255, 0, 255), 2)
    # circle_center_3, radius_4, radius_3 = detect_and_draw_contours(image, mask_red, (255, 0, 255), 1.25)
    # circle_center_5, radius_6, radius_5 = detect_and_draw_contours(image, mask_blue, (255, 0, 255), 1.2)
    # circle_center_7, radius_8, radius_7 = detect_and_draw_contours(image, mask_black, (255, 0, 255), 1.15)

    detect_and_draw_contours(image, not_white, (255, 0, 255), 1)
    cv2.imshow('uye', not_white)  
    # radius_9 = int(radius_8 * 1.1)
    # circle_center_9 = circle_center_10 = circle_center_7
    # radius_10 = int(radius_8 * 1.24)

    # cv2.circle(image, circle_center_9, radius_9, (255, 0, 255), 5)
    # cv2.circle(image, circle_center_10, radius_10, (255, 0, 255), 5)
    # center_x, center_y= circle_center_10
    # upperBound = center_y - radius_10
    # lowerBound = center_y + radius_10
    # leftBound = center_x - radius_10
    # rightBound = center_x + radius_10
    # circle_center_2 = circle_center_1 
    # circle_center_4 = circle_center_3 
    # circle_center_6 = circle_center_5 
    # circle_center_8 = circle_center_7 
    # # radius_10,  circle_center_9 = circle_center_10, radius_9 = detect_and_draw_contours(image, mask_white, (255, 0, 255), 2)


    # # # # Find lines using Hough Line Transform
    # lines = cv2.HoughLinesP(arrow, 1, np.pi / 180, threshold=100, minLineLength=10, maxLineGap=30)
    # count_line = 0
    # if lines is not None:
        
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         # cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
    #         # cv2.circle(image, (x1, y1), 2, (255, 255, 255), 5)
    #         if((y1 > upperBound and y1 < lowerBound and x1 > leftBound and x1 < rightBound) or 
    #            (y2 > upperBound and y2 < lowerBound and x2 > leftBound and x2 < rightBound)):
    #             count_line += 1
    #             print(f'line = {count_line}')
    #             if count_line > 1:
    #                 break
    #         # if(y2 > upperBound and y2 < lowerBound and x2 > leftBound and x2 < rightBound):
    #         # if(y1 > upperBound and y1 < lowerBound and x1 > leftBound and x1 < rightBound):
    #             tip_pos_1 = euclidean_distance((x1, y1), circle_center_1)
    #             tip_pos_1_end = euclidean_distance((x2, y2), circle_center_1)
    #             tip_pos_2 = euclidean_distance((x1, y1), circle_center_2)
    #             tip_pos_2_end = euclidean_distance((x2, y2), circle_center_2)
    #             tip_pos_3 = euclidean_distance((x1, y1), circle_center_3)
    #             tip_pos_3_end = euclidean_distance((x2, y2), circle_center_3)
    #             tip_pos_4 = euclidean_distance((x1, y1), circle_center_4)
    #             tip_pos_4_end = euclidean_distance((x2, y2), circle_center_4)
    #             tip_pos_5 = euclidean_distance((x1, y1), circle_center_5)
    #             tip_pos_5_end = euclidean_distance((x2, y2), circle_center_5)
    #             tip_pos_6 = euclidean_distance((x1, y1), circle_center_6)
    #             tip_pos_6_end = euclidean_distance((x2, y2), circle_center_6)
    #             tip_pos_7 = euclidean_distance((x1, y1), circle_center_7)
    #             tip_pos_7_end = euclidean_distance((x2, y2), circle_center_7)
    #             tip_pos_8 = euclidean_distance((x1, y1), circle_center_8)
    #             tip_pos_8_end = euclidean_distance((x2, y2), circle_center_8)
    #             tip_pos_9 = euclidean_distance((x1, y1), circle_center_9)
    #             tip_pos_9_end = euclidean_distance((x2, y2), circle_center_9)
    #             tip_pos_10 = euclidean_distance((x1, y1), circle_center_10)
    #             tip_pos_10_end = euclidean_distance((x2, y2), circle_center_10)

    #             is_inside_1 = tip_pos_1 <= radius_1 or tip_pos_1_end <= radius_1
    #             is_inside_2 = tip_pos_2 <= radius_2 or tip_pos_2_end <= radius_2
    #             is_inside_3 = tip_pos_3 <= radius_3 or tip_pos_3_end <= radius_3
    #             is_inside_4 = tip_pos_4 <= radius_4 or tip_pos_4_end <= radius_4
    #             is_inside_5 = tip_pos_5 <= radius_5 or tip_pos_5_end <= radius_5
    #             is_inside_6 = tip_pos_6 <= radius_6 or tip_pos_6_end <= radius_6
    #             is_inside_7 = tip_pos_7 <= radius_7 or tip_pos_7_end <= radius_7
    #             is_inside_8 = tip_pos_8 <= radius_8 or tip_pos_8_end <= radius_8
    #             is_inside_9 = tip_pos_9 <= radius_9 or tip_pos_9_end <= radius_9
    #             is_inside_10 = tip_pos_10 <= radius_10 or tip_pos_10_end <= radius_10

    #             if(is_inside_1):
    #                 print(f'tengah : {is_inside_1}')
    #                 cv2.putText(image, f'score : 10 point', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    #             elif(is_inside_2):
    #                 cv2.putText(image, f'score : 9 point', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    #                 print(f'9 : {is_inside_2}')
    #             elif(is_inside_3):
    #                 cv2.putText(image, f'score : 8 point', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    #                 print(f'8 : {is_inside_3}')
    #             elif(is_inside_4):
    #                 cv2.putText(image, f'score : 7 point', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    #                 print(f'7 : {is_inside_4}')
    #             elif(is_inside_5):
    #                 cv2.putText(image, f'score : 6 point', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    #                 print(f'6 : {is_inside_5}')
    #             elif(is_inside_6):
    #                 cv2.putText(image, f'score : 5 point', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    #                 print(f'5 : {is_inside_6}')
    #             elif(is_inside_7):
    #                 cv2.putText(image, f'score : 4 point', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    #                 print(f'4 : {is_inside_7}')
    #             elif(is_inside_8):
    #                 cv2.putText(image, f'score : 3 point', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    #                 print(f'3 : {is_inside_8}')
    #             elif(is_inside_9):
    #                 cv2.putText(image, f'score : 2 point', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    #                 print(f'2 : {is_inside_9}')
    #             elif(is_inside_10):
    #                 cv2.putText(image, f'score : 1 point', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    #                 print(f'1 : {is_inside_10}')
    #             else:
    #                 cv2.putText(image, f'score : 0 point', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    #             cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 4)
    #             cv2.circle(image, (x1, y1), 2, (255, 255, 255), 5)
    #             cv2.circle(image, (x2, y2), 2, (0, 0, 255), 5)

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

    # Wait for key press and exit on 'q'
    while cv2.waitKey(0) & 0xFF != ord('q'):
        pass

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
