import cv2
import numpy as np
from datetime import datetime

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

def detect_green_circles(main_frame,mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for contour in contours:
        # Hitung momen kontur
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))
            # Gambar lingkaran dan pusat kontur (opsional untuk visualisasi)
            cv2.circle(main_frame, (cX, cY), 10, (0, 255, 0), -1)
        else:
            cX, cY = 0, 0
    
    if len(centers) == 4:
        # Urutkan berdasarkan koordinat x dan y
        centers_sorted = sorted(centers, key=lambda k: (k[1], k[0]))  # sort by y first, then by x

        # Buat list untuk mengkategorikan pusat lingkaran
        center1, center2, center3, center4 = None, None, None, None

        # Dapatkan dimensi gambar untuk referensi pembagian area
        height, width, _ = main_frame.shape
        mid_x, mid_y = width // 2, height // 2

        # Kategorikan berdasarkan posisi relatif terhadap tengah gambar
        for center in centers_sorted:
            x, y = center
            if x <= mid_x and y <= mid_y:
                center1 = center  # atas kiri
            elif x > mid_x and y <= mid_y:
                center2 = center  # atas kanan
            elif x <= mid_x and y > mid_y:
                center3 = center  # bawah kiri
            elif x > mid_x and y > mid_y:
                center4 = center  # bawah kanan

    else:
        center1 = center2 = center3 = center4 = None
        # print("Jumlah lingkaran hijau yang terdeteksi bukan 4.")
    return center1, center2, center3, center4

def main():
    min_hue_temp1, max_hue_temp1, min_saturation_temp1, max_saturation_temp1, min_value_temp1, max_value_temp1 = load_hsv_ranges("param_tes/color1.txt")
    min_hue_temp2, max_hue_temp2, min_saturation_temp2, max_saturation_temp2, min_value_temp2, max_value_temp2 = load_hsv_ranges("param_tes/color2.txt")
    min_hue_temp3, max_hue_temp3, min_saturation_temp3, max_saturation_temp3, min_value_temp3, max_value_temp3 = load_hsv_ranges("param_tes/color3.txt")
    min_hue_temp4, max_hue_temp4, min_saturation_temp4, max_saturation_temp4, min_value_temp4, max_value_temp4 = load_hsv_ranges("param_tes/color4.txt")
    min_hue_temp5, max_hue_temp5, min_saturation_temp5, max_saturation_temp5, min_value_temp5, max_value_temp5 = load_hsv_ranges("param_tes/color5.txt")

    # Load the image
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.imread('new_tes/new7.png')  # Replace 'your_image.jpg' with the path to your image file
    blurred_image = cv2.GaussianBlur(image, (11, 11), 0)
    hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
    border = detect_color(hsv_image, min_hue_temp5, max_hue_temp5, min_saturation_temp5, max_saturation_temp5, min_value_temp5, max_value_temp5)
    mask_border = cv2.dilate(border,kernel,iterations = 1)
    center1, center2, center3, center4 = detect_green_circles(image,mask_border)
    width_trans = 640
    height_trans = 640

    pts1 = np.float32([center1,center2,center3,center4])
    pts2 = np.float32([[0,0],[width_trans,0],[0,height_trans],[width_trans,height_trans]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgOut = cv2.warpPerspective(image,matrix,(width_trans,height_trans))

    blurred_imgOut = cv2.GaussianBlur(imgOut, (11, 11), 0)
    hsv_imgOut = cv2.cvtColor(blurred_imgOut, cv2.COLOR_BGR2HSV)

    # cv2.imshow('Image trans', imgOut)

    mask_raw_blue = detect_color(hsv_imgOut, min_hue_temp1, max_hue_temp1, min_saturation_temp1, max_saturation_temp1, min_value_temp1, max_value_temp1)
    mask_raw_red = detect_color(hsv_imgOut, min_hue_temp2, max_hue_temp2, min_saturation_temp2, max_saturation_temp2, min_value_temp2, max_value_temp2)
    mask_raw_yellow = detect_color(hsv_imgOut, min_hue_temp3, max_hue_temp3, min_saturation_temp3, max_saturation_temp3, min_value_temp3, max_value_temp3)
    mask_raw_yellow = detect_color(hsv_imgOut, min_hue_temp3, max_hue_temp3, min_saturation_temp3, max_saturation_temp3, min_value_temp3, max_value_temp3)
    mask_raw_white = detect_color(hsv_imgOut, min_hue_temp4, max_hue_temp4, min_saturation_temp4, max_saturation_temp4, min_value_temp4, max_value_temp4)

    # Apply morphology operations to masks

    not_mask_blue = cv2.bitwise_not(mask_raw_blue)
    not_mask_red = cv2.bitwise_not(mask_raw_red)
    not_mask_yellow = cv2.bitwise_not(mask_raw_yellow)

    mask_yellow = cv2.bitwise_and(mask_raw_yellow, not_mask_blue)
    not_yellow = cv2.bitwise_not(mask_yellow)
    mask_red = cv2.bitwise_and(mask_raw_red, cv2.bitwise_and(not_mask_blue, not_yellow))
    not_red = cv2.bitwise_not(mask_red)
    mask_blue = cv2.bitwise_and(mask_raw_blue, cv2.bitwise_and(not_red, not_yellow))
    not_blue = cv2.bitwise_not(mask_blue)
    mask_white = cv2.bitwise_and(mask_raw_white, cv2.bitwise_and(not_yellow, cv2.bitwise_and(not_red, not_blue)))
    not_white = cv2.bitwise_not(mask_white)
 
    arrow = cv2.bitwise_and(not_blue, cv2.bitwise_and(not_red, cv2.bitwise_and(not_white, not_yellow)))
    arrow = cv2.morphologyEx(arrow, cv2.MORPH_CLOSE, kernel)
    arrow = cv2.dilate(arrow,kernel,iterations = 1)

    circle_center_1, radius_2, radius_1 = detect_and_draw_contours(imgOut, mask_yellow, (255, 0, 255), 2)
    circle_center_3, radius_4, radius_3 = detect_and_draw_contours(imgOut, mask_red, (0, 0, 255), 1.3)
    circle_center_5, radius_6, radius_5 = detect_and_draw_contours(imgOut, mask_blue, (255, 0, 0), 1.2)

    center_x, center_y= circle_center_5
    upperBound = center_y - radius_6
    lowerBound = center_y + radius_6
    leftBound = center_x - radius_6
    rightBound = center_x + radius_6

    circle_center_2 = circle_center_1
    circle_center_4 = circle_center_3
    circle_center_6 = circle_center_5

    # # # Find lines using Hough Line Transform
    lines = cv2.HoughLinesP(arrow, 1, np.pi / 180, threshold = 100,minLineLength = 100,maxLineGap = 10)
    count_line = 0
    score = "No score"
    if lines is not None:
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if((y1 > upperBound and y1 < lowerBound and x1 > leftBound and x1 < rightBound) or 
               (y2 > upperBound and y2 < lowerBound and x2 > leftBound and x2 < rightBound)):
                count_line += 1
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
                    score = "10 point"
                    cv2.putText(image, f'score : 10 point', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                elif(is_inside_2):
                    score = "9 point"
                    cv2.putText(image, f'score : 9 point', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    print(f'9 : {is_inside_2}')
                elif(is_inside_3):
                    score = "8 point"
                    cv2.putText(image, f'score : 8 point', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    print(f'8 : {is_inside_3}')
                elif(is_inside_4):
                    score = "7 point"
                    cv2.putText(image, f'score : 7 point', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    print(f'7 : {is_inside_4}')
                elif(is_inside_5):
                    score = "6 point"
                    cv2.putText(image, f'score : 6 point', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    print(f'6 : {is_inside_5}')
                elif(is_inside_6):
                    score = "5 point"
                    cv2.putText(image, f'score : 5 point', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    print(f'5 : {is_inside_6}')

                cv2.line(imgOut, (x1, y1), (x2, y2), (255, 0, 0), 4)
                cv2.circle(imgOut, (x1, y1), 2, (255, 255, 255), 5)
                cv2.circle(imgOut, (x2, y2), 2, (0, 0, 255), 5)

    # Save the processed image
    # output_image_path = 'processed_image.png'
    # cv2.imwrite(output_image_path, image)

    # Write score and time to score.txt
    # with open('score.txt', 'a') as file:
    #     current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    #     file.write(f'Time: {current_time}, Score: {score}\n')

    # Display result color
    
    # cv2.imshow('Biru', mask_blue)
    cv2.imshow('merah', mask_red)
    cv2.imshow('kuning', mask_yellow)
    # cv2.imshow('batas', border)

    # cv2.imshow('not_biru', not_blue)
    # cv2.imshow('not_merah', not_red)
    # cv2.imshow('not_kuning', not_yellow)

    # Display result
    # cv2.imshow('Combined Masks', combined_mask)
    cv2.imshow('arrow', arrow)  
    cv2.imshow('Image with Circles', image)
    cv2.imshow('Image trans', imgOut)

    # Wait for key press and exit on 'q'
    while cv2.waitKey(0) & 0xFF != ord('q'):
        pass

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
