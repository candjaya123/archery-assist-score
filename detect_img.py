import cv2
import numpy as np
countLine = 0
countCircle = 0

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

def main():
    min_hue_temp1, max_hue_temp1, min_saturation_temp1, max_saturation_temp1, min_value_temp1, max_value_temp1 = load_hsv_ranges("param/color1.txt")
    min_hue_temp2, max_hue_temp2, min_saturation_temp2, max_saturation_temp2, min_value_temp2, max_value_temp2 = load_hsv_ranges("param/color2.txt")
    min_hue_temp3, max_hue_temp3, min_saturation_temp3, max_saturation_temp3, min_value_temp3, max_value_temp3 = load_hsv_ranges("param/color3.txt")
    min_hue_temp4, max_hue_temp4, min_saturation_temp4, max_saturation_temp4, min_value_temp4, max_value_temp4 = load_hsv_ranges("param/color4.txt")
    min_hue_temp5, max_hue_temp5, min_saturation_temp5, max_saturation_temp5, min_value_temp5, max_value_temp5 = load_hsv_ranges("param/color5.txt")

    # Load the image
    image = cv2.imread('target2.jpg')  # Replace 'your_image.jpg' with the path to your image file

    # Blur the image
    blurred_image = cv2.GaussianBlur(image, (11, 11), 0)

    # Convert image to HSV
    hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

    # Detect colors
    mask1 = detect_color(hsv_image, min_hue_temp1, max_hue_temp1, min_saturation_temp1, max_saturation_temp1, min_value_temp1, max_value_temp1)
    mask2 = detect_color(hsv_image, min_hue_temp2, max_hue_temp2, min_saturation_temp2, max_saturation_temp2, min_value_temp2, max_value_temp2)
    mask3 = detect_color(hsv_image, min_hue_temp3, max_hue_temp3, min_saturation_temp3, max_saturation_temp3, min_value_temp3, max_value_temp3)
    mask4 = detect_color(hsv_image, min_hue_temp4, max_hue_temp4, min_saturation_temp4, max_saturation_temp4, min_value_temp4, max_value_temp4)
    mask5 = detect_color(hsv_image, min_hue_temp5, max_hue_temp5, min_saturation_temp5, max_saturation_temp5, min_value_temp5, max_value_temp5)

    # Apply morphology operations to masks
    kernel = np.ones((5, 5), np.uint8)
    # mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
    # mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
    # mask3 = cv2.morphologyEx(mask3, cv2.MORPH_OPEN, kernel)
    # mask5 = cv2.morphologyEx(mask5, cv2.MORPH_OPEN, kernel)
    # mask4 = cv2.morphologyEx(mask4, cv2.MORPH_OPEN, kernel)
    # mask5 = cv2.morphologyEx(mask5, cv2.MORPH_OPEN, kernel)


    # mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)
    # mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
    # mask3 = cv2.morphologyEx(mask3, cv2.MORPH_CLOSE, kernel)
    # mask4 = cv2.morphologyEx(mask4, cv2.MORPH_CLOSE, kernel)
    # mask5 = cv2.morphologyEx(mask5, cv2.MORPH_CLOSE, kernel)

    not_mask1 = cv2.bitwise_not(mask1)
    not_mask2 = cv2.bitwise_not(mask2)
    not_mask3 = cv2.bitwise_not(mask3)
    not_mask4 = cv2.bitwise_not(mask4)
    not_mask5 = cv2.bitwise_not(mask5)

    combined_mask = cv2.bitwise_or(mask1, cv2.bitwise_or(mask2, cv2.bitwise_or(mask3, cv2.bitwise_or(mask4, mask5))))
    arrow = cv2.bitwise_not(combined_mask)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    rows = gray.shape[0]
    # Define circle parameters
    circle_params = [
        (1, rows / 8, 100, 50, 1, 40),    # kuning paling dalam
        (1, rows / 8, 100, 50, 1, 70),    # kuning dalam
        (1, rows / 8, 100, 50, 1, 100),   # kuning luar
        (1, rows / 8, 100, 60, 100, 120), # merah dalam
        (1, rows / 8, 100, 50, 100, 200), # merah luar
        (1, rows / 8, 100, 60, 70, 200),  # biru dalam
        (1, rows / 8, 100, 60, 185, 220),  # biru luar
        (1, rows / 8, 100, 70, 250, 300),  # hitam dalam
        (1, rows / 8, 100, 70, 280, 320),  # hitam luar
        (1, rows / 8, 100, 70, 320, 340),  # hitam dalam
        (1, rows / 8, 100, 70, 320, 400),  # hitam luar
    ]
    
    for dp, minDist, param1, param2, minRadius, maxRadius in circle_params:
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, minDist,
                                   param1=param1, param2=param2,
                                   minRadius=minRadius, maxRadius=maxRadius)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                global countCircle
                countCircle += 1
                cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)  # Draw the outer circle
                if(countCircle == 11):
                    # cv2.circle(image, (i[0], i[1] - i[2]), 2, (0, 0, 255), 2) 
                    # cv2.circle(image, (i[0], i[1] + i[2]), 2, (0, 0, 255), 2)
                    # cv2.circle(image, (i[0] - i[2] , i[1]), 2, (0, 0, 255), 2)
                    # cv2.circle(image, (i[0] + i[2] , i[1]), 2, (0, 0, 255), 2)
                    upperBound = i[1] - i[2]
                    lowerBound = i[1] + i[2]
                    leftBound = i[0] - i[2]
                    rightBound = i[0] + i[2]

    # Find lines using Hough Line Transform
    lines = cv2.HoughLinesP(arrow, 1, np.pi / 180, threshold=400, minLineLength=50, maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
            # cv2.circle(image, (x1, y1), 2, (255, 255, 255), 5)
            if(y1 > upperBound and y1 < lowerBound and x1 > leftBound and x1 < rightBound):
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.circle(image, (x1, y1), 2, (255, 255, 255), 5)

    # Display result color
    # cv2.imshow('Putih', mask1)
    # cv2.imshow('Hitam', mask2)
    # cv2.imshow('Biru', mask3)
    # cv2.imshow('merah', mask4)
    # cv2.imshow('kuning', mask5)

    # cv2.imshow('not_putih', not_mask1)
    # cv2.imshow('not_hitam', not_mask2)
    # cv2.imshow('not_biru', not_mask3)
    # cv2.imshow('not_merah', not_mask4)
    # cv2.imshow('not_kuning', not_mask5)

    # Display result
    # cv2.imshow('Combined Masks', combined_mask)
    # cv2.imshow('arrow', arrow)  
    cv2.imshow('Image with Circles', image)

    # Wait for key press and exit on 'q'
    while cv2.waitKey(0) & 0xFF != ord('q'):
        pass

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
