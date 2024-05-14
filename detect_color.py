import cv2
import numpy as np

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
    # min_hue_temp6, max_hue_temp6, min_saturation_temp6, max_saturation_temp6, min_value_temp6, max_value_temp6 = load_hsv_ranges("color6.txt")

    # Open camera
    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()

        # Convert frame to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Detect color
        mask1 = detect_color(hsv_frame, min_hue_temp1, max_hue_temp1, min_saturation_temp1, max_saturation_temp1, min_value_temp1, max_value_temp1)
        mask2 = detect_color(hsv_frame, min_hue_temp2, max_hue_temp2, min_saturation_temp2, max_saturation_temp2, min_value_temp2, max_value_temp2)
        mask3 = detect_color(hsv_frame, min_hue_temp3, max_hue_temp3, min_saturation_temp3, max_saturation_temp3, min_value_temp3, max_value_temp3)
        mask4 = detect_color(hsv_frame, min_hue_temp4, max_hue_temp4, min_saturation_temp4, max_saturation_temp4, min_value_temp4, max_value_temp4)
        mask5 = detect_color(hsv_frame, min_hue_temp5, max_hue_temp5, min_saturation_temp5, max_saturation_temp5, min_value_temp5, max_value_temp5)
        # mask6 = detect_color(hsv_frame, min_hue_temp6, max_hue_temp6, min_saturation_temp6, max_saturation_temp6, min_value_temp6, max_value_temp6)
        # combined_mask = cv2.bitwise_or(mask1, cv2.bitwise_or(mask2, cv2.bitwise_or(mask3, cv2.bitwise_or(mask4, cv2.bitwise_or(mask5, mask6)))))
        combined_mask = cv2.bitwise_or(mask1, cv2.bitwise_or(mask2, cv2.bitwise_or(mask3, cv2.bitwise_or(mask4, mask5))))
        arrow = cv2.bitwise_not(combined_mask)

        # Display result color
        cv2.imshow('Binary Mask1', mask1)
        cv2.imshow('Binary Mask2', mask2)
        cv2.imshow('Binary Mask3', mask3)
        cv2.imshow('Binary Mask4', mask4)
        cv2.imshow('Binary Mask5', mask5)
        # cv2.imshow('Binary Mask6', mask6)

        # Display result
        cv2.imshow('Combined Masks', combined_mask)
        cv2.imshow('arrow', arrow)
        cv2.imshow('Camera Feed', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
