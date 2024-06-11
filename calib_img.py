import cv2
import numpy as np

# Load the image
image = cv2.imread('image.png')  # Replace 'your_image.jpg' with the path to your image

# Create a window to display the image
cv2.namedWindow('Image')

# Default values for the rectangle
rect_x, rect_y, rect_width, rect_height = 100, 100, 200, 200

# Default HSV parameters
min_hue, max_hue = 0, 0
min_saturation, max_saturation = 0, 0
min_value, max_value = 0, 0

# Set the default file name
file_name = 'param_tes/color.txt'
def adjust_rectangle(event, x, y, flags, param):
    global rect_x, rect_y, rect_width, rect_height
    if event == cv2.EVENT_LBUTTONDOWN:
        rect_x, rect_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        rect_width, rect_height = x - rect_x, y - rect_y

min_hue_temp = min_hue
min_saturation_temp = min_saturation
min_value_temp = min_value
max_hue_temp = max_hue
max_saturation_temp = max_saturation
max_value_temp = max_value
cv2.setMouseCallback('Image', adjust_rectangle)
kernel = np.ones((5, 5), np.uint8)

def write_file(param_file_name):
    with open(param_file_name, 'w') as param_file:
        param_file.write(f"min_hue={min_hue}\n")
        param_file.write(f"max_hue={max_hue}\n")
        param_file.write(f"min_saturation={min_saturation}\n")
        param_file.write(f"max_saturation={max_saturation}\n")
        param_file.write(f"min_value={min_value}\n")
        param_file.write(f"max_value={max_value}\n")

while True:
    # Display the image
    adjusted_image = image.copy()
    cv2.rectangle(adjusted_image, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 0, 255), 2)

    # Display the image with the adjustable rectangle
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([min_hue_temp, min_saturation_temp, min_value_temp])
    upper_bound = np.array([max_hue_temp, max_saturation_temp, max_value_temp])

    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    inverted_image = cv2.bitwise_not(mask)

    object_image = cv2.bitwise_and(image, image, mask=mask)

    roi = image[rect_y:rect_y + rect_height, rect_x:rect_x + rect_width]

    if roi.shape[0] > 0 and roi.shape[1] > 0:  
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        min_hue = np.min(hsv_roi[:, :, 0])
        max_hue = np.max(hsv_roi[:, :, 0])
        min_saturation = np.min(hsv_roi[:, :, 1])
        max_saturation = np.max(hsv_roi[:, :, 1])
        min_value = np.min(hsv_roi[:, :, 2])
        max_value = np.max(hsv_roi[:, :, 2])

    cv2.putText(adjusted_image, f'color: {str(file_name)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Image', adjusted_image)
    cv2.imshow('Binary Mask', mask)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('1'):
        file_name = "param_tes/color1.txt"
    elif key & 0xFF == ord('2'):
        file_name = "param_tes/color2.txt"
    elif key & 0xFF == ord('3'):
        file_name = "param_tes/color3.txt"
    elif key & 0xFF == ord('4'):
        file_name = "param_tes/color4.txt"
    elif key & 0xFF == ord('5'):
        file_name = "param_tes/color5.txt"

    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'):
        write_file(file_name)

        
        min_hue_temp = min_hue
        min_saturation_temp = min_saturation
        min_value_temp = min_value

        max_hue_temp = max_hue
        max_saturation_temp = max_saturation
        max_value_temp = max_value

cv2.destroyAllWindows()