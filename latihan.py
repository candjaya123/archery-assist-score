import cv2
import numpy as np

# Load the image
image = cv2.imread("new_tes/target7.png")

# Resize the image to fit inside the rectangle
image = cv2.resize(image, (500, 700))  # Resize to fit on the left side

# Read score from score.txt
with open("score.txt", "r") as file:
    score_data = file.read().strip()

# Create a black background
background = np.zeros((800, 1000, 3), dtype=np.uint8)

# Paste the image onto the left side of the background
background[100:900, 100:600] = image

# Define rectangle parameters for displaying score
top_left_score = (700, 100)
bottom_right_score = (900, 200)
color_score = (0, 0, 255)  # Red color in BGR
thickness_score = 2  # Line thickness

# Draw rectangle on the right side to display score
cv2.rectangle(background, top_left_score, bottom_right_score, color_score, thickness_score)

# Display the score text
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
text_size = cv2.getTextSize(score_data, font, font_scale, font_thickness)[0]
text_x = int((bottom_right_score[0] + top_left_score[0] - text_size[0]) / 2)
text_y = int((bottom_right_score[1] + top_left_score[1] + text_size[1]) / 2)
cv2.putText(background, score_data, (text_x, text_y), font, font_scale, color_score, font_thickness)

# Display the image
cv2.imshow("Image with Score", background)
cv2.waitKey(0)
cv2.destroyAllWindows()
