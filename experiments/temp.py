import cv2

# Load an image (make sure image.jpg is in the same folder)
img = cv2.imread('/Users/salonikarekar/Desktop/College work/sem 5/sentiment analysis project files /PrivateTest_1623042.jpg')

# Check if image is loaded properly
if img is None:
    print("Error: Could not read the image.")
else:
    print(img.shape)
    cv2.imshow("Displayed Image", img)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()

