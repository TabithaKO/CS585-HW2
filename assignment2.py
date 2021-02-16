import cv2
import numpy as np
import math

def my_skin_detect(src):
    '''
    Function that detects whether a pixel belongs to the skin based on RGB values
    Args: 
        src The source color image
    Returns: 
        dst The destination grayscale image where skin pixels are colored white and the rest are colored black
    Surveys of skin color modeling and detection techniques:
    Vezhnevets, Vladimir, Vassili Sazonov, and Alla Andreeva. "A survey on pixel-based skin color detection techniques." Proc. Graphicon. Vol. 3. 2003.
    Kakumanu, Praveen, Sokratis Makrogiannis, and Nikolaos Bourbakis. "A survey of skin-color modeling and detection methods." Pattern recognition 40.3 (2007): 1106-1122.
    '''
    dst = np.zeros(np.shape(src)[:-1], dtype= np.uint8)
    
    mask = np.logical_and.reduce((src[:,:,2]>180, src[:,:,1]>50))
    dst[mask] = 255

    return dst

def my_frame_differencing(prev, curr):
    '''
    Function that does frame differencing between the current frame and the previous frame
    Args:
        src The current color image
        prev The previous color image
    Returns:
        dst The destination grayscale image where pixels are colored white if the corresponding pixel intensities in the current
    and previous image are not the same
    '''
    dst = cv2.absdiff(prev, curr)
    gs = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    dst = (gs > 50).astype(np.uint8) * 255
    return dst
    


def pyramids(img, temp_list):
    temp_list.append(img)
    layer = img.copy()
    for i in range(6):
        layer = cv2.pyrUp(layer)
        temp_list.append(layer)

    

# turn on the webcam
cap = cv2.VideoCapture(0)
_, bg = cap.read()
# bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY) 


# read the templates and detect the skin

hand = cv2.imread("y.jpg")
hand = hand[:,:,2]
blur = cv2.blur(hand,(5,5))
_, hand = cv2.threshold(blur, 130, 255, cv2.THRESH_BINARY)
# hand_mask = my_skin_detect(hand)

thumbs_up = cv2.imread("thumb.jpg")
thumbs_up = thumbs_up[:,:,2]
blur = cv2.blur(thumbs_up,(10,10))
_, thumbs_up = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
# thumb_mask = my_skin_detect(thumbs_up)

fist = cv2.imread("smol.jpg")
fist = fist[:,:,2]
blur = cv2.blur(fist,(10,10))
_, fist = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY)
# fist_mask = my_skin_detect(fist)

v = cv2.imread("v.jpg")
v = v[:,:,2]
blur = cv2.blur(v,(10,10))
_, v = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
# bbs_mask = my_skin_detect(baby_shark)


template_list = [hand, thumbs_up, fist, v]

# for i in template_list:
#     pyramids(i, temp_list)

# for i in temp_list:
#     cv2.imshow("image.jpeg",i)

# print("tlist",len(temp_list))

# count the number of frames
# counter  = 0 
colors = {
    0: (0, 0, 255),
    1: (0, 255, 0),
    2: (255, 0, 0),
    3: (230, 245, 66)
}

while True:
    _, frame = cap.read()
    my_frame = my_frame_differencing(bg,frame)
    # my_frame = my_skin_detect(frame)
    
    # global vars
    max_id = -1
    curr_max = -1

    top_left = 0
    bottom_right = 0

    for temp in range(0, len(template_list)):
    
        result = cv2.matchTemplate(my_frame, template_list[temp], cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        h,w = template_list[temp].shape

        if max_val > curr_max:
          curr_max = max_val
          max_id = temp
          top_left =  max_loc
          bottom_right =  (top_left[0] + w, top_left[1] + h)
    
    
    cv2.rectangle(frame,top_left, bottom_right, colors[max_id], 2)

    cv2.imshow("temp",template_list[max_id])
    cv2.imshow("my frame",my_frame)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)

    # counter += 1
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()