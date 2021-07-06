import cv2
import numpy as np
from argparse import ArgumentParser
import sys
 
def main():
    parser = ArgumentParser()
    parser.add_argument("-a", "--alpha", help="Set alpha value")
    args = parser.parse_args()
    alpha = float(args.alpha)

    cap = cv2.VideoCapture(0)
    _, img = cap.read()
    
    
    #initialise average image (gray level)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    movingAverage = np.float32(img)
    
    
    while(1):
    
        # grab image from camera
        (grabbed, img) = cap.read()
        cv2.imshow('Input Window', img)
        
        # moving average
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.accumulateWeighted(image, movingAverage, alpha)
         
        # float32 average background image to 8uint
        background= cv2.convertScaleAbs(movingAverage)
        
        cv2.imshow('Background (Moving Average)', background)
        
        # get foreground
        foreground = cv2.absdiff(image, background)
        _,mask= cv2.threshold(foreground, 5, 255, cv2.THRESH_BINARY)
        kernel = np.ones((15,15),np.uint8)
        mask= cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        image= cv2.bitwise_and(img,img,mask = mask)  
        cv2.imshow('Foreground', image)

#        img_gray = process_image(img)
#        r1_gray = process_image(movingAverage)
#        r1 = img_gray - r1_gray
#        r1 = process_image(r1, False)

#        img_copy = cv2.bitwise_and(img_copy,img_copy,mask = 255-r1)
#        cv2.imshow("Subtraction", img_copy)

        # ESC to terminate   
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()