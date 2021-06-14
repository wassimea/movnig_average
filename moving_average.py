import cv2
import numpy as np
 
def process_image(img, convert=True):
    gray = img
    if convert:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    gray = cv2.subtract(255,gray)
    ret,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return thresh

def main():
    cap = cv2.VideoCapture(0)
    _, img = cap.read()
    
    averageValue = np.float32(img)
    
    
    while(1):
        (grabbed, img) = cap.read()
        img_copy = img.copy()
    
        cv2.accumulateWeighted(img, averageValue, 0.05)
    
        movingAverage = cv2.convertScaleAbs(averageValue)

        cv2.imshow('Input Window', img)
    
        cv2.imshow('Moving Average', movingAverage)

        img_gray = process_image(img)
        r1_gray = process_image(movingAverage)
        r1 = img_gray - r1_gray
        r1 = process_image(r1, False)

        img_copy = cv2.bitwise_and(img_copy,img_copy,mask = 255-r1)
        cv2.imshow("Subtraction", img_copy)

    
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()