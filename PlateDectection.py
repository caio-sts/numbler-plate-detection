import os
import cv2.cv2 as cv2
import numpy as np
import pytesseract


def coords2array(points):
    return np.array(points)

# cria o método para o evento de clique
def click_event(event, i, j, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        coordinates.append([i, j])
        cv2.circle(imgRoI, (i, j), 5, (0, 255, 0), 10)
        cv2.imshow('Region of Interest', imgRoI)
    if len(coordinates) == 4:
        cv2.destroyAllWindows()
        coords2array(coordinates)


pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\caios\\Documents\\APPS\\Tesseract\\tesseract.exe'

numberPlateDetector = cv2.CascadeClassifier('classifiers/numberPlate.xml')

for image in os.listdir('car_plates'):

    img = cv2.imread("car_plates/"+str(image))
    img = np.array(cv2.resize(img, (960, 720)))

    scale_value = 1.01

    nPlateDetector = numberPlateDetector.detectMultiScale(img, scale_value)

    std = 20

    coordinates = []

    if len(nPlateDetector) != 0:
        for (x, y, w, h) in nPlateDetector:
            imgRoI = np.array(img[y - std:y+h + std, x - std:x+w + std])
            RoICPY = imgRoI.copy()
            cv2.imshow('Region of Interest', imgRoI)

            # recebe o retorno do mouse
            cv2.setMouseCallback('Region of Interest', click_event)

            while len(coordinates) != 4:
                cv2.imshow('Region of Interest', imgRoI)
                cv2.waitKey(1)
                if len(coordinates) == 4:
                    cv2.destroyAllWindows()

            pts1 = np.float32(coordinates)

            pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

            M = cv2.getPerspectiveTransform(pts1, pts2)

            warped = cv2.warpPerspective(imgRoI, M, (w, h))

            warpedGray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

            kernel = np.ones((3, 3))
            warpedGray = cv2.morphologyEx(warpedGray, cv2.MORPH_OPEN, kernel)

            ret, thresh = cv2.threshold(warpedGray, np.mean(warpedGray), 255, cv2.THRESH_BINARY)

            cv2.imshow('Warped', thresh)
            charsDetected = pytesseract.image_to_string(thresh)
            cv2.destroyAllWindows()
            cv2.imshow(charsDetected, thresh)

            key = cv2.waitKey(0)

            if key == 115:  # "s" save image
                cv2.imwrite("carPlates/"+charsDetected+".jpg", img)
            elif key == 101:  # "e" edit filename and save
                namefile = input("Real plate number is:\n")
                cv2.imwrite("carPlates/" + namefile + ".jpg", img)
            elif key == 107:  # "k" skip image
                pass

            cv2.destroyAllWindows()
