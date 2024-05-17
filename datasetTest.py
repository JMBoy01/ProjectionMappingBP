import cv2 as cv
import numpy as np
import os

def detectAndDrawCirclesPatternFind(img, blobDetector, isImgPatternImg):
    patternSize = (3, 5) # 3 horizontaal, 5 vertical mag 1 schuin geteld worden

    binary = img.copy()
    if not isImgPatternImg:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, binary = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)
        # kleine blur om de randen minder scherp en pixelated te maken
        binary = cv.GaussianBlur(binary, (3, 3), 1)

    patternWasFound, centersCam = cv.findCirclesGrid(binary, patternSize, flags=(cv.CALIB_CB_ASYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING), blobDetector=blobDetector)

    if patternWasFound:
        img = cv.drawChessboardCorners(img, patternSize, centersCam, patternWasFound)

        centersCam = np.squeeze(centersCam)
        return img, binary, centersCam
    else:
        return img, binary, None

def loadImageListFromFolder(path, fileStructerName, fileExtension):
    if path[-1] != '/':
        path += '/'
    
    if fileExtension[0] != '.':
        fileExtension = '.' + fileExtension

    imgList = []
    fileAmount = 0
    try:
        files = os.listdir(path)
        fileAmount = len(files)
    except Exception:
        print("Path not found...")
        return None
    
    for i in range(0, fileAmount, 1):
        img = cv.imread(path + fileStructerName + str(i) + fileExtension)
        imgList.append(img)
    
    return imgList

def drawDetectionsAndSave(imgList, charucoDetector, blobDetector, path, fileStructureName):
    if path[-1] != '/':
        path += '/'

    count = 0
    for img in imgList:
        gray = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)
        charucoCorners, _, _, _ = charucoDetector.detectBoard(gray)
        if charucoCorners is not None:
            patternWasFound = False
            if len(charucoCorners) == 35:
                patternWasFound = True
            img = cv.drawChessboardCorners(img, (7, 5), charucoCorners, patternWasFound)

        img, _, _ = detectAndDrawCirclesPatternFind(img, blobDetector, False)

        cv.imwrite(path + fileStructureName + str(count) + ".png", img)
        count += 1

def initBlobDetector():
    params = cv.SimpleBlobDetector_Params()
    params.blobColor = 255 # 0 = zwart, 255 = wit
    params.filterByColor = True
    params.filterByArea = True
    params.minCircularity = 0.5
    params.minDistBetweenBlobs = 5
    params.filterByCircularity = True
    params.filterByConvexity = False
    params.filterByInertia = False
    params.collectContours = True

    blobDetector = cv.SimpleBlobDetector_create(params)
    return blobDetector

def findDict(img):
    dictionaries = [
        cv.aruco.DICT_4X4_50,
        cv.aruco.DICT_4X4_100,
        cv.aruco.DICT_4X4_250,
        cv.aruco.DICT_4X4_1000,
        cv.aruco.DICT_5X5_50,
        cv.aruco.DICT_5X5_100,
        cv.aruco.DICT_5X5_250,
        cv.aruco.DICT_5X5_1000,
        cv.aruco.DICT_6X6_50,
        cv.aruco.DICT_6X6_100,
        cv.aruco.DICT_6X6_250,
        cv.aruco.DICT_6X6_1000,
        cv.aruco.DICT_7X7_50,
        cv.aruco.DICT_7X7_100,
        cv.aruco.DICT_7X7_250,
        cv.aruco.DICT_7X7_1000,
        cv.aruco.DICT_ARUCO_ORIGINAL
    ]

    possibleDicts = []
    for dictionary in dictionaries:
        dictionaryObj = cv.aruco.getPredefinedDictionary(dictionary)
        arucoDetector = cv.aruco.ArucoDetector(dictionaryObj)
        _, usedMarkerIds, _ = arucoDetector.detectMarkers(img)
        # print(usedMarkerIds)
        if usedMarkerIds is not None and len(usedMarkerIds) == 24:
            possibleDicts.append(dictionary)

    return possibleDicts

def loadCameraCalibration(filename):
    fs = cv.FileStorage(filename, cv.FILE_STORAGE_READ)
    cameraMatrix = fs.getNode("cameraMatrix").mat()
    distCoeffs = fs.getNode("distCoeffs").mat()
    fs.release()
    return cameraMatrix, distCoeffs

def initCharucoDetector():
    boardPhoto = cv.imread("./Overige Images/board.jpg")

    cameraMatrix, distCoeffs = loadCameraCalibration("cameraHome")

    dictionaries = findDict(cv.cvtColor(boardPhoto, cv.COLOR_BGR2GRAY))
    dictionary = cv.aruco.getPredefinedDictionary(dictionaries[1])

    boardSize = (8, 6) # aantal vakjes (hor, ver)
    squareLength = 0.03 # in meter
    markerLength = 0.015 # in meter
    charucoBoard = cv.aruco.CharucoBoard(boardSize, squareLength, markerLength, dictionary)

    # saveCharucoBoard(charucoBoard, (1920, 1080), "board_img.jpg")

    charucoDetector = cv.aruco.CharucoDetector(charucoBoard)
    return charucoDetector

def main():
    blobDetector = initBlobDetector()
    charucoDetector = initCharucoDetector()

    print("Loading image list...")
    imgList = loadImageListFromFolder("./projCalibrationData/rawData", "img", "png")
    patternImgList = loadImageListFromFolder("./patternImages/", "patternImg", ".png")

    print("Detecting charuco board and circle pattern on images...")
    drawDetectionsAndSave(imgList, charucoDetector, blobDetector, "./projCalibrationData/detectedData", "detectedImg")
    
    count = 0
    for img in patternImgList:
        img, _, _ = detectAndDrawCirclesPatternFind(img, blobDetector, False)
        cv.imwrite("./projCalibrationData/detectedPattern/detectedPatternImg" + str(count) + ".png", img)
        count += 1

    print("Done detecting on all images, exiting program...")

if __name__ == "__main__":
    main()