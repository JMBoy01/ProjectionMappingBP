import cv2 as cv
import numpy as np

def saveCameraCalibration(filename, cameraMatrix, distCoeffs):
    fs = cv.FileStorage(filename, cv.FILE_STORAGE_WRITE)
    fs.write("cameraMatrix", cameraMatrix)
    fs.write("distCoeffs", distCoeffs)

    fs.release()
    print("Calibratiegegevens opgeslagen in", filename)

# source: Joni
def detectCharucoCorners(gray, arucoDetector, charucoDetector):
    corners, ids, rejectedImgPoints = arucoDetector.detectMarkers(gray)
    corners, ids, rejectedImgPoints, recoveredIds = arucoDetector.refineDetectedMarkers(
            image = gray,
            board = charucoDetector.getBoard(),
            detectedCorners = corners,
            detectedIds = ids,
            rejectedCorners = rejectedImgPoints,
            # cameraMatrix = cam_int,
            # distCoeffs = cam_dist
            )   
    # Only try to find CharucoBoard if we found markers
    if ids is not None:
        if len(ids) > 0:
            # Get charuco corners and ids from detected aruco markers
            charucoCorners, charucoIds, _, _ = charucoDetector.detectBoard(gray, markerCorners=corners, markerIds=ids)
            # response, charuco_corners, charuco_ids = cv.aruco.interpolateCornersCharuco(
            #         markerCorners=corners,
            #         markerIds=ids,
            #         image=gray,
            #         board=charucoDetector.getBoard())
            return charucoCorners, charucoIds
    return np.array([]), np.array([])

def collectCalibrationData(img, arucoDetector, charucoDetector):
    imgScaled = cv.resize(img.copy(), (int(img.shape[1]/2), int(img.shape[0]/2)))
    cv.imshow("camera", imgScaled)

    gray = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)
    charucoCorners, charucoIds = detectCharucoCorners(gray, arucoDetector, charucoDetector)

    patternWasFound = False
    if charucoIds is not None:
        if len(charucoIds) == 35:
            patternWasFound = True

    if charucoCorners is None:
        return None, patternWasFound
    
    img = cv.drawChessboardCorners(img, (7, 5), charucoCorners, patternWasFound)
    imgScaled = cv.resize(img.copy(), (int(img.shape[1]/2), int(img.shape[0]/2)))
    cv.imshow("camera", imgScaled)

    charucoCorners = np.array([point[0] for point in charucoCorners])
    return charucoCorners, patternWasFound

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

def initCharucoDetector():
    boardPhoto = cv.imread("./Overige Images/board.jpg")

    dictionaries = findDict(cv.cvtColor(boardPhoto, cv.COLOR_BGR2GRAY))
    dictionary = cv.aruco.getPredefinedDictionary(dictionaries[1])

    boardSize = (8, 6) # aantal vakjes (hor, ver)
    squareLength = 0.03 # in meter
    markerLength = 0.015 # in meter
    charucoBoard = cv.aruco.CharucoBoard(boardSize, squareLength, markerLength, dictionary)

    charucoDetector = cv.aruco.CharucoDetector(charucoBoard)

    return charucoDetector

def initArucoDetector():
    boardPhoto = cv.imread("./Overige Images/board.jpg")

    dictionaries = findDict(cv.cvtColor(boardPhoto, cv.COLOR_BGR2GRAY))
    dictionary = cv.aruco.getPredefinedDictionary(dictionaries[1])

    arucoDetector = cv.aruco.ArucoDetector(dictionary)
    return arucoDetector


def main():
    cap = cv.VideoCapture(0, apiPreference=cv.CAP_ANY, params=[cv.CAP_PROP_FRAME_WIDTH, 1920, cv.CAP_PROP_FRAME_HEIGHT, 1080])
    _, img = cap.read()
    imgSize = (img.shape[1], img.shape[0])
    
    # Debug print
    print("imgSize:")
    print(imgSize)
    
    charucoDetector = initCharucoDetector()
    arucoDetector = initArucoDetector()

    objPoints = np.zeros((5*7,3), np.float32)
    objPoints[:,:2] = 4.43 *np.mgrid[0:7,0:5].T.reshape(-1,2)

    allCharucoCorners = []
    allObjPoints = []

    while True:
        succes, img = cap.read()
        if not succes:
            continue
        
        charucoCorners, patternWasFound = collectCalibrationData(img, arucoDetector, charucoDetector)
        if charucoCorners is None:
            continue
        
        key = cv.waitKey(1)

        if key == 13 and patternWasFound: # ENTER
            allCharucoCorners.append(charucoCorners)
            allObjPoints.append(objPoints)

            print("Images collected: " + str(len(allCharucoCorners)))

        elif key == 99 and len(allCharucoCorners) >= 20: # c
            break
    
    print("Calibrating camera...")
    retval, cameraMatrix, distCoeffs, _, _ = cv.calibrateCamera(allObjPoints, allCharucoCorners, imgSize, None, None)
    
    # Debug prints
    print("reprojectionError:")
    print(retval)
    print("-----------------------------")
    print("cameraMatrix:")
    print(cameraMatrix)
    print("-----------------------------")
    print("distCoeffs:")
    print(distCoeffs)
    print("-----------------------------")

    saveCameraCalibration("cameraSchoolDepthDefect", cameraMatrix, distCoeffs)

if __name__ == "__main__":
    main()