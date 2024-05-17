import cv2 as cv
import numpy as np
import os

# Source Joni
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

def detectAndDrawCirclesPatternFind(img, blobDetector, isImgPatternImg):
    patternSize = (5, 7) # 4 horizontaal, 9 vertical mag 1 schuin geteld worden

    binary = img.copy()
    if not isImgPatternImg:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, binary = cv.threshold(gray, 225, 255, cv.THRESH_BINARY)
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
    
    for i in range(0, fileAmount+1, 1):
        img = cv.imread(path + fileStructerName + str(i) + fileExtension)
        if img is None:
            continue

        imgList.append(img)
    print(len(imgList))
    return imgList

def getCentersProjectionPlane(img, blobDetector, charucoDetector, arucoDetector):
    scale = 2
    imgResized = cv.resize(img.copy(), (int(img.shape[1]/scale), int(img.shape[0]/scale)))
    cv.imshow("camera", imgResized)

    gray = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)
    # gray = cv.equalizeHist(gray)
    charucoCorners, charucoIds = detectCharucoCorners(gray, arucoDetector, charucoDetector)

    if charucoCorners is None:
        print("charucoCorners is None")
        return None, None

    charucoCorners = np.array([point[0] for point in charucoCorners])
    charucoSize = (7,5)

    objPoints = np.zeros((5*7,3), np.float32)
    objPoints[:,:2] = 4.43 *np.mgrid[0:7,0:5].T.reshape(-1,2)

    if len(charucoCorners) != len(objPoints):
        # print("len(charucoCorners) = " + str(len(charucoCorners)) + "; expected = " + str(len(objPoints2D)))
        return None, None

    img = cv.drawChessboardCorners(img, charucoSize, charucoCorners, True)

    imgResized = cv.resize(img.copy(), (int(img.shape[1]/scale), int(img.shape[0]/scale)))
    cv.imshow("camera", imgResized)

    # Debug prints
    # print("charucoCorners:")
    # print(charucoCorners)
    # print("-------------------------")
    # print("objPoints:")
    # print(objPoints)
    # print("-------------------------")
    # print("objPoints2D:")
    # print(np.array(objPoints2D))
    # print("-------------------------")

    H, _ = cv.findHomography(charucoCorners, np.array(objPoints))

    img, mask, centersCam = detectAndDrawCirclesPatternFind(img, blobDetector, False)

    if centersCam is None:
        return None, None

    imgResized = cv.resize(img.copy(), (int(img.shape[1]/scale), int(img.shape[0]/scale)))
    cv.imshow("camera", imgResized)
    # cv.imshow("mask", mask)

    centersConvertedHmg = []
    centersCamHmg = cv.convertPointsToHomogeneous(np.array(centersCam, np.float32))
    for center in centersCamHmg:
        centerConverted = (H @ center.T).T
        centersConvertedHmg.append(centerConverted)

    centersConverted = cv.convertPointsFromHomogeneous(np.array(centersConvertedHmg))
    centersConverted = np.array([point[0] for point in centersConverted])
    centersConverted = np.array([[point[0], point[1], 0] for point in centersConverted], np.float32)
    return centersConverted, centersCam

def drawDetectionsAndSave(imgList, patternImgList, charucoDetector, blobDetector, arucoDetector, path, fileStructureName):
    if path[-1] != '/':
        path += '/'

    allCircleCentersProjPlane = []
    allCircleCentersCam = []
    allPatternCenters = []

    count = 0
    patternImgIndex = 0

    while True:
        if count >= len(imgList):
            break

        img = imgList[count].copy()

        patternImg = patternImgList[patternImgIndex].copy()

        circleCentersProjPlane, circleCentersCam = getCentersProjectionPlane(img, blobDetector, charucoDetector, arucoDetector)

        patternImg, _, patternCenters = detectAndDrawCirclesPatternFind(patternImg, blobDetector, True)

        key = cv.waitKey(1000)

        # ENTER to get the next image
        if key == 13 and circleCentersProjPlane is not None and circleCentersCam is not None and patternCenters is not None:
            allCircleCentersProjPlane.append(circleCentersProjPlane)
            allCircleCentersCam.append(circleCentersCam)
            allPatternCenters.append(patternCenters)
            
            size = (patternImgList[0].shape[0], patternImgList[0].shape[1])
            retval, projMatrix, projDistCoeffs, _, _ = cv.calibrateCamera(allCircleCentersProjPlane, allPatternCenters, size, None, None)

            # Debug prints
            # print("allPatternCenters:")
            # print(allPatternCenters)
            # print("projected_image_points:")
            # print(projected_image_points)
            # print("------------------------------------")

            # Debug prints
            print("img number: " + str(count))
            print("pattern img number: " + str(patternImgIndex + 1))
            print("RMS: " + str(retval))
            print("------------------------------------")
            
            count += 1
            if count % 4 == 0 and count != 0:
                patternImgIndex += 1

            cv.imwrite(path + fileStructureName + str(count) + ".png", img)
            cv.imwrite("./projCalibrationData/Home/detectedPattern/detectedPatternImg" + str(patternImgIndex) + ".png", patternImg)
        elif key == 115: # s voor skip
            print("img number: " + str(count))
            print("pattern img number: " + str(patternImgIndex + 1))
            print("Skipped")
            print("------------------------------------")

            cv.imwrite(path + "skipped" + str(count) + ".png", img)

            count += 1
            if count % 4 == 0 and count != 0:
                patternImgIndex += 1
        else:
            # print("circleCentersProjPlane:")
            # print(circleCentersProjPlane)
            # print("circleCentersCam:")
            # print(circleCentersCam)
            # print("patternCenters:")
            # print(patternCenters)
            # print("--------------------------------------------")
            pass
    
    return allCircleCentersProjPlane, allCircleCentersCam, allPatternCenters

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

def constructTransformationMatrix(R, tvec):
    rotationMatrix = None

    if R.shape == (3, 3):
        rotationMatrix = R
    else:
        rotationMatrix, _ = cv.Rodrigues(R)
    
    transformationMatrix = np.eye(4)
    transformationMatrix[:3, :3] = rotationMatrix
    transformationMatrix[:3, 3] = tvec.flatten()
    transformationMatrix[3] = [0, 0, 0, 1]
    return transformationMatrix


def main():
    blobDetector = initBlobDetector()
    charucoDetector = initCharucoDetector()
    arucoDetector = initArucoDetector()

    print("Loading image list...")
    imgList = loadImageListFromFolder("./projCalibrationData/Home/rawData", "img", ".png")
    patternImgList = loadImageListFromFolder("./patternImages/", "patternImg", ".png")

    print("Detecting charuco board and circle pattern on images...")
    allCircleCentersProjPlane, allCircleCentersCam, allPatternCenters = drawDetectionsAndSave(imgList, patternImgList, charucoDetector, blobDetector, arucoDetector, "./projCalibrationData/Home/detectedData", "detected")
    print("Done detecting on all images...")

    print("Calibrating cam proj...")
    size = (patternImgList[0].shape[0], patternImgList[0].shape[1])
    retval, projMatrix, projDistCoeffs, _, _, _, _, perViewErrors = cv.calibrateCameraExtended(allCircleCentersProjPlane, allPatternCenters, size, None, None)

    if not retval:
        print("Could not stereo calibrate camera and projector, try again...")
        print("Exiting program...")
        exit(0)

    # Debug prints
    print("reprojectionError:")
    print(retval)
    print("-------------------------------")
    # print("perViewErrors:")
    # print(perViewErrors)
    # print("-------------------------------")
    print("projMatrix:")
    print(projMatrix)
    print("-------------------------------")
    print("projDistCoeffs:")
    print(projDistCoeffs)
    print("-------------------------------")

    print("Stereo calibrating cam proj...")
    cameraMatrix, distCoeffs = loadCameraCalibration("camHome")
    retval, _, _, _, _, R, T, E, F = cv.stereoCalibrate(allCircleCentersProjPlane, allCircleCentersCam, allPatternCenters, cameraMatrix, distCoeffs, projMatrix, projDistCoeffs, size)

    if not retval:
        print("Could not stereo calibrate camera and projector, try again...")
        print("Exiting program...")
        exit(0)
    
    transfMat = constructTransformationMatrix(R, T)

    # Debug prints
    print("essMat:")
    print(E)
    print("-------------------------------")
    print("transfMat:")
    print(transfMat)
    print("-------------------------------")

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()