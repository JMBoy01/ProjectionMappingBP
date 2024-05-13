import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def loadCameraCalibration(filename):
    fs = cv.FileStorage(filename, cv.FILE_STORAGE_READ)
    cameraMatrix = fs.getNode("cameraMatrix").mat()
    distCoeffs = fs.getNode("distCoeffs").mat()
    rvecs = fs.getNode("rvecs").mat()
    tvecs = fs.getNode("tvecs").mat()
    fs.release()
    return cameraMatrix, distCoeffs, rvecs, tvecs

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

def makeInitCirclePattern(charucoDetector):
    boardImg = cv.imread("board_img.jpg")
    charucoCorners, _, _, _ = charucoDetector.detectBoard(boardImg)
    charucoCorners = np.array(charucoCorners, dtype=np.int32)
    # print("corners: " + str(charucoCorners))

    boardObjPoints = charucoDetector.getBoard().getChessboardCorners()
    circleObjPoints = []

    scaleFactor = 4

    # Debug prints
    # print("boardObjPoints:")
    # print(boardObjPoints)

    patternImg = np.zeros((1080, 1920))
    for i in range(0, len(charucoCorners), 2):
        skips = [6, 20, 34]
        if i in skips:
            continue
    
        patternImg = cv.circle(patternImg, charucoCorners[i][0], 40, 255, -1)
        # print(boardObjPoints[i])
        circleObjPoints.append(boardObjPoints[i])

    # cv.imwrite("patternImg.jpg", patternImg)

    patternImg = cv.resize(patternImg, (int(patternImg.shape[1]/scaleFactor), int(patternImg.shape[0]/scaleFactor)))

    # plt.imshow(patternImg)
    # plt.show()
        
    img = np.zeros((768, 1024), dtype=np.uint8)
    x = int((img.shape[1] - patternImg.shape[1])/2)
    y = int((img.shape[0] - patternImg.shape[0])/2)

    x_start = x
    x_end = x + patternImg.shape[1]
    y_start = y
    y_end = y + patternImg.shape[0]

    # Place the smaller image onto the larger image
    img[y_start:y_end, x_start:x_end] = patternImg

    circleCenters = []
    for i in range(0, len(charucoCorners), 2):
        skips = [6, 20, 34]
        if i in skips:
            continue

        corner = charucoCorners[i][0]/scaleFactor
        circleCenters.append([int(corner[0] + x), int(corner[1] + y)])
    
    # Debug prints:
    print("circleCenters:")
    print(np.array(circleCenters, dtype=np.float32))

    # cv.imshow("patternImg", img)
    plt.imshow(img)
    plt.show()

    return img, np.array(circleObjPoints, dtype=np.float32), np.array(circleCenters, dtype=np.float32)

def getCalibrationSurfacePose(charucoDetector, img, cameraMatrix, distCoeffs):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    charucoImgPoints, charucoIds, _, _  = charucoDetector.detectBoard(gray)
    if charucoIds is None:
        return None

    board = charucoDetector.getBoard()
    allObjPoints = board.getChessboardCorners()
    objPoints = []

    for id in charucoIds:
        objPoints.append(allObjPoints[id])

    # Debug prints
    # print("objPoints:")
    # print(objPoints)
    # print("charucoImgPoints:")
    # print(charucoImgPoints)
    # print("len(objPoints):")
    # print(len(objPoints))
    # print("charucoIds:")
    # print(charucoIds)
    # print("len(charucoImgPoints):")
    # print(len(charucoImgPoints))

    if len(charucoIds) < 4:
        return None

    retval, rvecs, tvec = cv.solvePnP(np.array(objPoints, dtype=np.float32), np.array(charucoImgPoints, dtype=np.float32), cameraMatrix, distCoeffs)
    if retval:
        transformationMatrix = constructTransformationMatrix(rvecs, tvec)
        return transformationMatrix
    else:
        return None

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

def decomposeTransformationMatrix(transfMat):
    R = transfMat[:3, :3]
    T = transfMat[:3, 3]
    return R, T

def saveCharucoBoard(board, size, filename):
    generatedBoardImg = board.generateImage(size)
    cv.imwrite(filename, generatedBoardImg)

def calculateEssMatCamProj(centersCam, centersProj, cameraMatrix, distCoeffs, projMatrix, projDistCoeffs):
    # Debug prints
    # print("centersProj:")
    # print(centersProj)
    # print("centersCam:")
    # print(centersCam)
    # print("objPoints:")
    # print(objPoints)    

    if len(centersCam) or centersCam is not None:
        retval, essMatCamProj, R, t, _ = cv.recoverPose(centersCam, centersProj, cameraMatrix, distCoeffs, projMatrix, projDistCoeffs, method=cv.RANSAC, prob=.99999, threshold=1)
        if retval:
            transfMatCamProj = constructTransformationMatrix(R, t)

            # Debug prints
            print("trasnfMatCamProj:")
            print(transfMatCamProj)

            visualizeCamProj(R, t, centersCam, centersProj, cameraMatrix, projMatrix)

            return essMatCamProj, transfMatCamProj
        else:
            print("No return value for recoverPose...")
            return None, None
    else:
        print("No centers found...")
        return None, None

def detectAndDrawCirclesPatternFind(img, blobDetector):
    patternSize = (3, 5) # 3 horizontaal, 5 vertical mag 1 schuin geteld worden
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

def detectAndDrawCirclesPatternHough(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)
    binary = cv.GaussianBlur(binary, (7, 7), 2)
    circles = cv.HoughCircles(binary, cv.HOUGH_GRADIENT, 1, 3, param1=50, param2=30, minRadius=0, maxRadius=15)

    centersCam = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        circles_sorted = sorted(circles[0], key=lambda c: (c[1], c[0]))

        # Debug print
        # print("circles_sorted:")
        # print(circles_sorted)

        count = 0
        for circle in circles_sorted:
            x, y, radius = circle

            # draw the outer and circle + number
            cv.circle(img, (x, y), radius, (0,255,0), 2)
            cv.circle(img, (x, y), 0, (0,0,255), 3)
            cv.putText(img, str(count), (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

            centersCam.append([x, y])

            count += 1

        return img, binary, centersCam
    else:
        return img, binary, None

def visualizeCamProj(R, t, centersCam, centersProj, cameraMatrix, projMatrix):
    # transfMatCamProj = constructTransformationMatrix(R, T)
    # transfMatProjCam = np.linalg.inv(transfMatCamProj)

    # projPos = np.array([0, 0, 0], dtype=np.float32)
    
    # camPos = cv.convertPointsToHomogeneous(np.array([projPos]))[0, 0]
    # print(camPos)
    # camPos = (transfMatProjCam @ camPos.T).T
    # print(camPos)
    # camPos = cv.convertPointsFromHomogeneous(np.array([camPos]))[0, 0]

    transfMatCamProj = constructTransformationMatrix(R, t)

    camPos = np.array([0, 0, 0], dtype=np.float32)
    
    projPos = cv.convertPointsToHomogeneous(np.array([camPos]))[0, 0]
    print(projPos)
    projPos = (transfMatCamProj @ projPos.T).T
    print(projPos)
    projPos = cv.convertPointsFromHomogeneous(np.array([projPos]))[0, 0]

    # Debug prints
    print("camPos:")
    print(camPos)
    print("projPos:")
    print(projPos)

    points3D = [camPos, projPos]

    # source: https://www.opencvhelp.org/tutorials/advanced/reconstruction-opencv/
    # R, T = decomposeTransformationMatrix(transfMatProjCam)
    # T = [[T[0]], [T[1]], [T[2]]]

    # Debug prints
    # print("R:")
    # print(R)
    # print("t:")
    # print(t)

    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, t))

    # Convert the projection matrices to the camera coordinate system
    P1 = cameraMatrix @ P1
    P2 = projMatrix @ P2

    # Debug prints
    # print("centersCam.shape:")
    # print(np.array(centersCam).shape)
    # print("centersProj.shape:")
    # print(np.array(centersProj).shape)s

    points4D = cv.triangulatePoints(P1, P2, centersCam.T, centersProj.T)
    points3DCircles = points4D / points4D[3]  # Convert from homogeneous to Cartesian coordinates
    points3DCircles = points3DCircles[:3, :].T
    
    colors = ['b', 'r']

    for i in range(len(points3DCircles)):
        points3D.append(points3DCircles[i])
        colors.append('k')
    
    points3D = np.array(points3D)
    colors = np.array(colors)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D points
    ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2], marker='o', s=5, c=colors, alpha=0.5)

    # Set scale of axis to be equal
    ax.axis('equal')

    # Configure the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def askCirclePatternDetectionMethod():
    while True:
        print("What algorithm do you want to use?")
        print("(1) findCirclesGrid [STABLE, RECOMMENDED]")
        print("(2) HoughCircles [UNSTABLE]")
        method = input("Type the number of the algorithm you want to use: ")
        if method == '1':
            print("Method: findCirclesGrid")
            break
        elif method == '2':
            print("Method: HoughCircles")
            break
        else:
            print("Invalid input...")

    return method

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

def makeMaskDynamicCal(img, patternImg, objPoints, transfMat, essMat, cameraMatrix, distCoeffs):
    R, T = decomposeTransformationMatrix(transfMat)
    imgPoints, _ = cv.projectPoints(np.array(objPoints), R, T, cameraMatrix, distCoeffs)

    # Debug prints
    print("imagePoints:")
    print(imgPoints)

    mask = np.zeros((img.shape[0], img.shape[1]))
    mask = cv.fillConvexPoly(mask, np.array(imgPoints, dtype=np.int32), 255)

    x, y, w, h = cv.boundingRect(np.array(imgPoints, dtype=np.int32))
    patternImg = cv.warpPerspective(patternImg, R, (w, h))

    mask_8u = mask.astype(np.uint8)
    mask_rgb = cv.cvtColor(mask_8u, cv.COLOR_GRAY2BGR)

    # Debug prints
    print("mask_rgb.shape:")
    print(mask_rgb.shape)
    print("patternImg.shape:")
    print(patternImg.shape)

    # cv.copyTo(patternImg, mask_rgb[y:y+h, x:x+w])

    return mask

    # Convert img to proj space
    # maskHom = cv.convertPointsToHomogeneous(maskHom)
    # maskProjHom = essMat @ maskHom
    # maskProj = cv.convertPointsFromHomogeneous(maskProjHom)

    # return maskProj

def getCalPatternObjPts(charucoDetector):
    board = charucoDetector.getBoard()
    boardObjPoints = board.getChessboardCorners()

    minX = min(p[0] for p in boardObjPoints)
    minY = min(p[1] for p in boardObjPoints)
    maxX = max(p[0] for p in boardObjPoints)
    maxY = max(p[1] for p in boardObjPoints)

    deltaX = maxX + 2 * minX
    deltaY = maxY + 2 * minY

    objPoints = [[0, deltaY + minY, 0], [deltaX, deltaY + minY, 0], [deltaX, 2*deltaY + minY, 0], [0, 2*deltaY + minY, 0]]

    # Debug prints
    print("calPatternObjPts:")
    print(objPoints)

    return objPoints

def saveProjCalibration(filename, projMatrix, projDistCoeffs):
    fs = cv.FileStorage(filename, cv.FILE_STORAGE_WRITE)
    fs.write("projMatrix", projMatrix)
    fs.write("projDistCoeffs", projDistCoeffs)

    fs.release()
    print("Matrices opgeslagen in", filename)

def saveEssAndTransfMat(filename, essMat, transfMat):
    fs = cv.FileStorage(filename, cv.FILE_STORAGE_WRITE)
    fs.write("essMat", essMat)
    fs.write("transfMat", transfMat)

    fs.release()
    print("Calibratiegegevens opgeslagen in", filename)

def convertPointsCamToProjSpace(centersCam, essMatCamProj):
    calcCentersProj = []

    homCalcCentersProj = cv.convertPointsToHomogeneous(centersCam)
    homCalcCentersProj = np.squeeze(homCalcCentersProj)
    # Debug prints
    print("centersCam:")
    print(centersCam)
    print("homCalcCentersProj:")
    print(homCalcCentersProj)
    print("---------------------------------------------------------------------")

    for homCalcCenter in homCalcCentersProj:
        point = (essMatCamProj @ homCalcCenter.T).T
        calcCentersProj.append(point)

        # Debug prints
        print("homCalcCenter:")
        print(homCalcCenter)
        print("point:")
        print(point)
    
    calcCentersProj = np.squeeze(calcCentersProj)
    print("calcCentersProj:")
    print(calcCentersProj)
    calcCentersProj = cv.convertPointsFromHomogeneous(calcCentersProj)
    calcCentersProj = np.squeeze(calcCentersProj)
    # imgPoints.append(calcCentersProj)

    # Debug prints
    print("---------------------------------------------------------------------")
    print("calcCentersProj:")
    print(calcCentersProj)

    return calcCentersProj

def collectCalibrationDataProj(img, blobDetector, allCentersCam, objPoints, allObjPoints):
    img, mask, centersCam = detectAndDrawCirclesPatternFind(img, blobDetector)

    cv.imshow("camera", img)
    cv.imshow("mask", mask)

    if centersCam is None:
        print("Value of centersCam is None")
        return img, mask
    elif len(centersCam) != 15:
        print("Detected amount of centers: " + str(len(centersCam)) + "; expected: 15")
        return img, mask
    
    key = cv.waitKey(1)
    if key == 13:
        allCentersCam.append(centersCam)
        allObjPoints.append(objPoints)

        # Debug prints
        print("len(allCentersCam):")
        print(len(allCentersCam))
        # print("allCentersCam:")
        # print(allCentersCam)
    
    return img, mask

def correctObjPoints(objPoints, charucoBoard):
    while True:
        print("Measure the distance between 2 circle centers and give value in meters...")
        distance = input("Distance (in meter): ")
        try:
            distance = float(distance)
            break
        except ValueError:
            print("Not a valid input, give the distance in meters and use a . instead of ,")

    scale = (distance/2)/charucoBoard.getSquareLength()

    # Debug print
    print("scale:")
    print(scale)

    return objPoints*scale

def getCentersProjectionPlane(img, blobDetector, charucoDetector):
    cv.imshow("camera", img)

    gray = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)
    charucoCorners, charucoIds, markerCorners, markerIds = charucoDetector.detectBoard(gray)

    if charucoCorners is None:
        # print("charucoCorners is None")
        return None, None

    charucoCorners = np.array([point[0] for point in charucoCorners])

    objPoints = charucoDetector.getBoard().getChessboardCorners()
    objPoints2D = [[point[0], point[1]] for point in objPoints]

    if len(charucoCorners) != len(objPoints2D):
        # print("len(charucoCorners) = " + str(len(charucoCorners)) + "; expected = " + str(len(objPoints2D)))
        return None, None

    charucoSize = charucoDetector.getBoard().getChessboardSize()
    img = cv.drawChessboardCorners(img, charucoSize, charucoCorners, False)
    cv.imshow("camera", img)

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

    H, _ = cv.findHomography(charucoCorners, np.array(objPoints2D))

    img, mask, centersCam = detectAndDrawCirclesPatternFind(img, blobDetector)

    if centersCam is None:
        return None, None

    cv.imshow("camera", img)

    centersConvertedHmg = []
    centersCamHmg = cv.convertPointsToHomogeneous(np.array(centersCam, np.float32))
    for center in centersCamHmg:
        centerConverted = (H @ center.T).T
        centersConvertedHmg.append(centerConverted)

    centersConverted = cv.convertPointsFromHomogeneous(np.array(centersConvertedHmg))
    centersConverted = np.array([point[0] for point in centersConverted])
    return centersConverted, centersCam

def makeCirclePatternImages(size = (768, 1024), spaceBetween = 50, circleSize = 25):
    patternImages = []
    for itr in range(0, 5, 1):
        start_x = 0
        start_y = 0

        match itr:
            case 0:
                start_x += spaceBetween
                start_y += spaceBetween
            case 1:
                start_x = size[1] - 3 * 2 * spaceBetween
                start_y += spaceBetween
            case 2:
                start_x = size[1] - 3 * 2 * spaceBetween
                start_y = size[0] - 5 * spaceBetween
            case 3:
                start_x += spaceBetween
                start_y = size[0] - 5 * spaceBetween
            case 4:
                start_x = size[1]/2 - 3 * spaceBetween
                start_y = size[0]/2 - 2 * spaceBetween
        
        img = np.zeros(size, dtype=np.uint8)
        for i in range(0, 5, 1):
            for j in range(0, 3, 1):
                x = int(start_x + 2 * j * spaceBetween + (i % 2) * spaceBetween)
                y = int(start_y + i * spaceBetween)
                img = cv.circle(img, (x, y), circleSize, 255, -1)

        patternImages.append(img)

        cv.imwrite('./patternImages/patternImg' + str(itr) + ".png", img)
    return patternImages

def main():
    cap = cv.VideoCapture(2)

    # Debug prints
    # succes, img = cap.read()
    # print("camera shape:")
    # print(img.shape)

    boardPhoto = cv.imread("./Overige Images/board.jpg")

    cameraMatrix, distCoeffs, _, _ = loadCameraCalibration("cameraSchoolDepthDefect")

    dictionaries = findDict(cv.cvtColor(boardPhoto, cv.COLOR_BGR2GRAY))
    dictionary = cv.aruco.getPredefinedDictionary(dictionaries[1])

    boardSize = (8, 6) # aantal vakjes (hor, ver)
    squareLength = 0.03 # in meter
    markerLength = 0.015 # in meter
    charucoBoard = cv.aruco.CharucoBoard(boardSize, squareLength, markerLength, dictionary)

    # saveCharucoBoard(charucoBoard, (1920, 1080), "board_img.jpg")

    charucoDetector = cv.aruco.CharucoDetector(charucoBoard)

    # patternImg, objPoints, centersProj = makeInitCirclePattern(charucoDetector)
    patternImages = makeCirclePatternImages()

    # method = askCirclePatternDetectionMethod()

    blobDetector = initBlobDetector()

    # ----------------------------------------------------------------------------------------------------------------------------------

    # NIEUWE CODE (MET UITLEG VAN JONI)

    print("Take 4 images of every pattern, there are 5 different patterns in total.")

    allCentersProjPlane = []
    allCentersProj = []

    patternImgIndex = 0
    while True:
        # Change patternImg every 4 times data has been collected
        if len(allCentersProj) % 4 == 0 and len(allCentersProj) != 0:
            patternImgIndex += 1

        patternImg = patternImages[patternImgIndex]
        cv.imshow("patternImg", patternImg)

        _, img = cap.read()
        img = cv.undistort(img, cameraMatrix, distCoeffs)

        centersProjPlane, _ = getCentersProjectionPlane(img, blobDetector, charucoDetector)
        _, _, centersProj = detectAndDrawCirclesPatternFind(patternImg, blobDetector)

        key = cv.waitKey(1)

        if key == 13 and centersProjPlane is not None:
            centersProjPlane = np.array([[point[0], point[1], 0] for point in centersProjPlane], np.float32)
            allCentersProjPlane.append(centersProjPlane)

            allCentersProj.append(centersProj)

            print("Images collected: " + str(len(allCentersProjPlane)))
        elif key == 99 and len(allCentersProjPlane) >= 20:
            break


    # Convert to 3D points to use as objPoints
    

    # Debug prints
    print("allCentersProjPlane:")
    print(allCentersProjPlane)
    print("--------------------------")
    print("allCentersProj:")
    print(allCentersProj)
    print("--------------------------")


    patternImgSize = (patternImg.shape[1], patternImg.shape[0])
    retval, projMatrix, projDistCoeffs, _, _ = cv.calibrateCamera(allCentersProjPlane, allCentersProj, patternImgSize, None, None)
    
    if not retval:
        print("Could not calibrate projector, try again...")
        print("Exiting program...")
        exit(0)

    saveProjCalibration("projSchool", projMatrix, projDistCoeffs)

    # Debug prints
    print("centersProjPlane:")
    print(centersProjPlane)
    print("-------------------------------")
    print("projMatrix:")
    print(projMatrix)
    print("-------------------------------")
    print("projDistCoeffs:")
    print(projDistCoeffs)
    print("-------------------------------")

    print("Put your camera in the set position to calibrate the extrinsic parameters.")
    print("Press ENTER when you have a good image...")

    allCentersProjPlane = []
    allCentersProj = []
    allCentersCam = []

    patternImgIndex = 0

    while True:
        if len(allCentersProj) % 4 == 0 and len(allCentersProj) != 0:
            patternImgIndex += 1

        patternImg = patternImages[patternImgIndex]
        cv.imshow("patternImg", patternImg)

        _, img = cap.read()
        img = cv.undistort(img, cameraMatrix, distCoeffs)

        centersProjPlane, centersCam = getCentersProjectionPlane(img, blobDetector, charucoDetector)
        _, _, centersProj = detectAndDrawCirclesPatternFind(patternImg, blobDetector)

        key = cv.waitKey(1)

        if key == 13 and centersProjPlane is not None:
            centersProjPlane = np.array([[point[0], point[1], 0] for point in centersProjPlane], np.float32)
            allCentersProjPlane.append(centersProjPlane)

            allCentersProj.append(centersProj)

            allCentersCam.append(centersCam)

            print("Images collected: " + str(len(allCentersProjPlane)))
        elif key == 99 and len(allCentersProjPlane) >= 15:
            break
        else:
            "centersProjPlane = None"

    retval, _, _, _, _, R, T, E, F = cv.stereoCalibrate(allCentersProjPlane, allCentersCam, allCentersProj, cameraMatrix, distCoeffs, projMatrix, projDistCoeffs, patternImgSize)

    if not retval:
        print("Could not stereo calibrate camera and projector, try again...")
        print("Exiting program...")
        exit(0)
    
    transfMat = constructTransformationMatrix(R, T)

    # Debug prints
    print("essMat:")
    print(E)
    print("transfMat:")
    print(transfMat)

    saveEssAndTransfMat("essAndTransfMatCamProjSchool", E, transfMat)

    # visualizeCamProj(R, T, centersCam, centersProj, cameraMatrix, projMatrix)

if __name__ == "__main__":
    main()