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
    
        patternImg = cv.circle(patternImg, charucoCorners[i][0], 75, 255, -1)
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

    cv.imshow("circle grid pattern", img)
    # plt.imshow(img)
    # plt.show()

    return img, np.array(circleObjPoints, dtype=np.float32), circleCenters

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

def calculateEssMatCamProj(centersCam, centersProj, cameraMatrix):
    # Debug prints
    # print("centersProj:")
    # print(centersProj)
    # print("centersCam:")
    # print(centersCam)
    # print("objPoints:")
    # print(objPoints)    

    if len(centersCam) or centersCam is not None:
        essMatCamProj, _ = cv.findEssentialMat(np.array(centersCam, dtype=np.float32), np.array(centersProj, dtype=np.float32), cameraMatrix, method=cv.RANSAC, prob=.99999, threshold=.1)
        if essMatCamProj is not None:
            # Debug prints
            print("essMatCamProj:")
            print(essMatCamProj)
            
            retval, R, T, _ = cv.recoverPose(essMatCamProj, np.array(centersCam, dtype=np.float32), np.array(centersProj, dtype=np.float32), cameraMatrix)
            if not retval:
                print("Wasn't able to recover pose...")
                return essMatCamProj, None
            transfMatCamProj = constructTransformationMatrix(R, T)

            # Debug prints
            print("trasnfMatCamProj:")
            print(transfMatCamProj)

            visualizeCamProj(R, T, centersCam, centersProj, cameraMatrix)

            return essMatCamProj, transfMatCamProj
        else:
            print("No return value for stereoCalibrate...")
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
    
    img = cv.drawChessboardCorners(img, patternSize, centersCam, patternWasFound)

    if patternWasFound:
        centersCam = [point[0] for point in centersCam]
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

def visualizeCamProj(R, T, centersCam, centersProj, cameraMatrix):
    transfMatCamProj = constructTransformationMatrix(R, T)

    camPos = np.array([0, 0, 0], dtype=np.float32)
    
    projPos = cv.convertPointsToHomogeneous(np.array([camPos]))[0, 0]
    projPos = (transfMatCamProj @ projPos.T).T
    projPos = cv.convertPointsFromHomogeneous(np.array([projPos]))[0, 0]

    # Debug prints
    print("camPos:")
    print(camPos)
    print("projPos:")
    print(projPos)

    points3D = [camPos, projPos]

    # source: https://www.opencvhelp.org/tutorials/advanced/reconstruction-opencv/
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, T))

    # Convert the projection matrices to the camera coordinate system
    P1 = cameraMatrix @ P1
    P2 = cameraMatrix @ P2

    # Debug prints
    # print("centersCam.shape:")
    # print(np.array(centersCam).shape)
    # print("centersProj.shape:")
    # print(np.array(centersProj).shape)s

    points4D = cv.triangulatePoints(P1, P2, np.array(centersCam, dtype=np.float32).T, np.array(centersProj, dtype=np.float32).T)
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
    print("Calibratiegegevens opgeslagen in", filename)

def main():
    cap = cv.VideoCapture(2)
    # succes, img = cap.read()

    boardPhoto = cv.imread("board.jpg")

    cameraMatrix, distCoeffs, rvecs, tvecs = loadCameraCalibration("cameraSchoolDepthDefect")

    dictionaries = findDict(cv.cvtColor(boardPhoto, cv.COLOR_BGR2GRAY))
    dictionary = cv.aruco.getPredefinedDictionary(dictionaries[1])

    boardSize = (8, 6) # aantal vakjes (hor, ver)
    squareLength = 0.03 # in meter
    markerLength = 0.015 # in meter
    charucoBoard = cv.aruco.CharucoBoard(boardSize, squareLength, markerLength, dictionary)

    # saveCharucoBoard(charucoBoard, (1920, 1080), "board_img.jpg")

    charucoDetector = cv.aruco.CharucoDetector(charucoBoard)

    patternImg, objPoints, centersProj = makeInitCirclePattern(charucoDetector)

    method = askCirclePatternDetectionMethod()

    blobDetector = initBlobDetector()

    img = None
    imgPoints = []
    print("It's recommended to collect at least 20 images with different camera positions.")
    print("Press enter when you are ready...")
    while True:
        print("Collected image amount: " + str(len(imgPoints)))
        _, img = cap.read()

        if method == '1':
            img, binary, centersCam = detectAndDrawCirclesPatternFind(img.copy(), blobDetector)
        elif method == '2':
            img, binary, centersCam = detectAndDrawCirclesPatternHough(img.copy())
        
        cv.imshow("camera - press ENTER to take antorher image - press C to calibrate with the collected data", img)
        cv.imshow("binary", binary)

        key = cv.waitKey(1)
        if key == 13 and len(centersCam) == len(centersProj):
            essMatCamProj, _ = calculateEssMatCamProj(centersCam, centersProj, cameraMatrix)
            if essMatCamProj is None:
                print("essMatCamProj == None")
                continue
            
            calcCentersProj = []
            homCalcCentersProj = cv.convertPointsToHomogeneous(np.array(centersCam))
            for homCalcCenter in homCalcCentersProj:
                calcCentersProj.append(essMatCamProj @ homCalcCenter)
            calcCentersProj = cv.convertPointsFromHomogeneous(calcCentersProj)
            imgPoints.append(calcCentersProj)
            # break
        elif key == 13: # ENTER
            print("len(centersCam): " + str(len(centersCam)) + ", len(centersProj): " + str(len(centersProj)))
            print("Not the same amount of points in centersCam as centersProj...")
        elif key == 99: # C
            retval, projMatrix, projDistCoeffs, _, _ = cv.calibrateCamera(objPoints, imgPoints, (768, 1024), None, None)
            if retval:
                saveProjCalibration("projHome", projMatrix, projDistCoeffs)
                break
            else:
                print("Camera calibration failed, try again...")

    cv.destroyAllWindows()
    # ----------------------------------------------------------------------------------------------------------------------------------

    # Dit is niet meer echt nodig omdat we enkel intrinsieke calibratie doen
    # essMatCamProj, transfMatCamProj = calculateEssMatCamProj(centersCam, centersProj, cameraMatrix)
    # if essMatCamProj is None:
    #     print("essMatCamProj == None")
    #     return

    # ----------------------------------------------------------------------------------------------------------------------------------
    
    # Dynamic calibration stuff

    # blackImg = np.zeros((768, 1024), dtype=np.uint8)
    # cv.imshow("black", blackImg)

    # patternImg = cv.imread("patternImg.jpg")

    # while True:
    #     _, img = cap.read()
        
    #     cv.imshow("cam", img)
    #     cv.waitKey(1)
    #     transformationMatrix = getCalibrationSurfacePose(charucoDetector, img, cameraMatrix, distCoeffs)
    #     if transformationMatrix is None:
    #         print("transformationMatrix == None")
    #         continue

    #     R, _ = decomposeTransformationMatrix(transformationMatrix)
    #     # cv.waitKey(0)

    #     maskProj = makeMaskDynamicCal(img, patternImg, getCalPatternObjPts(charucoDetector), transformationMatrix, essMatCamProj, cameraMatrix, distCoeffs)
    #     cv.imshow("maskProj", maskProj)

if __name__ == "__main__":
    main()