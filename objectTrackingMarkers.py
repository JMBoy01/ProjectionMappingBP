import cv2 as cv
import numpy as np
import socket

def loadCameraCalibration(filename):
    fs = cv.FileStorage(filename, cv.FILE_STORAGE_READ)
    cameraMatrix = fs.getNode("cameraMatrix").mat()
    distCoeffs = fs.getNode("distCoeffs").mat()
    rvecs = fs.getNode("rvecs").mat()
    tvecs = fs.getNode("tvecs").mat()
    fs.release()
    return cameraMatrix, distCoeffs, rvecs, tvecs

def loadObjectPoints(filename):
    fs = cv.FileStorage(filename, cv.FILE_STORAGE_READ)
    
    objectPoints00 = fs.getNode("objectPoints00").mat()
    objectPoints10 = fs.getNode("objectPoints10").mat()
    objectPoints20 = fs.getNode("objectPoints20").mat()

    fs.release()
    return [objectPoints00, objectPoints10, objectPoints20]

def sendDataToUnity(socket, serverAddressPort, transformationMatrix):
    # data = iterations, rotationMatrix, tvecAvg
    data = transformationMatrix[0], transformationMatrix[1], transformationMatrix[2], transformationMatrix[3]
    data = str.encode(str(data))
    socket.sendto(data, serverAddressPort)
    # print("Data send: " + str(data))

def trackBox(img, objectPoints, cameraMatrix, distCoeffs, arucoDetector, kalman):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    markerCorners, markerIds, _ = arucoDetector.detectMarkers(gray)
    # markerCorners = np.array(markerCorners, dtype=np.float32)
    # print(str(markerCorners))

    if markerIds is None or len(markerIds) < 2:
        return img

    # neem lijsten samen
    zippedData = zip(markerIds, markerCorners)

    # sorteer op markerIds
    sortedData = sorted(zippedData, key=lambda x: x[0])

    # splits de lijsten weer op
    sortedMarkerIds, sortedMarkerCorners = zip(*sortedData)
    sortedMarkerIds = np.array(sortedMarkerIds)
    sortedMarkerCorners = np.array(sortedMarkerCorners)

    # split arrays in subarrays van surfaces
    markerCornersPerSurface = [[]]
    markerIdsPerSurface = [[]]
    under = 0
    upper = 9
    for i in range(len(sortedMarkerIds)):
        if under <= sortedMarkerIds[i] <= upper:
            # De id bevindt zich tussen het huidige bereik, dus voeg de hoek toe aan de huidige subarray
            markerCornersPerSurface[-1].append(sortedMarkerCorners[i])
            markerIdsPerSurface[-1].append(sortedMarkerIds[i])
        else:
            # De id valt buiten het huidige bereik, dus maak een nieuwe subarray aan en voeg de hoek toe
            markerCornersPerSurface.append([sortedMarkerCorners[i]])
            markerIdsPerSurface.append([sortedMarkerIds[i]])
            under += 10
            upper += 10
    
    #--------------------------------------------------------Poging 4------------------------------------------------------------------#
    allObjPoints = []
    allImgPoints = []
    
    objPoints = []
    # print("len(markerCornersPerSurface[0]): " + str(len(markerCornersPerSurface[0])))
    # Hier moet ik nog maken wat er moet gebeuren als er maar 1 vlak gevonden word
    if len(markerCornersPerSurface) == 1:
        # Als er maar 1 hoek van 1 vlak gedetecteerd word kan ik niet tracken
        if len(markerCornersPerSurface[0]) in range(0, 2):
            return None
    
        # Eerst weten welk vlak
        surfaceCorners = markerCornersPerSurface[0]
        surfaceIds = markerIdsPerSurface[0]

        # Neem de objPoints en zorg ervoor dat die in het midden ligt -> ongeveer predicten hoe die moet staan van diepte
        objPointsIndex = int(surfaceIds[0]/10)
        objPoints = objectPoints[objPointsIndex].copy()
        
        # Logica maken om te weten welke lengte ik + of - moet doen
        b = 0.39 # = x (in meter)
        h = 0.13 # = z (in meter)
        d = 0.275 # = y (in meter)
        boxSize = [b, d, h]

        zeroIndex = None
        for i in range(len(objPoints[0][0])):
            if all(subarray[i] == 0 for subarray in objPoints[0]):
                zeroIndex = i

        if zeroIndex is not None:
            for cornerObjPoints in objPoints:
                for objPoint in cornerObjPoints:
                    objPoint[zeroIndex] += boxSize[zeroIndex]/2
        else:
            return None

    index = 0
    print("markerIdsPerSurface: " + str(markerIdsPerSurface))
    for surfaceCorners in markerCornersPerSurface:
        if len(surfaceCorners) in range(0, 2):
            index += 1
            continue

        if len(objPoints) == 0:
            objPoints = objectPoints[index]

        surfaceIds = markerIdsPerSurface[index]
        print("surfaceIds: " + str(surfaceIds))
        predictedCoords = []
        if len(surfaceCorners) in range(2, 4):
            # Als er maar 2 corners gevonden worden proberen een predictie maken van de andere 2 om een betere totale positie schatting te krijgen
            imgPoints = []
            for corners in surfaceCorners:
                # print("corner[0]: " + str(corners[0]))
                for corner in corners[0]:
                    imgPoints.append(corner)
            imgPoints = np.array(imgPoints)

            test = []
            for id in surfaceIds:
                print("id: " + str(id))
                objPointIndex = id - (int(id/10)*10) - 1
                print("objPointIndex: " + str(objPointIndex))
                print("objPoints[objPointIndex]: " + str(objPoints[objPointIndex[0]]))
                for objPoint in objPoints[objPointIndex][0]:
                    test.append(objPoint)
            test = np.array(test)

            retval, rvecs, tvec = cv.solvePnP(test, imgPoints, cameraMatrix, distCoeffs)
            # print("retval: " + str(retval))
            if retval:
                predictedCoords = predictCorners(rvecs, tvec, surfaceIds, objPoints)
                # print("predictedCoords: " + str(predictedCoords))
        
        idRange = int(surfaceIds[0]/10)*10
        # Alle obj en img points in de lijsten en dezelfde volgorde zetten
        allSurfaceIds = [idRange+1, idRange+2, idRange+3, idRange+4]
        detectedIndex = 0
        predictedIndex = 0
        for id in allSurfaceIds:
            cornerIndex = id - idRange - 1
            allObjPoints.append(objPoints[cornerIndex][cornerIndex])

            if id in surfaceIds:
                allImgPoints.append(surfaceCorners[detectedIndex][0][cornerIndex])
                detectedIndex += 1
            else:
                # print("predictedIndex: " + str(predictedIndex))
                allImgPoints.append(convertTo2DCoord(predictedCoords[predictedIndex], cameraMatrix, distCoeffs))
                predictedIndex += 1
        
        index += 1

    allObjPoints = np.array(allObjPoints)
    allImgPoints = np.array(allImgPoints)

    # print("len(allObjPoints): " + str(len(allObjPoints)) + "\nallObjPoints: " + str(allObjPoints))
    # print("len(allImgPoints): " + str(len(allImgPoints)) + "\nallImgPoints: " + str(allImgPoints))
    
    if len(allImgPoints) < 3 or len(allObjPoints) < 3:
        return None
    
    print("allObjPoints: " + str(allObjPoints))
    print("allImgPoints: " + str(allImgPoints))

    transformationMatrix = calculateAndConstructTransformationMatrix(allObjPoints, allImgPoints, cameraMatrix, distCoeffs)
    print("transformationMatrix:\n" + str(transformationMatrix))
    
    # if transformationMatrix is not None:
    #     estimatedTransMat = useKalmanFilter(transformationMatrix, kalman)
    #     return estimatedTransMat
    
    return transformationMatrix

def predictCorners(rvecs, tvec, surfaceIds, objPoints):
    idRange = int(surfaceIds[0]/10)*10
    markerIdsToPredict = [idRange+1, idRange+2, idRange+3, idRange+4]
    predictedCoords = []
    for id in markerIdsToPredict:
        if id not in surfaceIds:
            # Nu hebben we een ID die we willen predicten
            index = id - idRange - 1
            objPointToPredict = objPoints[index][index]
            # Voor elke gekende id een prediction maken van het te predicten ID
            predictedCoords.append(convertTo3DCoord(objPointToPredict, rvecs, tvec))
            
    return predictedCoords
    # idRange = int(surfaceIds[0]/10)*10
    # markerIdsToPredict = [idRange+1, idRange+2, idRange+3, idRange+4]
    # for surfaceId in surfaceIds:
    #     if surfaceId in markerIdsToPredict:
    #         markerIdsToPredict.remove(surfaceId)

    # predictedCoords = []
    # for id in markerIdsToPredict:
    #     index = id - idRange - 1
    #     objPointToPredict = objPoints[index][index]
    #     predictedCoords.append(convertTo3DCoord(objPointToPredict, rvecs, tvec))
    # return predictedCoords

def calculateAndConstructTransformationMatrix(objPoints, imgPoints, cameraMatrix, distCoeffs):
    retval, rvecs, tvec = cv.solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs)
    if retval:
        rotationMatrix, _ = cv.Rodrigues(rvecs)
        transformationMatrix = np.eye(4)
        transformationMatrix[:3, :3] = rotationMatrix
        transformationMatrix[:3, 3] = tvec.flatten()
        transformationMatrix[3] = [0, 0, 0, 1]
        return transformationMatrix
    return None

def constructTransformationMatrix(rvecs, tvec):
    rotationMatrix, _ = cv.Rodrigues(rvecs)
    transformationMatrix = np.eye(4)
    transformationMatrix[:3, :3] = rotationMatrix
    transformationMatrix[:3, 3] = tvec.flatten()
    transformationMatrix[3] = [0, 0, 0, 1]
    return transformationMatrix

def convertTo3DCoord(objPoint, rvecs, tvecs):
    # Definieer de hoekpunten van de marker in het lokale markercoördinatensysteem
    objPoint = np.array([objPoint], dtype=np.float32)
    # print("objPoint: " + str(objPoint))
    homogeneObjPoint = cv.convertPointsToHomogeneous(objPoint)[0].flatten()
    # print("homogeneObjPoint: " + str(homogeneObjPoint))
    # Transformeer de hoekpunten naar het wereldcoördinatensysteem met de rotatie- en translatiematrices
    rotation_matrix, _ = cv.Rodrigues(rvecs)

    transformationMatrix = np.eye(4)

    # Vul de linkerbovenhoek van de 4x4-matrix met de rotatiematrix
    transformationMatrix[:3, :3] = rotation_matrix

    # Vul de rechterkolom van de 4x4-matrix met de translatievector
    transformationMatrix[:3, 3] = tvecs.flatten()

    # Voeg een lege rij toe aan de onderkant van de matrix
    transformationMatrix[3] = [0, 0, 0, 1]

    # print("transformationMatrix:\n" + str(transformationMatrix))

    worldCoord = np.dot(transformationMatrix, homogeneObjPoint)
    worldCoord = cv.convertPointsFromHomogeneous(np.array([worldCoord], dtype=np.float32))
    # print("marker_corners_world: " + str(worldCoord))

    return worldCoord

def convertTo2DCoord(coord, cameraMatrix, distCoeffs):
    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.zeros((3, 1), dtype=np.float32)
    point, _ = cv.projectPoints(coord, rvec, tvec, cameraMatrix, distCoeffs)
    point = np.array(point, dtype=np.int32)[0][0]
    return point

def draw3DPoints(img, coord, cameraMatrix, distCoeffs, color):
    point = convertTo2DCoord(coord, cameraMatrix, distCoeffs)
    img = cv.circle(img, (point[0], point[1]), 5, color, -1)
    return img

def initKalmanFilter():
    # https://docs.opencv.org/4.x/dc/d2c/tutorial_real_time_pose.html#gsc.tab=0
    nStates = 18
    nMeasurements = 6
    nInputs = 0
    dt = 1.0

    kalman = cv.KalmanFilter(nStates, nMeasurements, nInputs, cv.CV_64F)

    kalman.processNoiseCov = np.eye(nStates) * 1e-5  # set process noise
    kalman.measurementNoiseCov = np.eye(nMeasurements) * 1e-4  # set measurement noise
    kalman.errorCovPost = np.eye(nStates) * 1  # error covariance

    # transitionMatrix eerst nog eens proberen met eye

    # DYNAMIC MODEL
    # [1 0 0 dt 0 0 dt2 0 0 0 0 0 0 0 0 0 0 0]
    # [0 1 0 0 dt 0 0 dt2 0 0 0 0 0 0 0 0 0 0]
    # [0 0 1 0 0 dt 0 0 dt2 0 0 0 0 0 0 0 0 0]
    # [0 0 0 1 0 0 dt 0 0 0 0 0 0 0 0 0 0 0]
    # [0 0 0 0 1 0 0 dt 0 0 0 0 0 0 0 0 0 0]
    # [0 0 0 0 0 1 0 0 dt 0 0 0 0 0 0 0 0 0]
    # [0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0]
    # [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0]
    # [0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0]
    # [0 0 0 0 0 0 0 0 0 1 0 0 dt 0 0 dt2 0 0]
    # [0 0 0 0 0 0 0 0 0 0 1 0 0 dt 0 0 dt2 0]
    # [0 0 0 0 0 0 0 0 0 0 0 1 0 0 dt 0 0 dt2]
    # [0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 dt 0 0]
    # [0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 dt 0]
    # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 dt]
    # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0]
    # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0]
    # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]

    # position
    kalman.transitionMatrix[0, 3] = dt
    kalman.transitionMatrix[1, 4] = dt
    kalman.transitionMatrix[2, 5] = dt
    kalman.transitionMatrix[3, 6] = dt
    kalman.transitionMatrix[4, 7] = dt
    kalman.transitionMatrix[5, 8] = dt
    kalman.transitionMatrix[0, 6] = 0.5 * dt**2
    kalman.transitionMatrix[1, 7] = 0.5 * dt**2
    kalman.transitionMatrix[2, 8] = 0.5 * dt**2

    # orientation
    kalman.transitionMatrix[9, 12] = dt
    kalman.transitionMatrix[10, 13] = dt
    kalman.transitionMatrix[11, 14] = dt
    kalman.transitionMatrix[12, 15] = dt
    kalman.transitionMatrix[13, 16] = dt
    kalman.transitionMatrix[14, 17] = dt
    kalman.transitionMatrix[9, 15] = 0.5 * dt**2
    kalman.transitionMatrix[10, 16] = 0.5 * dt**2
    kalman.transitionMatrix[11, 17] = 0.5 * dt**2

    # MEASUREMENT MODEL, hoe hard vertrouwd ge u echte metingen -> laag = goed, hoog = slecht
    # [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    # [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    # [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    # [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
    # [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
    # [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]

    kalman.measurementMatrix[0, 0] = 1  # x
    kalman.measurementMatrix[1, 1] = 1  # y
    kalman.measurementMatrix[2, 2] = 1  # z
    kalman.measurementMatrix[3, 9] = 1  # roll
    kalman.measurementMatrix[4, 10] = 1  # pitch
    kalman.measurementMatrix[5, 11] = 1  # yaw

    return kalman

def useKalmanFilter(transformationMatrix, kalman):
    rotMat = transformationMatrix[:3, :3]
    tvec = transformationMatrix[:3, 3]

    euler = rotMat2Euler(rotMat)

    print("tvec: " + str(tvec))
    print("rotMat: " + str(rotMat))
    print("euler: " + str(euler))

    measurements = [tvec[0], tvec[1], tvec[2], euler[0], euler[1], euler[2]]
    measurements = np.array(measurements)

    print("measurements: " + str(measurements))

    prediction = kalman.predict()
    estimated = kalman.correct(measurements)
    
    print("kalman_corrected_state: " + str(estimated))

    # euler vervangen door quaternion, als ge pos quat meegeeft aan kalman -> altijd pos quat meegeven!

    tvecEstimated = np.array([estimated[0], estimated[1], estimated[2]])
    eulerEstimated = np.array([estimated[9], estimated[10], estimated[11]])

    rotMatEstimated = euler2RotMat(eulerEstimated)

    estimatedTransMat = constructTransformationMatrix(rotMatEstimated, tvecEstimated)
    
    # Return de gecorrigeerde 4x4-transformatiematrix
    return estimatedTransMat

def rotMat2Euler(m): 
    # https://euclideanspace.com/maths/geometry/rotations/conversions/matrixToEuler/index.htm

    # this conversion uses conventions as described on page:
    # https://www.euclideanspace.com/maths/geometry/rotations/euler/index.htm
    # Coordinate System: right hand
    # Positive angle: right hand
    # Order of euler angles: heading first, then attitude, then bank
    # matrix row column ordering:
    # [m00 m01 m02]
    # [m10 m11 m12]
    # [m20 m21 m22]
    
    # Assuming the angles are in radians.
    if m[1, 0] > 0.998:  # singularity at north pole
        heading = np.arctan2(m[0, 2], m[2, 2])
        attitude = np.pi / 2
        bank = 0
        return heading, attitude, bank
    if m[1, 0] < -0.998:  # singularity at south pole
        heading = np.arctan2(m[0, 2], m[2, 2])
        attitude = -np.pi / 2
        bank = 0
        return heading, attitude, bank
    heading = np.arctan2(-m[2, 0], m[0, 0])
    bank = np.arctan2(-m[1, 2], m[1, 1])
    attitude = np.arcsin(m[1, 0])
    return [bank, attitude, heading] # bank = roll, attitude = pitch, heading = yaw

def euler2RotMat(euler):
    # https://euclideanspace.com/maths/geometry/rotations/conversions/eulerToMatrix/index.htm

    # this conversion uses NASA standard aeroplane conventions as described on page:
    # https://www.euclideanspace.com/maths/geometry/rotations/euler/index.htm
    # Coordinate System: right hand
    # Positive angle: right hand
    # Order of euler angles: heading first, then attitude, then bank
    # matrix row column ordering:
    # [m00 m01 m02]
    # [m10 m11 m12]
    # [m20 m21 m22]

    heading = euler[2]
    attitude = euler[1]
    bank = euler[0]

    ch = np.cos(heading)
    sh = np.sin(heading)
    ca = np.cos(attitude)
    sa = np.sin(attitude)
    cb = np.cos(bank)
    sb = np.sin(bank)

    m00 = ch * ca
    m01 = sh * sb - ch * sa * cb
    m02 = ch * sa * sb + sh * cb
    m10 = sa
    m11 = ca * cb
    m12 = -ca * sb
    m20 = -sh * ca
    m21 = sh * sa * cb + ch * sb
    m22 = -sh * sa * sb + ch * cb

    return np.array([[m00, m01, m02],
                     [m10, m11, m12],
                     [m20, m21, m22]])

def trackSurfaces(img, arucoDetector):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    markerCorners, markerIds, _ = arucoDetector.detectMarkers(gray)

    if markerIds is not None and len(markerIds) >= 4:
        # neem lijsten samen
        zippedData = zip(markerIds, markerCorners)

        # sorteer op markerIds
        sortedData = sorted(zippedData, key=lambda x: x[0])

        # splits de lijsten weer op
        sortedMarkerIds, sortedMarkerCorners = zip(*sortedData)
        sortedMarkerIds = np.array(sortedMarkerIds)
        sortedMarkerCorners = np.array(sortedMarkerCorners)

        # print("sortedMarkerIds: " + str(sortedMarkerIds))
        # print("sortedMarkerCorners: " + str(sortedMarkerCorners))

        markerPoints = []
        itterations = int(len(sortedMarkerIds)/4)
        # print("iterations: " + str(itterations))
        for i in range(itterations):
            markerPoints.append([])
            for j in range(4):
                markerPoints[i].append(sortedMarkerCorners[i*4 + j][0][j])
                if j == 3:
                    markerPoints[i] = np.array(markerPoints[i], dtype=np.int32)

        # print("markerPoints: " + str(markerPoints))

        for surfacePoints in markerPoints:
            surfaceImg = cv.polylines(img, [surfacePoints], True, (255, 0, 255))
        return surfaceImg, markerPoints
    else:
        return img, None

def trackMarkers(img, arucoDetector):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    markerCorners, markerIds, _ = arucoDetector.detectMarkers(gray)

    if markerIds is not None:
        for i in range(len(markerIds)):
            # Bepaal rotatie- en translatievectoren voor elke marker
            # Werkt niet meer, deprecated method :(
            # rvec, tvec, _ = cv.aruco.estimatePoseSingleMarkers(markerCorners[i], markerLength, cameraMatrix, distCoeffs)

            markerImg = cv.aruco.drawDetectedMarkers(img, markerCorners, markerIds, (255, 0, 255))
        return markerImg
    else:
        return img

def main():
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_100)
    arucoDetector = cv.aruco.ArucoDetector(dictionary)

    # cameraSchool is de zware, de DepthDefect is de lichte
    cameraMatrix, distCoeffs, rvecs, tvecs = loadCameraCalibration("cameraSchoolDepthDefect")
    objPoints = loadObjectPoints("objectPoints")
    
    boxMarkerLength = 0.053 # in meter

    cap = cv.VideoCapture(1)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    serverAddressPort = ("127.0.0.1", 6969)

    while True:
        succes, img = cap.read()
        img = cv.undistort(img, cameraMatrix, distCoeffs)
        imgMarkers = trackMarkers(img.copy(), arucoDetector)

        transformationMatrix = trackBox(img, objPoints, cameraMatrix, distCoeffs, arucoDetector, initKalmanFilter())
        if transformationMatrix is not None:
            sendDataToUnity(sock, serverAddressPort, transformationMatrix)

        cv.imshow("camera", imgMarkers)
        # cv.imshow("predicted", imgPredicted)

        cv.waitKey(1)

if __name__ == "__main__":
    main()