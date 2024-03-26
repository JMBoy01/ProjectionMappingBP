import cv2 as cv
import numpy as np

def detectCharucoBoard(img, charuco_board, arucoDetector, allCharucoCorners, allCharucoIds):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    markerCorners, markerIds, _ = arucoDetector.detectMarkers(gray)
    # print("markers: " + str(markerIds))
    if markerIds is not None:
        retval, charucoCorners, charucoIds = cv.aruco.interpolateCornersCharuco(markerCorners, markerIds, gray, charuco_board)
        # print("charuco: " + str(charucoIds))
        # print("retval: " + str(retval))
        if retval and len(charucoIds) >= 6:
            img = cv.aruco.drawDetectedCornersCharuco(img, charucoCorners, charucoIds, (255, 0, 255))
            allCharucoCorners.append(charucoCorners)
            allCharucoIds.append(charucoIds)

            print("retval: " + str(retval))

    return img

def saveCameraCalibration(filename, cameraMatrix, distCoeffs, rvecs, tvecs):
    fs = cv.FileStorage(filename, cv.FILE_STORAGE_WRITE)
    fs.write("cameraMatrix", cameraMatrix)
    fs.write("distCoeffs", distCoeffs)

    rvecs = np.array(rvecs)
    tvecs = np.array(tvecs)
    fs.write("rvecs", rvecs)
    fs.write("tvecs", tvecs)
    fs.release()
    print("Calibratiegegevens opgeslagen in", filename)

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

#---------------------------------------------------------------------------------------------------------------------------------#

def main():
    cap = cv.VideoCapture(1)

    boardPhoto = cv.imread("board.jpg")

    dictionaries = findDict(cv.cvtColor(boardPhoto, cv.COLOR_BGR2GRAY))
    print(dictionaries)

    dictionary = cv.aruco.getPredefinedDictionary(dictionaries[1])
    arucoDetector = cv.aruco.ArucoDetector(dictionary)

    corners, usedMarkerIds, _ = arucoDetector.detectMarkers(boardPhoto)
    boardPhoto = cv.aruco.drawDetectedMarkers(boardPhoto, corners, usedMarkerIds, (255, 0, 255))

    # cv.imshow("boardPhoto", boardPhoto)
    # cv.waitKey(0)

    boardSize = (8, 6) # aantal vakjes (hor, ver)
    squareLength = 0.03 # in meter
    markerLength = 0.015 # in meter
    charuco_board = cv.aruco.CharucoBoard(boardSize, squareLength, markerLength, dictionary)

    allCharucoCorners = []
    allCharucoIds = []

    FPS = 1

    while True:
        succes, img = cap.read()

        detectedImg = detectCharucoBoard(img, charuco_board, arucoDetector, allCharucoCorners, allCharucoIds)
        
        cv.imshow("Collecting frames - press C to calibrate camera", detectedImg)
        key = cv.waitKey(int(1000/FPS))

        if key == ord('c'):
            cv.destroyAllWindows()
            break

    print("Calibrating camera...")
    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv.aruco.calibrateCameraCharuco(allCharucoCorners, allCharucoIds, charuco_board, (img.shape[0], img.shape[1]), None, None)
    if retval:
        print("Camera calibrated")
        saveCameraCalibration("cameraSchoolDepthDefect", cameraMatrix, distCoeffs, rvecs, tvecs)
        print("Calibration data saved!")
    else:
        print("Couldn't calibrate camera, exiting program...")
        exit(0)

if __name__ == "__main__":
    main()
