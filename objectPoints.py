import numpy as np
import cv2 as cv

def saveCameraCalibration(filename, objectPoints00, objectPoints10, objectPoints20):
    fs = cv.FileStorage(filename, cv.FILE_STORAGE_WRITE)

    objectPoints00 = np.array(objectPoints00)
    objectPoints10 = np.array(objectPoints10)
    objectPoints20 = np.array(objectPoints20)

    fs.write("objectPoints00", objectPoints00)
    fs.write("objectPoints10", objectPoints10)
    fs.write("objectPoints20", objectPoints20)
    fs.release()
    print("Object points opgeslagen in", filename)


def main():
    boxMarkerLength = 0.053 # in meter
    b = 0.39 # in meter
    h = 0.13 # in meter
    d = 0.275 # in meter

    # objectPoints per oppervlak + in volgorde van id
    # volgorde markers en hoeken = links boven, rechts boven, rechts onder, links onder
    objectPoints00 = [[[0, 0, 0], [boxMarkerLength, 0, 0], [boxMarkerLength, boxMarkerLength, 0], [0, boxMarkerLength, 0]],
                    [[b-boxMarkerLength, 0, 0], [b, 0, 0], [b, boxMarkerLength, 0], [b-boxMarkerLength, boxMarkerLength, 0]],
                    [[b-boxMarkerLength, d-boxMarkerLength, 0], [b, d-boxMarkerLength, 0], [b, d, 0], [b-boxMarkerLength, d, 0]],
                    [[0, d-boxMarkerLength, 0], [boxMarkerLength, d-boxMarkerLength, 0], [boxMarkerLength, d, 0], [0, d, 0]]]

    objectPoints10 = [[[0, 0, h], [boxMarkerLength, 0, h], [boxMarkerLength, 0, h-boxMarkerLength], [0, 0, h-boxMarkerLength]],
                    [[b-boxMarkerLength, 0, h], [b, 0, h], [b, 0, h-boxMarkerLength], [b-boxMarkerLength, 0, h-boxMarkerLength]],
                    [[b-boxMarkerLength, 0, boxMarkerLength], [b, 0, boxMarkerLength], [b, 0, 0], [b-boxMarkerLength, 0, 0]],
                    [[0, 0, boxMarkerLength], [boxMarkerLength, 0, boxMarkerLength], [boxMarkerLength, 0, 0], [0, 0, 0]]]

    objectPoints20 = [[[0, d, h], [0, d-boxMarkerLength, h], [0, d-boxMarkerLength, h-boxMarkerLength], [0, d, h-boxMarkerLength]],
                    [[0, boxMarkerLength, h], [0, 0, h], [0, 0, h-boxMarkerLength], [0, boxMarkerLength, h-boxMarkerLength]], 
                    [[0, boxMarkerLength, boxMarkerLength], [0, 0, boxMarkerLength], [0, 0, 0], [0, boxMarkerLength, 0]], 
                    [[0, d, boxMarkerLength], [0, d-boxMarkerLength, boxMarkerLength], [0, d-boxMarkerLength, 0], [0, d, 0]]]
    
    saveCameraCalibration("objectPoints", objectPoints00, objectPoints10, objectPoints20)

if __name__ == "__main__":
    main()