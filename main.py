import cv2
from MyChessFunction import get_chessboard,create_data_set,undistort,pre_processing,boxes_matrix,diff_test,get_trackbar,get_coordinates,draw_coordinates,extreme_corners,draw_chessboard_sides,test_draw_coordinates,get_final_coordinates
from Calibration import camera_calibration,save_img_for_calibration

#Variabili di sistema
chessboard_found = False
flag = False
newPosition = False
boxes_found = False
i = 0

# Codice numerico per la webcam , flag per api microsoft direct show
webcam = cv2.VideoCapture(0,cv2.CAP_DSHOW)

# Controllo del funzionamento della webcam da console
print("Camera collegata" if webcam.isOpened() else "Camera non trovata")

"""Utilizzato per creare immagini all'interno della configurazione"""
#save_img_for_calibration(webcam)

ret, mtx, dist, rvecs, tvecs, h, w = camera_calibration()
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))


# Ciclo per registrare il flusso delle immagini
while (True):

    success, img = webcam.read()

    #Rimuovere distorsione
    img = undistort(img, mtx, dist, newcameramtx, roi)

    # Pre-Elaborazione dell'immagine
    img = pre_processing(img)

    # Ricerca della scacchiera
    if(chessboard_found == False):
        chessboard_found , chessboard_corners = cv2.findChessboardCorners(img, (7, 7), None)

    #Estrazione scacchiera dall'immagine
    img = get_chessboard(img, chessboard_found, chessboard_corners)

    #Se la scacchiera è stata trovata
    if chessboard_found:

        #Ricerca delle case della scacchiera se queste non sono state mai trovate
        if not boxes_found:
            boxes_found, box_corners = cv2.findChessboardCorners(img, (7, 7), None)

        #Se vengono trovate le case , dividi la scacchiera in 64 quadrati ( le case )
        if boxes_found:
            coordinates = get_final_coordinates(box_corners)
            boxes = boxes_matrix(img, coordinates)


    cv2.imshow("Numbers",img)

    if cv2.waitKey(1) & 0xFF == ord('n') :
        create_data_set(boxes)
    if cv2.waitKey(2) & 0xFF == ord('r') :
        chessboard_found = False
        boxes_found = False
    if cv2.waitKey(3) & 0xFF == ord('q') :
        # End the program
        break


# Rilascio delle risorse allocate per la webcam
webcam.release()

# Controllo che la webcam sia stata chiusa
print("Camera scollegata" if not webcam.isOpened() else "Camera non chiusa")

cv2.destroyAllWindows()

