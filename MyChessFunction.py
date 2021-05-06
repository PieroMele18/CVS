"""Import di alcune librerie utili :
Numpy per gestire le matrice
Opencv per gestire la parte della visione artificiale"""
import operator

import numpy as np
import cv2
import chess

"""Restituisce una tupla che costituisce 
le coordinate dei punti , contenuti all'interno
di una struttura vettoriale (singolo elemento)"""
def get_coordinates(coordinates):
    return int(coordinates[0]),int(coordinates[1])

"""Stampa i bordi della scacchiera"""
def draw_chessboard_sides(img,found,corners):
    if found:
        coord = extreme_corners(corners)
        a,b,d,c = get_coordinates(coord[0]),get_coordinates(coord[1]),get_coordinates(coord[2]),get_coordinates(coord[3])

        cv2.line(img,a,b,(255,255,255),2)
        cv2.line(img,b,c,(255,255,255),2)
        cv2.line(img,c,d,(255,255,255),2)
        cv2.line(img,d,a,(255,255,255),2)

        cv2.imwrite("Border.jpg", img)

    return img

def get_chessboard(img,found,corners):
    if found :
        width, height = 390, 390
        coord = extreme_corners(corners)
        a,b,d,c = get_coordinates(coord[0]),get_coordinates(coord[1]),get_coordinates(coord[2]),get_coordinates(coord[3])
        pts1 = np.float32([[a[0], a[1]],
                           [b[0], b[1]],
                           [d[0], d[1]],
                           [c[0], c[1]]])
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, matrix, (width, height))
    return img

"""Crea una matrice costituita dagli estremi della scacchiera"""
def extreme_corners(corners):

    extreme_corners = np.zeros(shape=(4,2))

    #Primo Punto
    a = corners[0]
    b = corners[8]

    extreme_corners[0][0] = a[0][0] - (b[0][0]-a[0][0])
    extreme_corners[0][1] = a[0][1] - (b[0][1]-a[0][1])

    #Secondo Punto
    a = corners[6]
    b = corners[12]

    extreme_corners[1][0] = a[0][0] - (b[0][0]-a[0][0])
    extreme_corners[1][1] = a[0][1] - (b[0][1]-a[0][1])

    #Terzo Punto
    b = corners[36]
    a = corners[42]

    extreme_corners[2][0] = a[0][0] - (b[0][0]-a[0][0])
    extreme_corners[2][1] = a[0][1] - (b[0][1]-a[0][1])

    #Quarto Punto
    b = corners[40]
    a = corners[48]

    extreme_corners[3][0] = a[0][0] - (b[0][0]-a[0][0])
    extreme_corners[3][1] = a[0][1] - (b[0][1]-a[0][1])

    return extreme_corners

"""Utilizzata per disegnare sulla scacchiera i punti trovati 
dalla funzione opencv per la calibrazione delle camera.
Questa funzione mostra l'ordinamento dei punti , poichè per 
ognuno stampa un valore numerico"""
def draw_coordinates(img,found,corners):
    if (found):
        i = 0
        for coordinates in corners:
            i = i + 1
            img = cv2.putText(img, str(i), get_coordinates(coordinates[0]), cv2.FONT_ITALIC, 1, (255, 0, 0), 2,
                              cv2.LINE_AA)
            # Draw a circle with blue line borders of thickness of 2 px
            img = cv2.circle(img, get_coordinates(coordinates[0]), 2, (0, 0, 0), 2)
    return img

"""Funzione di test per provare quale tecnica di riduzione
del rumore utilizzare"""
def test_convolution(img,warp_points):
    width, height = 310, 310
    pts1 = np.float32([[warp_points[0][0], warp_points[0][1]],
                       [warp_points[1][0], warp_points[1][1]],
                        [warp_points[3][0], warp_points[3][1]],
                       [warp_points[2][0], warp_points[2][1]]])
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    img = cv2.warpPerspective(img,matrix,(width,height))

    # Blurring tramite convoluzione con kernel HPF
    kernel = np.ones((5, 5), np.float32) / 25
    blur = cv2.filter2D(img, -1, kernel)
    cv2.imwrite("hpf.jpg", blur)

    #Blurring tramite convoluzione di Gaussiana
    blur = cv2.GaussianBlur(img,(5,5),0)
    cv2.imwrite("Gaussian.jpg", blur)

    #Blurring tramite averaging
    blur = cv2.blur(img,(5,5))
    cv2.imwrite("Averaging.jpg", blur)

    #Blurring tramite tecnica mediana
    blur = cv2.medianBlur(img,5)
    cv2.imwrite("Median.jpg", blur)

    #Blurring tramite tecnica bilaterale
    blur = cv2.bilateralFilter(img,9,75,75)
    cv2.imwrite("Bilateral.jpg", blur)

"""Funzione per ridurre il rumore e rendere grigia l'immagine"""
def pre_processing(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.bilateralFilter(gray,9,75,75)
    return gray_blur

"""Funzione che stampa le coordinate a partire da una matrice (81,2)"""
def test_draw_coordinates(img,corners):

    i=0
    for coordinates in corners:
        i = i+1
        # Stampa i numeri corrispondenti ai punti
        img = cv2.putText(img, str(i), (int(coordinates[0]),int(coordinates[1])), cv2.FONT_ITALIC, 0.5, (255, 0, 0), 2,
                              cv2.LINE_4)
        # Stampa il punto
        img = cv2.circle(img, (int(coordinates[0]),int(coordinates[1])), 2, (255, 0, 0), 2)
    return img

"""Funzione che crea la matrice dei punti presenti nella scacchiera 
a partire dalla funzione per la calibrazione della fotocamera """
def get_final_coordinates(corners):
    final_coordinates = np.zeros((81, 2))
    #Disegna i punti che vanno da 1 a 5
    for x in range(0,5):

        a = corners[0+x]
        b = corners[8+x]

        final_coordinates[x][0] = a[0][0] - (b[0][0]-a[0][0])
        final_coordinates[x][1] = a[0][1] - (b[0][1]-a[0][1])

    #Disegna i punti che vanno da 6 a 9
    for x in range(0,4):
        a = corners[6-x]
        b = corners[12-x]

        final_coordinates[8-x][0] = a[0][0] - (b[0][0]-a[0][0])
        final_coordinates[8-x][1] = a[0][1] - (b[0][1]-a[0][1])

    i = 0
    #Disegna i punti laterali ( 9 - 18 - 27 - 36 - 45  )
    for x in range(0,35,7):

        a = corners[7+x]
        b = corners[15+x]

        i = i+1

        final_coordinates[9*i][0] = a[0][0] - (b[0][0]-a[0][0])
        final_coordinates[9*i][1] = a[0][1] - (b[0][1]-a[0][1])


    i = 0
    #Disegna i punti laterali (54-63-72-81)
    for x in range(0,28,7):

        a = corners[42-x]
        b = corners[36-x]


        final_coordinates[72-i*9][0] = a[0][0] - (b[0][0]-a[0][0])
        final_coordinates[72-i*9][1] = a[0][1] - (b[0][1]-a[0][1])

        i = i+1

    #Punti da 74 a 78
    for x in range(0,5):

        a = corners[43+x]
        b = corners[37+x]

        final_coordinates[73+x][0] = a[0][0] - (b[0][0]-a[0][0])
        final_coordinates[73+x][1] = a[0][1] - (b[0][1]-a[0][1])

    #Punti da 79 a 81
    for x in range(0,4):

        a = corners[48-x]
        b = corners[40-x]

        final_coordinates[80-x][0] = a[0][0] - (b[0][0]-a[0][0])
        final_coordinates[80-x][1] = a[0][1] - (b[0][1]-a[0][1])

    i = 1
    #Disegna i punti laterali ( )
    for x in range(0,35,7):

        a = corners[13+x]
        b = corners[19+x]

        i = i+1

        final_coordinates[9*i-1][0] = a[0][0] - (b[0][0]-a[0][0])
        final_coordinates[9*i-1][1] = a[0][1] - (b[0][1]-a[0][1])


    i = 0
    #Disegna i punti laterali (54-63-72-81)
    for x in range(0,14,7):

            a = corners[41-x]
            b = corners[33-x]


            final_coordinates[71-i*9][0] = a[0][0] - (b[0][0]-a[0][0])
            final_coordinates[71-i*9][1] = a[0][1] - (b[0][1]-a[0][1])

            i = i+1

    x = 0;
    for i in range(0,71):
        a = corners[x]

        if  final_coordinates[i][0] == 0 and final_coordinates[i][1] == 0:
            final_coordinates[i][0] = a[0][0]
            final_coordinates[i][1] = a[0][1]
            x+=1


    return final_coordinates;

def get_trackbar():
    cv2.namedWindow("TrackBar")
    cv2.resizeWindow("TrackBar", 640, 240)
    cv2.createTrackbar("Hue Min", "TrackBar", 0, 255, empty)
    cv2.createTrackbar("Hue Max","TrackBar",1,255,empty)
    cv2.createTrackbar("Sat Min", "TrackBar", 0, 255, empty)
    cv2.createTrackbar("Sat Max","TrackBar",1,255,empty)
    cv2.createTrackbar("Val Min", "TrackBar", 0, 255, empty)
    cv2.createTrackbar("Val Max","TrackBar",255,255,empty)

def hsv_test(img):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min","TrackBar")
    h_max = cv2.getTrackbarPos("Hue Max","TrackBar")
    sat_min = cv2.getTrackbarPos("Sat Min","TrackBar")
    sat_max = cv2.getTrackbarPos("Sat Max","TrackBar")
    val_min = cv2.getTrackbarPos("Val Min","TrackBar")
    val_max = cv2.getTrackbarPos("Val Max","TrackBar")

    lower = np.array([h_min,sat_min,val_min])
    upper = np.array([h_max,sat_max,val_max])
    mask = cv2.inRange(imgHSV,lower,upper)

    cv2.imshow("Test",mask)

def diff_test(img):

    board = cv2.imread("Chessboard.jpg")
    #board = cv2.cvtColor(board,cv2.COLOR_BGR2GRAY)

    cv2.imshow("board",board)
    cv2.imshow("img",img)
    result = cv2.subtract(board,img)
    diff_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    #h_min = cv2.getTrackbarPos("Hue Min", "TrackBar")
    #h_max = cv2.getTrackbarPos("Hue Max", "TrackBar")
    #matrix, result = cv2.threshold(diff_gray, h_min, h_max, cv2.THRESH_BINARY)

    return result

def empty(x):
    pass

def undistort(img,mtx,dist,newcameramtx,roi):
    img = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    img = img [y:y + h, x:x + w]
    return img

def boxes_matrix(img,cor):

    #Vettore che conterrà le immagini delle case
    boxes = []
    for j in range(0,64,9):
        for i in range(0,8,1):

            # Larghezza e altezza , le dimensioni possono differire a causa della prospettiva
            width,height = 128,128
            a,b,c,d = cor[j+i],cor[j+i+1],cor[j+i+10],cor[j+i+9]
            pts1 = np.float32([[a[0], a[1]],
                               [b[0], b[1]],
                               [d[0], d[1]],
                               [c[0], c[1]]])
            pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            frame = cv2.warpPerspective(img, matrix, (width, height))
            frame = frame[24:104, 24:104] # Medesima Roi utilizzata per machine learning
            boxes.append(frame)

    return boxes

def create_data_set(boxes):

    for i in range(0,64):

        #Per avere un dataset di una certa consistenza creo quattro versioni per ogni casa
        frame = boxes[i][24:104,24:104] # Estrazione Roi
        frame90 = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) #Giro l'immagine di 90 gradi
        frame180 = cv2.rotate(frame,  cv2.ROTATE_180 ) #Giro l'immagine di 180 gradi
        frame270 = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) #Giro l'immagine di 270 gradi

        cv2.imwrite("img"+str(i)+".jpg",frame)
        cv2.imwrite("img90"+str(i)+".jpg",frame90)
        cv2.imwrite("img180"+str(i)+".jpg",frame180)
        cv2.imwrite("img270"+str(i)+".jpg",frame270)

def print_positional_matrix(matrix):
    print("MATRICE POSIZIONALE")
    for x in range(8):
        for y in range(8):
            print(matrix[x][y], end='  ')
        print("")

def get_move_single(old,new,chessboard,old_opp):

    matrix_chessboard = [["h1","g1","f1","e1","d1","c1","b1","a1"],
                         ["h2","g2","f2","e2","d2","c2","b2","a2"],
                         ["h3","g3","f3","e3","d3","c3","b3","a3"],
                         ["h4","g4","f4","e4","d4","c4","b4","a4"],
                         ["h5","g5","f5","e5","d5","c5","b5","a5"],
                         ["h6","g6","f6","e6","d6","c6","b6","a6"],
                         ["h7","g7","f7","e7","d7","c7","b7","a7"],
                         ["h8","g8","f8","e8","d8","c8","b8","a8"]]
    move = [[0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0]]



    for x in range(8):
        for y in range(8):
            move[x][y] = old[x][y] - new[x][y]



    da = ""
    a = ""

    for x in range(8):
        for y in range(8):

            if move[x][y] == -1 :
                a = matrix_chessboard[x][y]

                # Salvo indici per la gestione delle mosse enpassant

                global index_x, index_y

                index_x = x
                index_y = y

            if move[x][y] ==  1 :
                da  = matrix_chessboard[x][y]


    if(da=="" or a==""):
        raise Exception("Mossa non valida")

    #Gestione EnPassant
    if(chessboard.is_en_passant(chess.Move(chess.parse_square(da),chess.parse_square(a)))):
        if(index_x == 5):
            whiteTakeEnpassant(old_opp,index_x,index_y)
        elif(index_x == 2):
            blackTakeEnpassant(old_opp,index_x,index_y)




    # Gestione Promozione Bianco

    piece_moved = chessboard.piece_at(chess.parse_square(da))

    if (piece_moved == chess.Piece(1,True)):
        if da == "a7" or da == "b7" or da == "c7" or da == "d7" or da == "e7" or da == "f7" or da == "g7" or da == "h7":
            return da + a + "q"


    # Gestione Promozione Nero

    if (piece_moved == chess.Piece(1,False)):
        if da == "a2" or da == "b2" or da == "c2" or da == "d2" or da == "e2" or da == "f2" or da == "g2" or da == "h2":
            return da + a + "q"

    #Gestione arrocco

    #Arrocco bianco

    #Condizioni da verificarsi in caso di arrocco corto
    if move[0][0]==1 and move[0][1]==-1 and move[0][2]==-1 and move [0][3] == 1 :
        return "e1g1"

    #Condizioni da verificarsi in caso di arrocco lungo
    if move[0][3]==1 and move[0][4]==-1 and move[0][5]==-1 and move [0][7] == 1 :
        return "e1c1"

    #Arrocco nero
    #Condizioni da verificarsi in caso di arrocco corto
    if move[7][0]==1 and move[7][1]==-1 and move[7][2]==-1 and move [7][3] == 1 :
        return "e8g8"

    #Condizioni da verificarsi in caso di arrocco lungo
    if move[7][3]==1 and move[7][4]==-1 and move[7][5]==-1 and move [7][7] == 1 :
        return "e8c8"

    return da+a

def get_move(old,new,chessboard):

    matrix_chessboard = [["h1","g1","f1","e1","d1","c1","b1","a1"],
                         ["h2","g2","f2","e2","d2","c2","b2","a2"],
                         ["h3","g3","f3","e3","d3","c3","b3","a3"],
                         ["h4","g4","f4","e4","d4","c4","b4","a4"],
                         ["h5","g5","f5","e5","d5","c5","b5","a5"],
                         ["h6","g6","f6","e6","d6","c6","b6","a6"],
                         ["h7","g7","f7","e7","d7","c7","b7","a7"],
                         ["h8","g8","f8","e8","d8","c8","b8","a8"]]
    move = [[0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0]]

    for x in range(8):
        for y in range(8):
            move[x][y] = old[x][y] - new[x][y]



    da = ""
    a = ""

    for x in range(8):
        for y in range(8):

            if move[x][y] == -1 :
                a = matrix_chessboard[x][y]

                # Salvo indici per la gestione delle mosse enpassant

                global index_x, index_y

                index_x = x
                index_y = y

            if move[x][y] ==  1 :
                da  = matrix_chessboard[x][y]


    if(da=="" or a==""):
        raise Exception("Mossa non valida")



    # Gestione Promozione Bianco

    piece_moved = chessboard.piece_at(chess.parse_square(da))

    if (piece_moved == chess.Piece(1,True)):
        if da == "a7" or da == "b7" or da == "c7" or da == "d7" or da == "e7" or da == "f7" or da == "g7" or da == "h7":
            return da + a + "q"


    # Gestione Promozione Nero

    if (piece_moved == chess.Piece(1,False)):
        if da == "a2" or da == "b2" or da == "c2" or da == "d2" or da == "e2" or da == "f2" or da == "g2" or da == "h2":
            return da + a + "q"

    #Gestione arrocco

    #Arrocco bianco

    #Condizioni da verificarsi in caso di arrocco corto
    if move[0][0]==1 and move[0][1]==-1 and move[0][2]==-1 and move [0][3] == 1 :
        return "e1g1"

    #Condizioni da verificarsi in caso di arrocco lungo
    if move[0][3]==1 and move[0][4]==-1 and move[0][5]==-1 and move [0][7] == 1 :
        return "e1c1"

    #Arrocco nero
    #Condizioni da verificarsi in caso di arrocco corto
    if move[7][0]==1 and move[7][1]==-1 and move[7][2]==-1 and move [7][3] == 1 :
        return "e8g8"

    #Condizioni da verificarsi in caso di arrocco lungo
    if move[7][3]==1 and move[7][4]==-1 and move[7][5]==-1 and move [7][7] == 1 :
        return "e8c8"

    return da+a


def isStart(matrix):
    if matrix==\
    [[1,1,1,1,1,1,1,1],
     [1,1,1,1,1,1,1,1],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [1,1,1,1,1,1,1,1],
     [1,1,1,1,1,1,1,1]]: return True
    else: return False

def isEmpty(matrix):
    if matrix==\
    [[0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0]] : return True
    else: return False

def isBlack(matrix):
        if matrix == \
                [[0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1]]:
            return True
        else:
            return False

def isWhite(matrix):
    if matrix == \
            [[1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0]]:
        return True
    else:
        return False

def setWhite():
    x = [[1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]]

    return x

def setBlack():
    x =[[0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1]]

    return x

def whiteTake(newWhite,oldBlack):
    for x in range(8):
        for y in range(8):
            if(newWhite[x][y] == oldBlack[x][y]):
                oldBlack[x][y] = 0

    return oldBlack


def blackTake(newBlack, oldWhite):
    for x in range(8):
        for y in range(8):
            if (newBlack[x][y] == oldWhite[x][y]):
                oldWhite[x][y] = 0

    return oldWhite


def whiteTakeEnpassant(oldBlack,x,y):
    oldBlack[x-1][y] = 0

def blackTakeEnpassant(oldWhite,x,y):
    oldWhite[x - 1][y] = 0

def computer_black_move(move,oldwhite):

    matrix_chessboard = [["h1", "g1", "f1", "e1", "d1", "c1", "b1", "a1"],
                         ["h2", "g2", "f2", "e2", "d2", "c2", "b2", "a2"],
                         ["h3", "g3", "f3", "e3", "d3", "c3", "b3", "a3"],
                         ["h4", "g4", "f4", "e4", "d4", "c4", "b4", "a4"],
                         ["h5", "g5", "f5", "e5", "d5", "c5", "b5", "a5"],
                         ["h6", "g6", "f6", "e6", "d6", "c6", "b6", "a6"],
                         ["h7", "g7", "f7", "e7", "d7", "c7", "b7", "a7"],
                         ["h8", "g8", "f8", "e8", "d8", "c8", "b8", "a8"]]


    move_matrix = [[0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0]]

    da,a = parse_move(move)

    for x in range(8):
        for y in range(8):
            if( da == matrix_chessboard[x][y] ) :
                move_matrix[x][y] = 1

            if( a == matrix_chessboard[x][y] ) :
                move_matrix[x][y] = -1

    for x in range(8):
        for y in range(8):
            if(move_matrix[x][y] == -1 and oldwhite[x][y] == 1):
                oldwhite[x][y] = 0

    return oldwhite


def parse_move(move):

    da = move[0]+move[1]
    a = move[2]+move[3]

    return da,a