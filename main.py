"""
Alcune delle funzioni che utilizzerò all'interno del programma
sono presneti nelle librerie da me definite
"""
from PyQt5.QtSvg import QSvgWidget

from MyChessFunction import *
from Calibration import *
from teachableMachine import *
from game_chess import *

"""
Librerie esterne in uso all'interno del programma :
cv2 per la computer vision 
python-chess per la parte legata al motore di scacchi
"""
import cv2
import chess

from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QPushButton
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np

#DEFINIZIONE VARIABILI UTILI
chessboard_found = False
boxes_found = False
chessboard_corners = None
box_corners = None
img_chessboard = None
coordinates = None
isWhiteTurn = True
isBlackTurn = False
oldblack = setBlack()
oldwhite = setWhite()
chessboard = chess.Board("8/8/8/8/8/8/8/8 w - - 0 1")




class VideoThread(QThread):
	change_pixmap_signal = pyqtSignal(np.ndarray)

	def __init__(self):
		super().__init__()
		self._run_flag = True

	def run(self):
		# GESTIONE WEBCAM
		webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

		# CALIBRAZIONE WEBCAM
		ret, mtx, dist, rvecs, tvecs, h, w = camera_calibration()
		newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

		while self._run_flag:
			success, img = webcam.read()

			if success:

				# Rimuovere distorsione
				img = undistort(img, mtx, dist, newcameramtx, roi)

				# Pre-Elaborazione dell'immagine
				img_ = pre_processing(img)
				# Ricerca della scacchiera

				#DEFINIZIONE VARIABILI GLOBALI
				global chessboard_found
				global chessboard_corners

				if(chessboard_found == False):
					#Ricerca della scacchiera
					chessboard_found,  chessboard_corners = cv2.findChessboardCorners(img_, (7, 7), None)

				# Se la scacchiera è stata trovata
				if chessboard_found:
					# Ricerca della scacchiera

					#VARIABILE GLOBALE UTILIZZATA
					global img_chessboard

					img_chessboard  = get_chessboard(img, chessboard_found, chessboard_corners)
					# Ricerca delle case della scacchiera se queste non sono state mai trovate

					#DEFINIZIONE VARIABILI GLOBALI
					global boxes_found
					global box_corners
					global coordinates

					if boxes_found == False:
						#Ricerca della case all'interno della scacchiera
						boxes_found, box_corners = cv2.findChessboardCorners(pre_processing(img_chessboard), (7, 7), None)
						coordinates = get_final_coordinates(box_corners)

				self.change_pixmap_signal.emit(img_chessboard)

		# shut down capture system
		webcam.release()

	def stop(self):
		"""Sets run flag to False and waits for thread to finish"""
		self._run_flag = False
		self.wait()




class App(QWidget):
	def __init__(self):
		super().__init__()
		self.setWindowTitle("Chess Computer System ")
		self.display_width = 1080
		self.display_height = 720
		self.setWindowIcon(QtGui.QIcon("horse.png"))

		# create the label that holds the image
		self.image_label = QLabel(self)
		self.image_label.setGeometry(10,10,400,400)

		# mostra scacchiera all'interno dell'interfaccia
		self.widgetSvg = QSvgWidget(parent=self)
		self.widgetSvg.setGeometry(950, 10, 400, 400)

		self.solitario = QPushButton('Gioco solitario', self)
		self.solitario.setToolTip('This is an example button')
		self.solitario.setGeometry(650, 70, 100, 40)
		self.solitario.clicked.connect(self.on_click_solitario)

		self.button = QPushButton('Prossima mossa', self)
		self.button.setGeometry(650, 70, 100, 40)
		#self.button.move(650, 70)
		self.button.clicked.connect(self.on_click_next)
		self.button.hide()


		self.buttonReset = QPushButton('Reset', self)
		self.buttonReset.setGeometry(650, 120, 100, 40)
		self.buttonReset.clicked.connect(self.on_click_reset)
		self.buttonReset.hide()

		self.chessboardSvg = chess.svg.board(chessboard).encode("UTF-8")
		self.widgetSvg.load(self.chessboardSvg)


		# create the video capture thread
		self.thread = VideoThread()
		# connect its signal to the update_image slot
		self.thread.change_pixmap_signal.connect(self.update_image)
		# start the thread
		self.thread.start()

	@ pyqtSlot()
	def on_click_solitario(self):

		global chessboard

		# Estrazione delle case della matrice
		if boxes_found:
			boxes = boxes_matrix(img_chessboard, coordinates)

		# Creazione matrice posizionale
		matrix = find_pieces(boxes,"all")

		if isStart(matrix):
			#La scacchiera è nel suo stato iniziale
			chessboard = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

			#Aggiorna pulsanti
			self.solitario.hide()
			self.button.show()
			self.buttonReset.show()

			#Aggiorno la scacchiera a schermo
			self.updateChessboard(chessboard)
		elif isEmpty(matrix):
			pass

	@pyqtSlot()
	def on_click_reset(self):
		global isWhiteTurn
		global isBlackTurn
		global oldblack
		global oldwhite
		global chessboard

		isWhiteTurn = True
		isBlackTurn = False

		oldblack = setBlack()
		oldwhite = setWhite()

		chessboard = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
		self.updateChessboard(chessboard)

	@pyqtSlot()
	def on_click_next(self):

		#Variabile globale da utilizzare
		global isWhiteTurn
		global isBlackTurn
		global oldblack
		global oldwhite

		# Estrazione delle case della matrice
		if boxes_found:
			boxes = boxes_matrix(img_chessboard, coordinates)
			create_data_set(boxes)

		#Se è il turno del bianco
		if isWhiteTurn:

			#Estrazione matrice posizionale bianca
			matrix = find_pieces(boxes, "white")
			#Elaborazione mossa del bianco
			move = get_move(oldwhite, matrix,chessboard)

			try :
				chessboard.push_uci(move)
			except :
				print("Mossa non valida")
				return
			#Gestione dei turni
			isWhiteTurn = False
			isBlackTurn = True
			#Aggiorna la scacchiera a schermo
			self.updateChessboard(chessboard)
			#Sovrascrivere variabile dello stato precedente con il nuovo stato
			oldwhite = matrix
			return

		#Se è il turno del bianco
		if isBlackTurn:
			#Estrazione matrice posizionale bianca
			matrix = find_pieces(boxes, "black")
			# Elaborazione mossa del bianco
			move = get_move(oldblack, matrix,chessboard)
			try:
				chessboard.push_uci(move)
			except:
				print("Mossa non valida")
				return
			#Passa il turno
			isWhiteTurn = True
			isBlackTurn = False
			#Aggiorna la scacchiera a schermo
			self.updateChessboard(chessboard)
			#Sovrascrivere variabile dello stato precedente con il nuovo stato
			oldblack = matrix
			return



	def closeEvent(self, event):
		self.thread.stop()
		event.accept()


	@pyqtSlot()
	def updateChessboard(self,chessboard):
		self.chessboardSvg = chess.svg.board(chessboard).encode("UTF-8")
		self.widgetSvg.load(self.chessboardSvg)


	@pyqtSlot(np.ndarray)
	def update_image(self, cv_img):
		"""Updates the image_label with a new opencv image"""
		qt_img = self.convert_cv_qt(cv_img)
		self.image_label.setPixmap(qt_img)

	def convert_cv_qt(self, cv_img):
		"""Convert from an opencv image to QPixmap"""
		cv_img = cv2.rotate(cv_img, cv2.ROTATE_180)
		rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
		h, w, ch = rgb_image.shape
		bytes_per_line = ch * w
		p = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
		return QPixmap.fromImage(p)

if __name__=="__main__":
	app = QApplication(sys.argv)
	app.setStyle('Fusion')
	a = App()
	a.show()
	sys.exit(app.exec_())


