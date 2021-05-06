"""
Alcune delle funzioni che utilizzerò all'interno del programma
sono presenti all'interno delle librerie da me definite :

- MyChessFunction , Contente le funzioni che fungono da interfaccia
tra OpenCV e il sistema di scacchi
-Calibration , per la calibrazione della camera e la creazione di set
per la medesima funzione
- Game_chess , contenente alcune funzioni utili per interfacciarsi con
il motore di scacchi
"""
# noinspection PyUnresolvedReferences
from MyChessFunction import *
# noinspection PyUnresolvedReferences
from Calibration import *
from game_chess import *



"""
Librerie esterne in uso all'interno del programma : 
-chess e chess.engine per la parte legata al motore di scacchi
-stockfish per le partite contro la CPU  
"""
import chess
import chess.engine
from stockfish import Stockfish

"""Tutte le librerie cui sotto sono state utilizzate per la
definizione dell'interfaccia e delle sue parti : 
Ho preferito utilizzare PyQt5 per la compatibilità rispetto ai file
SVG , formato nel quale la scacchiera viene codificata"""
from PyQt5.QtSvg import QSvgWidget
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QPushButton , QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import sys

"""Classe per la gestione della visione artificiale e per
la elaborazione delle immagini """
# noinspection PyUnresolvedReferences
import cv2

"""Classe per la gestione delle matrici presenti all'interno 
del programma e gestione numerica di una serie di funzioni """
import numpy as np


"""Variabili utili per il corretto funzionamento del programma :
per lo più flag , ma anche variabili globali che ho usato all'interno 
delle funzioni di gioco """

# Flag che indica se la scacchiera è stata trovata
chessboard_found = False
# Flag che indica se le case della scacchiera sono state trovate
boxes_found = False
# Variabile contente le coordinate dei punti che costiuiscono le case
chessboard_corners = None
# Variabile contente i quattro vertici della scacchiera
box_corners = None
# Immagine della scacchiera ( da webcam )
img_chessboard = None
# Variabile contente tutte le coordinate della scacchiera ( case comprese )
coordinates = None
# Flag per la gestione dei turni
isWhiteTurn = True
isBlackTurn = False
# Matrice per lo stato S0 per quanto riguarda entrambi i lati di gioco
oldblack = setBlack()
oldwhite = setWhite()
# Oggetto di tipo scacchiera , contenente lo stato di gioco
chessboard = chess.Board("8/8/8/8/8/8/8/8 w - - 0 1")
# Motore Stockfish , per la scelta delle mosse migliori e per la gestione delle partite vs CPU
stockfish = Stockfish ("stockfish-10-win\Windows\stockfish_10_x64")


"""Classe che estende un Thread per l'utilizzo della schermata
della webcam , che riprende la scacchiera , all'interno 
dell'interfaccia . Oltre al costruttore , sono presenti la funzione 
di stop per chiudere la finestra e quella di run , che elabora 
l'immagine e ottiene le informazioni utili relative al posizionamento 
della scacchiera e delle case """
class VideoThread(QThread):

	change_pixmap_signal = pyqtSignal(np.ndarray)

	# Costruttore della classe
	def __init__(self):
		super().__init__()
		self._run_flag = True

	# Corpo del thread
	def run(self):

		# Gestione webcam
		webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

		# Calibrazione webcam
		ret, mtx, dist, rvecs, tvecs, h, w = camera_calibration()
		newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

		# Ciclo per la cattura dei frame
		while self._run_flag:
			success, img = webcam.read()

			if success:

				# Rimuovere distorsione
				img = undistort(img, mtx, dist, newcameramtx, roi)

				# Pre-Elaborazione dell'immagine
				img_ = pre_processing(img)


				# Definizione delle variabili globali da utilizzare nel corpo del metodo
				global chessboard_found
				global chessboard_corners

				# Nel caso la scacchiera è stata già trovata , la ricerca viene conclusa
				if(chessboard_found == False):
					#Ricerca della scacchiera
					chessboard_found,  chessboard_corners = cv2.findChessboardCorners(img_, (7, 7), None)

				# Se la scacchiera è stata trovata
				if chessboard_found:

					# Variabile globale per l'immagine della scacchiera
					global img_chessboard

					# L'immagine della scacchiera viene estratta
					img_chessboard  = get_chessboard(img, chessboard_found, chessboard_corners)


					# Definizione nuove variabili globali in utilizzo
					global boxes_found
					global box_corners
					global coordinates

					# Ricerca delle case della scacchiera se queste non sono state mai trovate
					if boxes_found == False:
						#Ricerca della case all'interno della scacchiera
						boxes_found, box_corners = cv2.findChessboardCorners(pre_processing(img_chessboard), (7, 7), None)
						#Salvataggio delle coordinate all'interno di una struttura dati
						coordinates = get_final_coordinates(box_corners)

				# Salvataggio immagine per il thread
				self.change_pixmap_signal.emit(img_chessboard)

		# Rilascio risorse allocate al termine delle operazioni
		webcam.release()


	# Quando il programma viene terminato
	def stop(self):
		self._run_flag = False
		self.wait()


"""Classe per la definizione dell'interfaccia del programma"""
class App(QWidget):
	def __init__(self):
		super().__init__()

		"""Definizione della dimensione della finestra principale
		oltre alla definizione di una icona e di un nome per
		la stessa """
		self.setWindowTitle("Chess Computer System ")
		self.display_width = 1080
		self.display_height = 720
		self.setWindowIcon(QtGui.QIcon("horse.png"))

		"""Definizione di una finestra all'interno della quale 
		verrà visualizzata l'istanza relativa alla webcam , quindi 
		la scacchiera in live """
		self.image_label = QLabel(self)
		self.image_label.setGeometry(10,10,400,400)

		"""Definzione di una finestra all'interno della quale verà 
		visualizzata la scacchiera elaborata a partire da un file svg"""
		self.widgetSvg = QSvgWidget(parent=self)
		self.widgetSvg.setGeometry(950, 10, 400, 400)

		"""Definizione di un pulsante per la modalità di analisi: 
		ho utilizzto questa modalità per lo più per testare le funzioni 
		e la funzionalità di alcuni strumenti, tra cui il riconoscimento 
		dei pezzi all'interno della scacchiera """
		self.solitario = QPushButton('Gioco solitario', self)
		self.solitario.setToolTip('This is an example button')
		self.solitario.setGeometry(650, 70, 100, 40)
		self.solitario.clicked.connect(self.on_click_solitario)


		"""Definizione di un etichetta che rappresenterà la stringa delle 
		mosse eseguite : l'etichetta presenta dei contorni e un colore di 
		sfondo , oltre a delle ombre per rendere l'interfaccia più
		piacevole. La stampa delle stringhe all'interno della stessa parte 
		dall'alto a destra ( SetAligment ) """
		self.label = QLabel("Qui verranno stampate le tue mosse",self)
		self.label.frameShadow()
		self.label.setGeometry(950, 450, 400, 200)
		self.label.setStyleSheet("background-color: white; border: 1px solid black;")
		self.label.setAlignment(Qt.AlignLeading | Qt.AlignLeft | Qt.AlignTop)
		self.label.setWordWrap(True)
		self.label.hide()

		"""Definizione di un pulsante per la modalità di gioco : 
		Bianco vs CPU """
		self.White = QPushButton('Gioca con il bianco', self)
		self.White.setToolTip('This is an example button')
		self.White.setGeometry(650, 130, 100, 40)
		self.White.clicked.connect(self.play_as_white)
		"""Definizione di un pulsante per la modalità di gioco : 
		Nero  vs CPU """
		self.Black = QPushButton('Gioca con il nero', self)
		self.Black.setToolTip('This is an example button')
		self.Black.setGeometry(650, 190, 100, 40)
		self.Black.clicked.connect(self.play_as_black)


		"""Pulsante per confermare l'esecuzione di una mossa """
		self.button = QPushButton('Prossima mossa', self)
		self.button.setGeometry(650, 70, 100, 40)
		self.button.clicked.connect(self.on_click_next)
		self.button.hide()

		"""Pulsante per confermare l'esecuzione della mossa,
		utilizzato nella modalità di gioco Bianco Vs Cpu"""
		self.next_white = QPushButton('Prossima mossa', self)
		self.next_white.setGeometry(650, 70, 100, 40)
		self.next_white.clicked.connect(self.on_click_next_white)
		self.next_white.hide()

		"""Pulsante per effettuare il reset della scacchiera
		e dello stato di gioco (Ad esempio quando si vuole 
		 iniziare una nuova partita) """
		self.buttonReset = QPushButton('Reset', self)
		self.buttonReset.setGeometry(650, 120, 100, 40)
		self.buttonReset.clicked.connect(self.on_click_reset)
		self.buttonReset.hide()

		"""Codifica della scacchiera formato svg per la lettura 
		e scrittura all'interno della finestra preposta """
		self.chessboardSvg = chess.svg.board(chessboard).encode("UTF-8")
		self.widgetSvg.load(self.chessboardSvg)


		"""Utilizzo di un thread per la gestione del video 
		catturato dalla webcam che ritrae la scacchiera : è 
		necessario utilizzare un thread per non bloccare 
		l'interfaccia ( INR ) """
		self.thread = VideoThread()
		self.thread.change_pixmap_signal.connect(self.update_image)
		self.thread.start()


	@ pyqtSlot()
	def play_as_white(self):
		global chessboard

		# Estrazione delle case della matrice
		if boxes_found:
			boxes = boxes_matrix(img_chessboard, coordinates)

		# Creazione matrice posizionale
		matrix = find_pieces(boxes, "all")

		if isStart(matrix):
			# La scacchiera è nel suo stato iniziale
			chessboard = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

			# Aggiorna pulsanti
			self.solitario.hide()
			self.White.hide()
			self.Black.hide()
			self.next_white.show()
			self.buttonReset.show()
			self.label.show()

			# Aggiorno la scacchiera a schermo
			self.updateChessboard(chessboard)

		elif isEmpty(matrix):
			pass




	@pyqtSlot()
	def play_as_black(self):
		pass



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
			self.label.show()
			self.White.hide()
			self.Black.hide()

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
	def on_click_next_white(self):
		# Variabile globale da utilizzare
		global oldwhite

		# Estrazione delle case della matrice
		if boxes_found:
			boxes = boxes_matrix(img_chessboard, coordinates)


		# Estrazione matrice posizionale bianca
		matrix = find_pieces(boxes, "white")
		# Elaborazione mossa del bianco
		try:
			move = get_move(oldwhite, matrix, chessboard)
			chessboard.push_uci(move)
		except:
			print("Mossa non valida")
			return


		# Vittoria per il bianco
		if (chessboard.is_checkmate()):
			print("Il bianco ha vinto!")

		# Aggiorna la scacchiera a schermo
		self.updateChessboard(chessboard)
		# Sovrascrivere variabile dello stato precedente con il nuovo stato
		oldwhite = matrix


		fen = chessboard.fen()
		stockfish.set_fen_position(fen)

		best_move = stockfish.get_best_move()
		chessboard.push_uci(best_move)

		oldwhite = computer_black_move(best_move,oldwhite)

		self.on_update_moves(chessboard)

		# Aggiorna la scacchiera a schermo
		self.updateChessboard(chessboard)

	def on_update_moves(self,chessboard):

		stack_moves = ""
		index_move = 2

		for move in chessboard.move_stack:
			stack_moves = stack_moves + str(int(index_move/2)) + "." + chessboard.uci(move) + " "
			index_move = index_move + 1

		self.label.setText(stack_moves)

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

		#Se è il turno del bianco
		if isWhiteTurn:

			#Estrazione matrice posizionale bianca
			matrix = find_pieces(boxes, "white")


			try :
				#Elaborazione mossa del bianco
				move = get_move_single(oldwhite, matrix,chessboard,oldblack)
				chessboard.push_uci(move)
			except :
				print("Mossa non valida")
				return

			#Gestione pedone mangiato
			oldblack = whiteTake(matrix,oldblack)

			#Vittoria per il bianco
			if(chessboard.is_checkmate()):
				print("Il bianco ha vinto!")

			self.on_update_moves(chessboard)

			#Gestione dei turni
			isWhiteTurn = False
			isBlackTurn = True
			#Aggiorna la scacchiera a schermo
			self.updateChessboard(chessboard)
			#Sovrascrivere variabile dello stato precedente con il nuovo stato
			oldwhite = matrix
			return


		#Se è il turno del nero
		if isBlackTurn:
			#Estrazione matrice posizionale nera
			matrix = find_pieces(boxes, "black")

			try:
				# Elaborazione mossa del bianco
				move = get_move_single(oldblack, matrix, chessboard, oldwhite)
				chessboard.push_uci(move)
			except:
				print("Mossa non valida")
				return


			#Gestione pedone mangiato
			oldwhite = blackTake(matrix,oldwhite)

			#Vittoria per il bianco
			if(chessboard.is_checkmate()):
				print("Il nero ha vinto!")

			self.on_update_moves(chessboard)

			#Passa il turno
			isWhiteTurn = True
			isBlackTurn = False
			#Aggiorna la scacchiera a schermo
			self.updateChessboard(chessboard)
			#Sovrascrivere variabile dello stato precedente con il nuovo stato
			oldblack = matrix
			return

	"""
	@pyqtSlot()
	def winMessage(self,player):
		self.msg = QMessageBox()
		self.msg.setIcon(QMessageBox.Information)

		if(player == "white "):
			self.msg.setText("Il bianco ha vinto")
		else:
			self.msg.setText("Il nero ha vinto")

		self.msg.setStandardButtons(QMessageBox.Ok)
	"""

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


