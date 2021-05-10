"""
Alcune delle funzioni che utilizzerò all'interno del programma
sono presenti all'interno delle librerie da me definite :

- MyChessFunction , Contente le funzioni che fungono da interfaccia
tra OpenCV e il sistema di scacchi
-Calibration , per la calibrazione della camera e la creazione di set
per la medesima funzione
- TeachableMachine , contente la parte relativa al machine learning
"""
# noinspection PyUnresolvedReferences
from MyChessFunction import *
# noinspection PyUnresolvedReferences
from Calibration import *

from teachableMachine import *



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
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QPushButton, QMessageBox, QToolTip, QMenuBar, \
	QMenu, QAction, QFrame, QProgressBar
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QTimer
import sys

"""Classe per la gestione della visione artificiale e per
la elaborazione delle immagini """
# noinspection PyUnresolvedReferences
import cv2

"""Classe per la gestione delle matrici presenti all'interno 
del programma e gestione numerica di una serie di funzioni """
import numpy as np

"""Utilizzato per la gestione della Splash Screen"""
from datetime import time


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


isLoading = True

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

		set_model()

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

		global isLoading
		isLoading = False

		"""Definizione della dimensione della finestra principale
		oltre alla definizione di una icona e di un nome per
		la stessa """
		self.setWindowTitle("Chess Computer System ")
		self.setObjectName("main_window")
		self.display_width = 1080
		self.display_height = 720
		self.setWindowIcon(QtGui.QIcon("horse.png"))


		"""Inserimento di un logo per riempire la interfacia"""
		self.logo = QLabel(self)
		self.logo.setGeometry(640,130,512,512)
		icon = QPixmap("horse.png")
		icon = icon.scaled(120,120)
		self.logo.setPixmap(icon)

		"""Contorno per l'immagine della webcam nella home"""
		self.backgroundweb = QLabel(self)
		self.backgroundweb.setGeometry(40,30,400,400)
		img_back = QPixmap("weback.png")
		img_back = img_back.scaled(400,400)
		self.backgroundweb.setPixmap(img_back)


		"""Definizione di una finestra all'interno della quale 
		verrà visualizzata l'istanza relativa alla webcam , quindi 
		la scacchiera in live """
		self.image_label = QLabel(self)
		self.image_label.setGeometry(55,45,370,370)
		self.image_label.setObjectName("LiveChessboard")
		self.image_label.setStyleSheet("border: 1px solid #202121;")




		"""Definzione di una finestra all'interno della quale verà 
		visualizzata la scacchiera elaborata a partire da un file svg"""
		self.widgetSvg = QSvgWidget(parent=self)
		self.widgetSvg.setGeometry(920, 30, 400, 400)



		"""Definizione di un pulsante per la modalità di analisi: 
		ho utilizzto questa modalità per lo più per testare le funzioni 
		e la funzionalità di alcuni strumenti, tra cui il riconoscimento 
		dei pezzi all'interno della scacchiera """
		self.solitario = QPushButton('Analisi partita', self)
		self.solitario.setToolTip('Motore di analisi')
		self.solitario.setGeometry(600, 70, 200, 40)
		self.solitario.clicked.connect(self.on_click_solitario)


		"""Definizione di un etichetta che rappresenterà la stringa delle 
		mosse eseguite : l'etichetta presenta dei contorni e un colore di 
		sfondo , oltre a delle ombre per rendere l'interfaccia più
		piacevole. La stampa delle stringhe all'interno della stessa parte 
		dall'alto a destra ( SetAligment ) """
		self.label = QLabel("Qui verranno mostrate le tue mosse",self)
		self.label.frameShadow()
		self.label.setGeometry(920, 450, 400, 200)
		self.label.setStyleSheet("""background-color: #eff4f7; border-radius:5px;
				border:1px solid #141010;font-family:Arial;
				font-size:12px;""")
		self.label.setAlignment(Qt.AlignLeading | Qt.AlignLeft | Qt.AlignTop)
		self.label.setWordWrap(True)
		self.label.hide()

		"""Definizione di un pulsante per la modalità di gioco : 
		Bianco vs CPU """
		self.White = QPushButton('Gioca con il bianco', self)
		self.White.setToolTip('Clicca per giocare contro il Computer')
		self.White.setGeometry(600, 130, 200, 40)
		self.White.clicked.connect(self.play_as_white)
		"""Definizione di un pulsante per la modalità di gioco : 
		Nero  vs CPU """
		self.Black = QPushButton('Gioca con il nero', self)
		self.Black.setToolTip('Clicca per giocare contro il Computer')
		self.Black.setGeometry(600, 190, 200, 40)
		self.Black.clicked.connect(self.play_as_black)


		"""Pulsante per confermare l'esecuzione di una mossa """
		self.button = QPushButton('Prossima mossa', self)
		self.button.setGeometry(600, 70, 200, 40)
		self.button.clicked.connect(self.on_click_next)
		self.button.hide()

		"""Pulsante per confermare l'esecuzione della mossa,
		utilizzato nella modalità di gioco Bianco Vs Cpu"""
		self.next_white = QPushButton('Prossima mossa', self)
		self.next_white.setGeometry(600, 70, 200, 40)
		self.next_white.clicked.connect(self.on_click_next_white)
		self.next_white.hide()

		"""Pulsante per effettuare il reset della scacchiera
		e dello stato di gioco (Ad esempio quando si vuole 
		 iniziare una nuova partita) """
		self.buttonReset = QPushButton('Reset', self)
		self.buttonReset.setGeometry(600, 120, 200, 40)
		self.buttonReset.clicked.connect(self.on_click_reset)
		self.buttonReset.hide()


		self.home = QPushButton("Torna alla home")
		self.home.setGeometry(600, 220, 200, 40)
		self.home.clicked.connect(self.on_click_reset)
		self.home.hide()


		"""Pulsante per effettuare la ricerca della scacchiera"""
		self.findChessboard = QPushButton('Ricerca Scacchiera', self)
		self.findChessboard.setGeometry(600, 250, 200, 40)
		self.findChessboard.clicked.connect(self.search_chessboard)



		"""Codifica della scacchiera formato svg per la lettura 
		e scrittura all'interno della finestra preposta """
		self.chessboardSvg = chess.svg.board(chessboard).encode("UTF-8")
		self.widgetSvg.load(self.chessboardSvg)

	"""Funzione che permette di tornare al menu iniziale 
	a partire da una schermata di gioco : vengono resettati i pulsanti 
	presenti nella home e viene resettata la scacchiera"""
	@pyqtSlot()
	def back_home(self):

		global isWhiteTurn
		global isBlackTurn
		global oldblack
		global oldwhite
		global chessboard

		isWhiteTurn = True
		isBlackTurn = False

		oldblack = setBlack()
		oldwhite = setWhite()

		chessboard = chess.Board("8/8/8/8/8/8/8/8 w - - 0 1")
		self.updateChessboard(chessboard)
		self.label.setText("")

		# Aggiorna pulsanti
		self.solitario.show()
		self.White.show()
		self.Black.show()
		self.next_white.hide()
		self.buttonReset.hide()
		self.label.hide()
		self.search_chessboard.show()
		self.home.hide()


	"""Funzione che permette di resettare i flag per la scacchiera :
	resettandoli riparte la ricerca della stessa """
	@pyqtSlot()
	def search_chessboard(self):

		global chessboard_found , boxes_found

		chessboard_found = False
		boxes_found = False


	"""Funzione che permette di iniziare la partita giocando con il 
	lato del bianco """
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
			self.findChessboard.hide()
			self.home.show()

			# Aggiorno la scacchiera a schermo
			self.updateChessboard(chessboard)

		elif isEmpty(matrix):
			self.msg = QMessageBox()
			self.msg.setWindowTitle("Tutorial on PyQt5")
			self.msg.setText("This is the main text!")
			self.msg.setIcon(QMessageBox.Question)
			self.msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Retry | QMessageBox.Ignore )
			self.msg.setDefaultButton(QMessageBox.Retry)
			self.msg.setInformativeText("informative text, ya!")

			self.msg.setDetailedText("details")


	@pyqtSlot()
	def play_as_black(self):
		pass


	"""Funzione che permette di giocare una modalità
	solitaria o analizzare una partita in corso , 
	salvare le mosse ..."""
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
			self.findChessboard.hide()

			#Aggiorno la scacchiera a schermo
			self.updateChessboard(chessboard)

		elif isEmpty(matrix):
			self.msg = QMessageBox()
			self.msg.setWindowTitle("OPS...")
			self.msg.setText("Inserisci i pezzi sulla scacchiera prima di cominciare.")
			self.msg.setIcon(QMessageBox.Information)
			self.msg.setStandardButtons(QMessageBox.Ok)
			self.msg.setWindowFlag(Qt.FramelessWindowHint)
			self.msg.show()


	"""Permette di resettare il gioco """
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
		self.label.setText("")

	"""Permette di passare da una mossa all'altra ,
	all'interno della modalità di gioco 
	bianco vs cpu """
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


	"""Aggiorna le mosse all'interno del label"""
	def on_update_moves(self,chessboard):

		stack_moves = ""
		index_move = 2

		for move in chessboard.move_stack:
			stack_moves = stack_moves + str(int(index_move/2)) + "." + chessboard.uci(move) + " "
			index_move = index_move + 1

		self.label.setText(stack_moves)


	"""Permette di passare da una mossa all'altra,
	all'interno della modalità solitario"""
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

	"""Aggiorna la scacchiera a schermo"""
	@pyqtSlot()
	def updateChessboard(self,chessboard):
		self.chessboardSvg = chess.svg.board(chessboard).encode("UTF-8")
		self.widgetSvg.load(self.chessboardSvg)

	"""Funzione per aggiornare l'immagine 
	catturata dalla webcam all'interno 
	della finestra presente nell'interfaccia """
	@pyqtSlot(np.ndarray)
	def update_image(self, cv_img):
		"""Updates the image_label with a new opencv image"""
		qt_img = self.convert_cv_qt(cv_img)
		self.image_label.setPixmap(qt_img)

	"""Funzione per convertire l'immagine gestita 
	tramite OpenCV in QpixMap """
	def convert_cv_qt(self, cv_img):
		#Rotazione necessaria per avere i pezzi bianchi sul proprio lato
		cv_img = cv2.rotate(cv_img, cv2.ROTATE_180)
		#Conversione in QImage
		rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
		h, w, ch = rgb_image.shape
		bytes_per_line = ch * w
		p = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
		return QPixmap.fromImage(p)



"""Definizione della schermata di caricamento"""
class SplashScreen(QWidget):
	def __init__(self):
		super().__init__()

		#Impostazione finestra principale
		self.setWindowTitle('Spash Screen')
		self.setFixedSize(1100, 500)

		#Rimozione barra titolo
		self.setWindowFlag(Qt.FramelessWindowHint)

		#Impostazione traslucenza
		self.setAttribute(Qt.WA_TranslucentBackground)

		#Contatore per il caricamento
		self.counter = 0
		self.n = 200

		#Inizializzazione interfaccia
		self.initUI()

		"""Utilizzo di un thread per la gestione del video 
		catturato dalla webcam che ritrae la scacchiera : è 
		necessario utilizzare un thread per non bloccare 
		l'interfaccia ( INR ) """
		self.thread = VideoThread()
		self.thread.change_pixmap_signal.connect(self.update_image)
		self.thread.start()


		#Gestione timer per il caricamento
		self.timer = QTimer()
		self.timer.timeout.connect(self.loading)
		self.timer.start(30)


	"""Funzione per la definizione degli elementi 
	presenti all'interno della interfaccia 
	della splash screen """
	def initUI(self):

		# Definizione tipologia di layout
		layout = QVBoxLayout()

		# Impostazione layout
		self.setLayout(layout)

		"""Definizione frame contenente la barra 
		di caricamento : il nome viene utilizzato all'interno
		di stylesheet , quindi all'interno del codice CSS """
		self.frame = QFrame()
		self.frame.setObjectName('FrameLoader')
		layout.addWidget(self.frame)

		# Definizione titolo della finestra
		self.labelTitle = QLabel(self.frame)
		self.labelTitle.setObjectName('LabelTitle')

		# Label centrale
		self.labelTitle.resize(self.width() - 10, 150)
		self.labelTitle.move(0, 40) # x, y
		self.labelTitle.setText('Chess Computer Vision System')
		self.labelTitle.setAlignment(Qt.AlignCenter)


		# Sottotitolo : cambia durante il caricamento
		self.labelDescription = QLabel(self.frame)
		self.labelDescription.resize(self.width() - 10, 50)
		self.labelDescription.move(0, self.labelTitle.height())
		self.labelDescription.setObjectName('LabelDesc')
		self.labelDescription.setText('<strong>Gestione intelligenza artificiale</strong>')
		self.labelDescription.setAlignment(Qt.AlignCenter)


		# ProgressBar per restituire un feedback per il caricamento
		self.progressBar = QProgressBar(self.frame)
		self.progressBar.resize(self.width() - 200 - 10, 50)
		self.progressBar.move(100, self.labelDescription.y() + 130)
		self.progressBar.setAlignment(Qt.AlignCenter)
		self.progressBar.setFormat('%p%')
		self.progressBar.setTextVisible(True)
		self.progressBar.setRange(0, self.n)
		self.progressBar.setValue(20)


		# Scritta che indica il processo in corso
		self.labelLoading = QLabel(self.frame)
		self.labelLoading.resize(self.width() - 10, 50)
		self.labelLoading.move(0, self.progressBar.y() + 70)
		self.labelLoading.setObjectName('LabelLoading')
		self.labelLoading.setAlignment(Qt.AlignCenter)
		self.labelLoading.setText('caricamento...')



	"""Come sopra , la gestione del thread è stata spostata 
	all'interno della splash screen , in modo tale da poter 
	caricare i contenuti necessari durante il caricamento """
	@pyqtSlot(np.ndarray)
	def update_image(self, cv_img):
		"""Updates the image_label with a new opencv image"""
		qt_img = self.convert_cv_qt(cv_img)
		if(not isLoading):
			self.myApp.image_label.setPixmap(qt_img)

	def convert_cv_qt(self, cv_img):
		"""Convert from an opencv image to QPixmap"""
		cv_img = cv2.rotate(cv_img, cv2.ROTATE_180)
		rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
		h, w, ch = rgb_image.shape
		bytes_per_line = ch * w
		p = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
		return QPixmap.fromImage(p)

	def closeEvent(self, event):
		if(not isLoading):
			self.thread.stop()
			event.accept()

	"""Funzione per l'aggiornamento della barra 
	di caricamento e i vari sottotitoli indicanti 
	il processo in corso """
	def loading(self):

		self.progressBar.setValue(self.counter)

		if self.counter == int(self.n * 0.3):
			self.labelDescription.setText('<strong>Gestione visione artificiale</strong>')
		elif self.counter == int(self.n * 0.6):
			self.labelDescription.setText('<strong>Caricamento motore di scacchi</strong>')
		elif self.counter >= self.n:
			self.timer.stop()
			self.close()


			self.myApp = App()
			self.myApp.show()

		self.counter += 1


"""Main dell'applicazione"""
if __name__=="__main__":
	app = QApplication(sys.argv)
	app.setStyleSheet('''
			#LabelTitle {
				font-size: 60px;
				color: #93deed;
			}

			#LabelDesc {
				font-size: 30px;
				color: #c2ced1;
			}

			#LabelLoading {
				font-size: 30px;
				color: #e8e8eb;
			}

			#FrameLoader {
				background-color: #2F4454;
				color: rgb(220, 220, 220);
			}

			QProgressBar {
				background-color: #DA7B93;
				color: rgb(200, 200, 200);
				border-style: none;
				border-radius: 10px;
				text-align: center;
				font-size: 30px;
			}

			QProgressBar::chunk {
				border-radius: 10px;
				background-color: qlineargradient(spread:pad x1:0, x2:1, y1:0.511364, y2:0.523, stop:0 #1C3334, stop:1 #376E6F);
			}
			
			
			#main_window{
				background-color : #2f4455;					
			}
				
			QPushButton{
				background:linear-gradient(to bottom, #8b9294 5%, #70787a 100%);
				background-color:#8b9294;
				border-radius:8px;
				border:2px solid #141010;
				display:inline-block;
				cursor:pointer;
				color:#ffffff;
				font-family:Arial;
				font-size:21px;
				text-decoration:none;
				text-shadow:0px 1px 0px #000000;
			}
			
			
			QPushButton:hover {
				background:linear-gradient(to bottom, #70787a 5%, #8b9294 100%);
				background-color:#70787a;
			}
			
			QPushButton:active {
				position:relative;
				top:1px;
			}
			
			
			#LiveChessboard {
				border : 3px gray;
			}
			
			QMessageBox{
				background-color: #2f4455;
				color:#ffffff;
				border : 2px solid black ; 
				border-radius: 10px;
				text-align: center;
				font-size: 15px;
				font-family:Arial;
			}
			
		''')
	splash = SplashScreen()
	splash.show()

	app.setStyle('Fusion')


	try:
		sys.exit(app.exec_())
	except SystemExit:
		print('Closing Window...')

