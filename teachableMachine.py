import tensorflow.keras
from PIL import Image, ImageOps
from MyChessFunction import *

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')


# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

def find_pieces(boxes , str ):

    positional_matrix = [[ 0 for x in range(8)] for y in range(8)]
    x = 0
    y = 0
    for box in boxes:

        size = (224, 224)

        # Conversione da OpenCV a PIL
        box = cv2.cvtColor(box, cv2.COLOR_BGR2RGB)
        box = Image.fromarray(box)

        # Ridimensionamento immagine
        image = ImageOps.fit(box, size, Image.ANTIALIAS)

        # Immagine convertita in array
        image_array = np.asarray(image)


        # Normalizzazione
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        prediction = model.predict(data)

        if str == "all" :
            # Ricerca tutti i pezzi posizionati sulla scacchiera
            positional_matrix[y][x] = get_prediction(prediction)
        elif str == "white" :
            #Ricerca solo i pezzi bianchi posizionati sulla scacchiera
            positional_matrix[y][x] = get_white_prediction(prediction)
        elif str == "black" :
            #Ricerca solo i pezzi neri posizionati sulla scacchiera
            positional_matrix[y][x] = get_black_prediction(prediction)


        #Gestione un pò caotica per l'inserimento degli elementi nella matrice
        x,y = set_index(x,y)

        #Chiusura del ciclo
        if y == 8 : break

    print_positional_matrix(positional_matrix)
    return positional_matrix

def get_prediction(prediction):
    if (prediction[0][0] < prediction[0][1]) or (prediction[0][0] < prediction[0][2]):
        return 1
    else : return 0

def get_black_prediction(prediction):
    if (prediction[0][1] > prediction [0][0]) and (prediction[0][1] > prediction [0][2]) :
        return 1
    else : return  0

def get_white_prediction(prediction):
    if (prediction[0][2] > prediction [0][0]) and (prediction[0][2] > prediction [0][1]) :
        return 1
    else : return  0

def set_index(x,y):
    if x == 7 :
        x = 0
        y = y + 1
        return x,y

    x = x + 1

    return x,y