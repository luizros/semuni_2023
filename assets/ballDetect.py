import cv2
import numpy as np

video = cv2.VideoCapture('../assets/ballVideo.mp4')

def Processamento_Imagem(image):
    
    img_processada = cv2.GaussianBlur(image, (5, 5), 3)
    img_processada = cv2.Canny(img_processada, 90, 140)
    kernel = np.ones((6,6), np.uint8)       # Definir kernel para usar na dilatacao e erosao
    img_processada = cv2.dilate(img_processada, kernel, iterations=10)
    img_processada = cv2.erode(img_processada, kernel, iterations=10)   # Filtro mais "pesado"
    
    return img_processada

while True:
    ret, image = video.read()
    if not ret:
        break
    
    img_processada = Processamento_Imagem(image)
    
    contorno, hierarquia = cv2.findContours(img_processada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)      # Considerar apenas os contornos externos

    for cnt in contorno:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    cv2.imshow('Imagem', image)
    # cv2.imshow('Imagem Processada', img_processada)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()