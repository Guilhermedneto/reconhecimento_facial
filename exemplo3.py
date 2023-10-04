import cv2
import numpy as np

video = cv2.VideoCapture(0)
classificadorFace = cv2.CascadeClassifier('Cascades\\haarcascade_frontalface_default.xml')
classificarolho = cv2.CascadeClassifier('Cascades\\haarcascade_eye.xml')
amostra = 1
numeroAmostras = 25
id = input('Digite seu identificador: ')
largura, altura = 220, 220
print('Capturando as faces...')

while True:
    conectado, frame = video.read()
    frameCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #print(np.average(frameCinza))
    facesDetectadas = classificadorFace.detectMultiScale(frameCinza)
    
    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255),2)
        regiao = frame[y:y + a, x:x + l]
        regiaoCinzaOlho = cv2.cvtColor(regiao,cv2.COLOR_BGR2GRAY)
        olhosDetectados = classificarolho.detectMultiScale(regiaoCinzaOlho)
        for (ox, oy, ol, oa) in olhosDetectados:
            cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                if np.average(frameCinza) > 110:
                    imagemFace = cv2.resize(frameCinza [y:y + a, x:x + l], (largura, altura))
                    cv2.imwrite('fotos/pessoa.' + str(id) +'.'+str(amostra) + '.jpg', imagemFace)
                    print('[foto ' + str(amostra) + ' capturada com sucesso]')
                    amostra += 1

    cv2.imshow('Video', frame)
    cv2.waitKey(1)
    if (amostra >= numeroAmostras + 1):
        break

print('Faces capturadas com sucesso')

video.release()
cv2.destroyAllWindows()
