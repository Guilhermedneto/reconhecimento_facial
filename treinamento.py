import cv2
import os
import numpy as np

eigenface = cv2.face.EigenFaceRecognizer_create(num_components=30, threshold=5)
fisherface = cv2.face.FisherFaceRecognizer_create(num_components=50, threshold=4)
lbph = cv2.face.LBPHFaceRecognizer_create()

def getImagemComId():
    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]
    faces = []
    ids = []
    for caminhoImagem in caminhos:
        imagemface = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminhoImagem)[-1].split('.')[1])
        #print(id)
        ids.append(id)
        faces.append(imagemface)
        cv2.imshow('Face', imagemface)
        cv2.waitKey(10)

    return np.array(ids), faces

ids, faces = getImagemComId()
#print(faces)

print('Treinando. . .')
eigenface.train(faces, ids)
eigenface.write('classificadorEigen.yml')

fisherface.train(faces, ids)
fisherface.write('classificadorFischer.yml')

lbph.train(faces, ids)
lbph.write('classificadorLBPH.yml')

print('Treinamento realizado')




getImagemComId()
