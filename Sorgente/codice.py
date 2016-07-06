
# coding: utf-8

# In[1]:

import cv2
import numpy as np
import pyglet


MISURAZIONI = 10
SOGLIA = 10

stato_occhi = {'rilevamenti': [2]*MISURAZIONI, 'occhi_chiusi': False}

def check(occhi_trovati):
    
    stato_occhi['rilevamenti'].append(occhi_trovati)
    stato_occhi['rilevamenti'].pop(0)
    numero_occhi = sum(stato_occhi['rilevamenti'])
    
    if numero_occhi > SOGLIA:
        stato_occhi['occhi_chiusi'] = False
    else:
        stato_occhi['occhi_chiusi'] = True
        
        
def cambia_colore(immagine, pixels, B, G, R):
    for (w,k) in pixels:
        immagine[w][k] = [B, G, R]
        
        
def draw(img1, img2, i, j):          # i = coordinata verticale, j = orizzontale
    h,w, _ = img2.shape
    img1[i:i+h, j:j+w] = img2[:h, :w]
    
    
def main_face(facce):                # restituisce i parametri della faccia più estesa rilevata nell'immagine
    fx = facce[0][0]
    fy = facce[0][1]
    fw = facce[0][2]
    fh = facce[0][3]
    
    for [x, y, w, h] in facce:
        if w*h > fw*fh:
            fx = x
            fy = y
            fw = w
            fh = h
    return fx, fy, fw, fh

        
face_cascade = cv2.CascadeClassifier('../Assets/lib/faccia.xml')
eye_cascade = cv2.CascadeClassifier('../Assets/lib/occhiali.xml')
        
cv2.VideoCapture(0)
cap = cv2.VideoCapture(0) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

quadro = cv2.imread('../Assets/Img/quadro.png')
start = cv2.imread('../Assets/Img/start.png')
guida_sicura = cv2.imread('../Assets/Img/guida_sicura.png')
guida_distratta = cv2.imread('../Assets/Img/distrazione.png')
pericolo = cv2.imread('../Assets/Img/Pericolo.png')
triangolo = cv2.imread('../Assets/Img/Allarme_sonno.png')
triangolo_giallo = cv2.imread('../Assets/Img/attenzione_giallo.png')
occhi_aperti = cv2.imread('../Assets/Img/occhi_aperti.png')
sound = pyglet.media.load('../Assets/audio/avviso.mp3')
sound2 = pyglet.media.load('../Assets/audio/avviso2.mp3')
pixels = np.loadtxt('../Assets/files/pixels_quadro.txt', dtype = 'int')


draw(quadro, start, 100, 370)

looper = pyglet.media.SourceGroup(sound.audio_format, None)          #crea looper audio (occhi chiusi)
looper.loop = True
looper.queue(sound)
avviso = pyglet.media.Player()
avviso.queue(looper)

looper2 = pyglet.media.SourceGroup(sound2.audio_format, None)          #crea looper audio 2 (distrazione)
looper2.loop = True
looper2.queue(sound2)
avviso2 = pyglet.media.Player()
avviso2.queue(looper2)


while cv2.waitKey(30) != ord('s'):
    cv2.imshow('C.C.A.S.A.: Car Concentration And Security Assistant', quadro)
else:
    draw(quadro, guida_sicura, 100, 370) 
    cambia_colore(quadro, pixels, 150, 210, 190)
    
while cv2.waitKey(30) != ord('q'):
    ret, camera = cap.read()
        
    gray = cv2.cvtColor(camera, cv2.COLOR_BGR2GRAY)
    faccia = face_cascade.detectMultiScale(gray, 1.3, 5)   
    
    
    if len(faccia) != 0:
        fx, fy, fw, fh = main_face(faccia)
        
        cv2.rectangle(camera,(fx,fy),(fx+fw,fy+fh),(150,210,190), 1)   
        
        roi_gray = gray[fy:fy+fh, fx:fx+fw]
        roi_color = camera[fy:fy+fh, fx:fx+fw]
        
        occhi = eye_cascade.detectMultiScale(roi_gray, maxSize=(int(fw/5),int(fh/5)), minSize=(int(fw/7),int(fh/7)))
        
        check(len(occhi))
        
        for [ex,ey,ew,eh] in occhi:                                   
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh), (150,210,190), 1)
        
        avviso2.pause()
        draw(quadro, guida_sicura, 100, 370)
        draw(quadro, occhi_aperti, 500, 455)
        
    else:
        draw(quadro, guida_distratta, 100, 370)
        draw(quadro, triangolo_giallo, 500, 455)
        avviso2.play()
        
    if stato_occhi['occhi_chiusi']:
        avviso.play()
        draw(quadro, pericolo, 100, 370)
        draw(quadro, triangolo, 500, 455)
    else:
        avviso.pause()
        
        
    draw(quadro, camera, 235, 375)        
    cv2.imshow('C.C.A.S.A.: Car Concentration And Security Assistant', quadro)

avviso.pause()
avviso2.pause()
        
cap.release()
cv2.destroyAllWindows()


# In[ ]:




# In[ ]:



