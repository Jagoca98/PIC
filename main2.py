import cv2
import numpy as np
import threading
import time
import math

class PIC:
    def thread_function(self, name):
        cap = cv2.VideoCapture('http://192.168.1.64:8080/video')
        self.img = cv2.imread('assets/ss.jpeg')
        # while(True):
        #     ret, frame = cap.read()
        #     self.img = frame
        #     if self.kill:
        #         break

    def main(self):
        self.kill = False

        thread = threading.Thread(target=self.thread_function, args=(1,))
        thread.start()

        time.sleep(2)

        width, height = 350, 250
        self.img = cv2.imread('assets/ss.jpeg')
        while(True):
            # img = cv2.imread('assets/pruebaNumeros.jpeg')	
            imgBN = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

            # create a CLAHE object (Arguments are optional).
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            imgBN = clahe.apply(imgBN)
            imthresh = imgBN
            ret,imthresh = cv2.threshold(imgBN,127,255,cv2.THRESH_BINARY)
            ret,imthresh = cv2.threshold(imgBN,200,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            # imthresh = cv2.medianBlur(imthresh,9)

            # Aplicar suavizado Gaussiano
            # gauss = cv2.GaussianBlur(imthresh, (5,5), 0)
        
            
            # Detectamos los bordes con Canny
            # canny = cv2.Canny(gauss, 50, 150)
        
            # cdstP = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
            
            # linesP = cv2.HoughLinesP(canny, 1, np.pi / 180, 50, None, 50, 10)
            
            # if linesP is not None:
            #     for i in range(0, len(linesP)):
            #         l = linesP[i][0]
            #         cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)


            corners = cv2.goodFeaturesToTrack(imthresh, 4, 0.4, 10, useHarrisDetector=True, k=0.04)
            
            corners = np.int0(corners)

            num = 0
            for i in corners:
                x,y = i.ravel()
                cv2.circle(imthresh, (x,y), 3, 255, -1)
                num = num+1


            if num >= 4:
                pts1 = np.float32([[corners[0].ravel()], [corners[1].ravel()], [corners[2].ravel()], [corners[3].ravel()]])
                pts2 = np.float32([[0,0], [width, 0], [height, 0], [width, height]])
                matrix = cv2.getPerspectiveTransform(pts1,pts2)
                img = cv2.warpPerspective(imthresh, matrix,(width, height))

            cv2.imshow("Crop", img)
            cv2.imshow('Imagen', imthresh)
            


            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.kill = True
                cv2.destroyAllWindows()
                break



if __name__ == '__main__':
    pic = PIC()
    pic.main()