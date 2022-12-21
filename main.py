import cv2
import numpy as np
from keras.models import load_model
import threading
import time

class PIC:
    def thread_function(self, name):
        cap = cv2.VideoCapture('http://192.168.1.64:8080/video')
        # self.img = cv2.imread('assets/ss.jpeg')
        while(True):
            ret, frame = cap.read()
            self.img = frame
            if self.kill:
                break

    def main(self):
        model_path: str = 'assets/mnist.h5'
        model = load_model(model_path)
        self.kill = False

        thread = threading.Thread(target=self.thread_function, args=(1,))
        thread.start()

        time.sleep(2)

        width, height = 720, 420
        sensitivity = 80

        puntos_anteriores = []
        p1 = [140, 150]
        p2 = [130, 390]
        p3 = [600, 150]
        p4 = [600, 380]
        puntos_anteriores= [p1,p2,p3,p4]
        puntos_anteriores2 = puntos_anteriores
        tolerancia = 100
        # self.img = cv2.imread('assets/ss.jpeg')
        while(True):
            try:
                hsv= cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
                
                # cv2.imshow('Original', self.img)


                lower_white = np.array([0, 0, 255-sensitivity])
                upper_white = np.array([255,sensitivity,255])

                maskwhite = cv2.inRange(hsv, lower_white, upper_white)

                res = cv2.bitwise_and(self.img,self.img, mask= maskwhite)

                # cv2.imshow('HSV', maskwhite)
                # cv2.imshow('aa',res)

                res = cv2.medianBlur(res, 5)
         
                # Aplicar suavizado Gaussiano
                gauss = cv2.GaussianBlur(res, (5,5), 0)
                ret,imthresh = cv2.threshold(gauss,127,255,cv2.THRESH_BINARY)
                # ret,imthresh = cv2.threshold(gauss,200,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

                # Detectamos los bordes con Canny
                canny = cv2.Canny(imthresh, 50, 150)

                kernel = np.ones((3,3),np.uint8)
                dilation = cv2.dilate(canny.copy(),kernel,iterations = 3)
                # cv2.imshow('dilate', dilation)
                dilation = cv2.medianBlur(dilation, 3)
                # cv2.imshow('dilate2', dilation)
                opening = cv2.morphologyEx(dilation,cv2.MORPH_OPEN,kernel, iterations = 3)
                # closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel, iterations = 10)

                # cv2.imshow('opened', opening)

                imthresh = cv2.medianBlur(opening, 3)

                ##################################################################

                opening = cv2.morphologyEx(imthresh,cv2.MORPH_OPEN,kernel, iterations = 1)
                # sure background area
                sure_bg = cv2.dilate(opening,kernel,iterations=4)
                # Finding sure foreground area
                dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
                dist_transform = cv2.distanceTransform(cv2.dilate(opening,kernel,iterations=3),cv2.DIST_L2,5)
                ret, sure_fg = cv2.threshold(dist_transform, 0.0*dist_transform.max(),255,0)
                # Finding unknown region
                sure_fg = np.uint8(sure_fg)
                unknown = cv2.subtract(sure_bg,sure_fg)
                # Marker labelling
                marker_count, markers = cv2.connectedComponents(sure_fg)
                # Add one to all labels so that sure background is not 0, but 1
                markers = markers+1
                if marker_count >= 5: marker_count=5
                # Now, mark the region of unknown with zero
                markers[unknown==255] = 0
                segmented = cv2.watershed(self.img,markers)
                # END of original watershed example
                
                output = np.zeros_like(self.img)
                output2 = self.img.copy()
                output3 = self.img.copy()
                original = self.img.copy()

                area_array = []

                # Iterate over all non-background labels
                for i in range(2, marker_count + 1):
                    mask = np.where(segmented==i, np.uint8(255), np.uint8(0))
                    x,y,w,h = cv2.boundingRect(mask)
                    area = cv2.countNonZero(mask[y:y+h,x:x+w])
                    area_array.append(area)
                    # print("Label %d at (%d, %d) size (%d x %d) area %d pixels" % (i,x,y,w,h,area))

                    # Visualize
                    color = np.uint8(np.random.randint(0, 255 + 1)).tolist()
                    output[mask!=0] = color
                    cv2.rectangle(output2, (x,y), (x+w,y+h), color, 1)
                
                index = area_array.index(max(area_array))+2
                # print(index)
                mask = np.where(segmented==index, np.uint8(255), np.uint8(0))
                # color = np.uint8(np.random.randint(0, 255 + 1)).tolist()
                output3[:,:] = 0
                output3[mask!=0] = 255
                output3 = cv2.erode(output3,kernel,iterations = 4)
                cv2.imshow('Borde', output3)
                #####################################################################

                # cv2.imshow('aa', opening) 
                opening2 = cv2.cvtColor(output3.copy(),cv2.COLOR_BGR2GRAY)
            
                corners = cv2.goodFeaturesToTrack(opening2, 4, 0.01, 10)
                corners = np.int0(corners)

                puntos = []
                puntos_corrected = []

                # if(corners.size ==4):
                for i in corners:
                    x,y = i.ravel()
                    puntos.append([x, y])
                    # cv2.circle(original, (x,y), 3, 255, -1)
                
                puntos = sorted(puntos, key=lambda k: [k[1]+2*k[0]])
                print('Puntos actuales', puntos)
                
                for i in range(4):
                    points = [round((puntos[i][0]+0.75*puntos_anteriores[i][0]+.25*puntos_anteriores2[i][0])/2), round((puntos[i][1]+0.75*puntos_anteriores[i][1]+0.25*puntos_anteriores2[i][1])/2)]
                    puntos_corrected.append(points)

                print('Puntos anteriores', puntos_anteriores)
                print('Puntos anteriores2', puntos_anteriores2)
                print('Puntos Corregidos', puntos_corrected)
                # for i in puntos:
                #     cv2.putText(original,'%d'%i,(i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 1, cv2.LINE_AA)
                
                puntos_anteriores2 = puntos_anteriores
                puntos_anteriores = puntos_corrected
                

                for i in puntos_anteriores:
                    cv2.circle(original, (i[0],i[1]), 3, 255, -1)


                # pts1 = np.float32([puntos[0], puntos[1], puntos[2], puntos[3]])
                pts1 = np.float32([puntos_corrected[0], puntos_corrected[1], puntos_corrected[2], puntos_corrected[3]])
                pts2 = np.float32([[0, 0], [0, height], [width, 0], [width, height]])
                matrix = cv2.getPerspectiveTransform(pts1, pts2)
                output4 = cv2.warpPerspective(self.img, matrix, (width, height))
                output4 = output4[int(0.05*height):int(0.95*height), int(0.05*width):int(0.95*width)]
                
                cv2.imshow('Original', original)
                # cv2.imshow('Recorte', output4)

            ##################################################################
                imgBN = cv2.cvtColor(output4, cv2.COLOR_BGR2GRAY)
            
                # create a CLAHE object (Arguments are optional).
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                imgBN = clahe.apply(imgBN)

                # ret,imthresh_folio = cv2.threshold(imgBN,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                ret,imthresh_folio = cv2.threshold(imgBN,150,255,cv2.THRESH_BINARY_INV)

                # cv2.imshow('BNbin' , imthresh_folio)
                # noise removal
                kernel = np.ones((3,3),np.uint8)
                # opening = cv2.morphologyEx(imthresh_folio,cv2.MORPH_OPEN,kernel, iterations = 1)
                # sure background area
                sure_bg = cv2.dilate(imthresh_folio,kernel,iterations=2)
                # cv2.imshow('Sure bg', sure_bg)
                # Finding sure foreground area
                dist_transform = cv2.distanceTransform(imthresh_folio,cv2.DIST_L2,5)
                dist_transform = cv2.distanceTransform(cv2.dilate(imthresh_folio,kernel,iterations=3),cv2.DIST_L2,5)
                ret, sure_fg = cv2.threshold(dist_transform, 0*dist_transform.max(),255,0)
                # Finding unknown region
                sure_fg = np.uint8(sure_fg)

                # cv2.imshow('Sure fg', sure_fg)

                unknown = cv2.subtract(sure_bg,sure_fg)
                # cv2.imshow('Unknown', unknown)
                # Marker labelling
                marker_count, markers = cv2.connectedComponents(sure_fg)
                # Add one to all labels so that sure background is not 0, but 1
                markers = markers+1
                if marker_count >= 5: marker_count=5
                # Now, mark the region of unknown with zero
                markers[unknown==255] = 0
                segmented = cv2.watershed(output4,markers)
                # END of original watershed example

                output5 = np.zeros_like(output4)
                output6 = output4.copy()
                array = []

                # Iterate over all non-background labels
                for i in range(2, marker_count + 1):
                    mask = np.where(segmented==i, np.uint8(255), np.uint8(0))
                    x,y,w,h = cv2.boundingRect(mask)
                    area = cv2.countNonZero(mask[y:y+h,x:x+w])
                    # print("Label %d at (%d, %d) size (%d x %d) area %d pixels" % (i,x,y,w,h,area))
                    
                    img_crop = imthresh_folio[y:y+h, x:x+w]
                    img_crop = cv2.resize(img_crop, [28,28])
                    array.append(img_crop)
                    # img_crop_array = np.array(array)
                    image = np.array([img_crop])
                    
                    prediction = model(image)
                    predicted_number = np.argmax(prediction)


                    # Visualize
                    color = np.uint8(np.random.randint(0, 255 + 1)).tolist()
                    output5[mask!=0] = color
                    cv2.rectangle(output6, (x,y), (x+w,y+h), color, 1)
                    cv2.putText(output6,'%d'%predicted_number,(round(x), round(y+h)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


                # cv2.imwrite('wshseg_colors.png', output5)
                # cv2.imwrite('wshseg_boxes.png', output6)
                # cv2.imshow('wshed', output5)
                cv2.imshow('Predicction', output6)

            except:
                continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.kill = True
                cv2.destroyAllWindows()
                break



if __name__ == '__main__':
    pic = PIC()
    pic.main()