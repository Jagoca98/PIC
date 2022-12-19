import cv2
import numpy as np
import threading
import time
import math

class PIC:
    def thread_function(self, name):
        # cap = cv2.VideoCapture('http://192.168.1.64:8080/video')
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
            # Aplicar suavizado Gaussiano
            gauss = cv2.GaussianBlur(self.img, (5,5), 0)
            
            # cv2.imshow("suavizado", gauss)

            # imthresh = imgBN
            ret,imthresh = cv2.threshold(gauss,127,255,cv2.THRESH_BINARY)
            # ret,imthresh = cv2.threshold(gauss,200,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            # Detectamos los bordes con Canny
            canny = cv2.Canny(imthresh, 50, 150)

            # Marker labelling
            marker_count, markers = cv2.connectedComponents(canny)
            # Add one to all labels so that sure background is not 0, but 1
            markers = markers+1
            # Now, mark the region of unknown with zero
            # markers[unknown==255] = 0
            segmented = cv2.watershed(self.img,markers)
            # END of original watershed example
            

            output = np.zeros_like(self.img)
            output2 = self.img.copy()


            kernel = np.ones((3,3),np.uint8)
            dilation = cv2.dilate(canny.copy(),kernel,iterations = 2)
            opening = cv2.morphologyEx(dilation,cv2.MORPH_OPEN,kernel, iterations = 1)
            imthresh = cv2.medianBlur(opening, 9)

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
            print(index)
            mask = np.where(segmented==index, np.uint8(255), np.uint8(0))
            # color = np.uint8(np.random.randint(0, 255 + 1)).tolist()
            output3[:,:] = 0
            output3[mask!=0] = 255
            cv2.imshow('a', output3)
            #####################################################################

            # cv2.imshow('aa', opening)

            corners = cv2.goodFeaturesToTrack(opening, 4, 0.01, 10)
            corners = np.int0(corners)
            
            for i in corners:
                x,y = i.ravel()
                cv2.circle(output2, (x,y), 3, 255, -1)
            
            # cv2.imshow('corner', output2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.kill = True
                cv2.destroyAllWindows()
                break



if __name__ == '__main__':
    pic = PIC()
    pic.main()