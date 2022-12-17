import cv2
import numpy as np
from keras.models import load_model
import threading
import time

class PIC:
    def thread_function(self, name):
        cap = cv2.VideoCapture('http://192.168.1.64:8080/video')
        while(True):
            ret, frame = cap.read()
            self.img = frame
            # time.sleep(0.5)
            # cv2.imshow('frame',frame)
            if self.kill:
                break

    def main(self):
        model_path: str = 'assets/mnist.h5'
        model = load_model(model_path)
        self.kill = False

        thread = threading.Thread(target=self.thread_function, args=(1,))
        thread.start()

        time.sleep(2)

        while(True):
            # img = cv2.imread('assets/pruebaNumeros.jpeg')	
            imgBN = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            ret,imthresh = cv2.threshold(imgBN,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            # ret,imthresh = cv2.threshold(imgBN,127,255,cv2.THRESH_BINARY_INV)

        
            # noise removal
            kernel = np.ones((3,3),np.uint8)
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
            array = []

            # Iterate over all non-background labels
            for i in range(2, marker_count + 1):
                mask = np.where(segmented==i, np.uint8(255), np.uint8(0))
                x,y,w,h = cv2.boundingRect(mask)
                area = cv2.countNonZero(mask[y:y+h,x:x+w])
                # print("Label %d at (%d, %d) size (%d x %d) area %d pixels" % (i,x,y,w,h,area))
                img_crop = imthresh[y:y+h, x:x+w]
                img_crop = cv2.resize(img_crop, [28,28])
                array.append(img_crop)
                # img_crop_array = np.array(array)
                image = np.array([img_crop])
                
                prediction = model(image)
                predicted_number = np.argmax(prediction)


                # Visualize
                color = np.uint8(np.random.randint(0, 255 + 1)).tolist()
                output[mask!=0] = color
                cv2.rectangle(output2, (x,y), (x+w,y+h), color, 1)
                cv2.putText(output2,'%d'%predicted_number,(round(x+w/4), round(y+h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            
            
            cv2.imshow('WaterShed', output)
            # cv2.imshow('Boxes', output2)
            # cv2.imshow('Boxes', output2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.kill = True
                cv2.destroyAllWindows()
                break


        cv2.imwrite('wshseg_colors.png', output)
        cv2.imwrite('wshseg_boxes.png', output2)

        # image = cv2.resize(output2, [28, 28])

        # cv2.imshow('Boxes', output2)
        # cv2.waitKey(0)



        # image = np.array([image])

        # # print(np.max(image))

        # prediction = model(image)
        # predicted_number = np.argmax(prediction)

        # print(predicted_number)



if __name__ == '__main__':
    pic = PIC()
    pic.main()