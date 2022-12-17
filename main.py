import cv2
import numpy as np
from keras.models import load_model

def main():
    model_path: str = 'assets/mnist.h5'
    model = load_model(model_path)

    # image = cv2.imread('assets/seven.png', cv2.IMREAD_GRAYSCALE)
    img = cv2.imread('assets/pruebaNumeros.jpeg')	
    imgBN = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret,imthresh = cv2.threshold(imgBN,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    ret,imthresh = cv2.threshold(imgBN,127,255,cv2.THRESH_BINARY_INV)

   
    # imthresh = cv2.medianBlur(imthresh, 9)
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
    # cv2.imshow('unknown', unknown)
    # cv2.imshow('sure', sure_fg)
    # Marker labelling
    marker_count, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    print(marker_count)
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    segmented = cv2.watershed(img,markers)
    # END of original watershed example

    output = np.zeros_like(img)
    output2 = img.copy()
    array = []

    # Iterate over all non-background labels
    for i in range(2, marker_count + 1):
        mask = np.where(segmented==i, np.uint8(255), np.uint8(0))
        x,y,w,h = cv2.boundingRect(mask)
        area = cv2.countNonZero(mask[y:y+h,x:x+w])
        print("Label %d at (%d, %d) size (%d x %d) area %d pixels" % (i,x,y,w,h,area))
        img_crop = imthresh[y:y+h, x:x+w]
        img_crop = cv2.resize(img_crop, [28,28])
        array.append(img_crop)
        img_crop_array = np.array(array)


        # Visualize
        # color = np.uint8(np.random.random_integers(0, 255, 3)).tolist()
        color = np.uint8(np.random.randint(0, 255 + 1)).tolist()
        output[mask!=0] = color
        cv2.rectangle(output2, (x,y), (x+w,y+h), color, 1)
        # cv2.putText(output2,'%d'%i,(x+w/4, y+h/2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    cv2.imwrite('wshseg_colors.png', output)
    cv2.imwrite('wshseg_boxes.png', output2)

    image = cv2.resize(output2, [28, 28])

    # cv2.imshow('Umbralizada', imthresh)
    # cv2.imshow('Reescalada', image)
    # cv2.imshow('WaterShed', output)
    # cv2.imshow('Boxes', output2)
    cv2.waitKey(0)


    for i in range(len(img_crop_array)):
        image = np.array([img_crop_array[i]])
        
        prediction = model(image)
        predicted_number = np.argmax(prediction)
        print(predicted_number)

        cv2.imshow('Numero', img_crop_array[i])
        cv2.waitKey(0)

    # image = np.array([image])

    # # print(np.max(image))

    # prediction = model(image)
    # predicted_number = np.argmax(prediction)

    # print(predicted_number)



if __name__ == '__main__':
    main()