

import cv2
import numpy as np

def main():
    thresh = 160

    #file = "building.png"
    #file = "im.jpg"
    #file = "blocks.png"
    #file = "checkerboard.png"
    #file = "junk.png"
    #file = "bub.png"
    #file = "cow.png"
    file = "n352604.jpg"
    
    #BGR
    #RGB
    ltblue = [127,127 , 255][::-1]
    black  = [0  , 0  , 0  ]
    blue   = [0  , 0  , 255][::-1] # Orange
    green  = [0  , 255, 0  ][::-1] # Red
    yellow = [255, 255, 0  ][::-1] # Purple
    orange = [255, 127, 0  ][::-1] # Blue
    purple = [127, 0  , 255][::-1] # yellow
    red    = [255, 0  , 0  ][::-1] # Green
    brick  = [127, 0  , 0  ][::-1]
    
    # colors used for the heat map, contrasting colors are places next to each other
    heatMapColors = [black,ltblue,blue,orange,purple,yellow,green,red,brick]
    heatMapColors = [black,blue,orange,purple,yellow,green,red,brick]
    
    image = cv2.imread(file)
    org   = cv2.imread(file)

    # Get size
    sizex = len(image[0])
    sizey = len(image)
    
    # convert to grayscale
    grayIm = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # gradient in X direction
    gradImX = cv2.Sobel(grayIm,cv2.CV_64F,1,0,ksize=3)

    # gradient in Y direction
    gradImY = cv2.Sobel(grayIm,cv2.CV_64F,0,1,ksize=3)

    # Pre calculated values to make this faster
    ixx = np.zeros((sizey, sizex ,1),np.float32)
    iyy = np.zeros((sizey, sizex ,1),np.float32)
    ixy = np.zeros((sizey, sizex ,1),np.float32)
    
    for i in range(len(grayIm)):
        for j in range(len(grayIm[0])):
            ixx[i][j] = gradImX[i][j]*gradImX[i][j]
            iyy[i][j] = gradImY[i][j]*gradImY[i][j]
            ixy[i][j] = gradImX[i][j]*gradImY[i][j]
    
    # Calculate lambda
    Lxx = np.zeros((sizey, sizex ,1),np.float32)
    Lyy = np.zeros((sizey, sizex ,1),np.float32)
    Lxy = np.zeros((sizey, sizex ,1),np.float32)
    
    rVals = np.zeros((sizey, sizex ,1),np.float32)
    print("[**] R Values (This takes the longest)")
    window = 3
    k=.05
    maxr=None
    minr=None
    max = [0,0]
    for y in range(len(grayIm)-window):
        for x in range(len(grayIm[0])-window):
            # Add up window 
            Lxx[y][x] = sumWindow(ixx,x,y,window)
            Lyy[y][x] = sumWindow(iyy,x,y,window)
            Lxy[y][x] = sumWindow(ixy,x,y,window)
            
            
            det         = (Lxx[y][x] * Lyy[y][x])-Lxy[y][x]*2
            trace       = Lxx[y][x] + Lyy[y][x]
            r           = det - k * (trace**2)
            rVals[y][x] = r
            
            # Find  max and min r values for changing ranges to 0-255
            if maxr is None:
                maxr=r
                max = [y,x]
            elif maxr<r:
                maxr=r
                max = [y,x]
            if minr is None:
                minr=r
            elif minr>r:
                minr=r
    print("[**] R Values finished")
    
    # Make an image that marks the largest Value
    cv2.circle(image,(max[1],max[0]), 3, (0,0,255), -1)
    
    print("[**] Ranging")
    # Make a new range         
    averageR = 0
    grayCornerness = np.zeros((len(grayIm),len(grayIm[0]),3),np.float32)
    for y in range(len(grayIm)):
        for x in range(len(grayIm[0])):
        
            rangedR = (rVals[y][x]-minr)*255/(maxr-minr)
            averageR += rangedR
            grayCornerness[y][x] = [rangedR]*3
            
    averageR = averageR/(len(grayIm)*len(grayIm[0]))
    
    # Dots on the original 
    for y in range(len(grayIm)):
        for x in range(len(grayIm[0])):
            if grayCornerness[y][x][0] > thresh:
                cv2.circle(org,(x,y), 3, (0,0,255), -1)
                
    # Look at a 100x100 windows and mark the highest
    markedWindows = rangedWindows(grayCornerness,100)
    print("[**] Ranging finished")
     
    # Make heatmap
    print("[**] Heatmap")
    heatmap2 = heatMap(grayCornerness,heatMapColors,averageR)

    print("average R:",averageR)
    cv2.imshow("Heatmap Cornerness", heatmap2/255)
    cv2.imshow("Original",org/255) 
    cv2.imshow("White hot Cornerness",grayCornerness/255)
    cv2.imshow("100x100 marked windows",markedWindows/255)
    cv2.imshow("Max over all",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # save for report
    cv2.imwrite("C:\\Users\\Owner\\Desktop\\UNCW\~ CLASSES\\Fall 2019\\340 Sci Comp\\HW2\\Heatmap Cornerness.png", heatmap2)
    cv2.imwrite("C:\\Users\\Owner\\Desktop\\UNCW\~ CLASSES\\Fall 2019\\340 Sci Comp\\HW2\\Original.png",org) 
    cv2.imwrite("C:\\Users\\Owner\\Desktop\\UNCW\~ CLASSES\\Fall 2019\\340 Sci Comp\\HW2\\White hot Cornerness.png",grayCornerness)
    cv2.imwrite("C:\\Users\\Owner\\Desktop\\UNCW\~ CLASSES\\Fall 2019\\340 Sci Comp\\HW2\\100x100 marked windows.png",markedWindows)
    cv2.imwrite("C:\\Users\\Owner\\Desktop\\UNCW\~ CLASSES\\Fall 2019\\340 Sci Comp\\HW2\\Max over all.png",image)
    #cv2.imwrite('C:/Users/N/Desktop/Test_gray.jpg', image_gray)

    
    
def sumWindow(Matrix,x,y,window):
    tmp = 0
    for i in range(y-window,y+1+window):
        for j in range(x-window,x+1+window):
            if i>0 and i<len(Matrix) and j>0 and j<len(Matrix[0]): 
                tmp += Matrix[i][j]
    return tmp

# Find the max number in a matrix's windows  
def rangedWindows(image,windowSize):
    imH = len(image)
    imW = len(image[0])
    
    markedIm = np.zeros((imH,imW,3),np.float32)
    coordList = []
    for y in range(0,(imH//windowSize)+1):
        for x in range(0,(imW//windowSize)+1):
            max = [0,0]
            for i in range(y*windowSize, y*windowSize+windowSize+1):
                for j in range(x*windowSize, x*windowSize+windowSize+1):
                    if i>=0 and i<len(markedIm) and j>=0 and j<len(markedIm[0]): 
                        if image[i,j][0]>image[max[0]][max[1]][0]:
                            max = [i,j]
                        markedIm[i][j] = image[i][j]
            coordList.append(max)
    for max in coordList:
        cv2.circle(markedIm,(max[1],max[0]), 3, (0,0,255), -1)
    return markedIm
    
def heatMap(image,colorList,averageR):
    assert len(colorList)>1
    # Make empty
    newIm = np.zeros((len(image),len(image[0]),3),np.float32)
    # Set dimension flag...in a crappy way haha
    thirdD = False
    try:
        image[0][0][0]
        thirdD = True
    except IndexError:
        pass
    # Make sub-range step increment 
    step = 256 / (len(colorList)-1)
    # Look at each pixel
    for i in range(len(image)):
        for j in range(len(image[0])):
            # Support for 2D lists and 3D. I didn't know what I wanted use
            if thirdD:
                item = image[i][j][0]
            else:
                item = image[i][j]
             
        
            # Look at each sub range
            for colorIndex in range(1,len(colorList)):
                # Check to see if number falls in current sub range
                if  (item <= step*colorIndex):
                    # Find multiplier to see how close a pixel is to the upper limit
                    x = (item-(step*(colorIndex-1)))*1/\
                        (step*colorIndex-(step*(colorIndex-1)))
                    # Use multiplier to mix the colors of the range limit
                    #   colors to get the new pixel color
                    newIm[i][j] =  [round(x*colorList[colorIndex][i]+\
                                    (1-x)*colorList[colorIndex-1][i])\
                                    for i in range(3)]
                    # Stop for loop
                    break
    return newIm
        
    
    
if __name__ == "__main__":
    main()
    
