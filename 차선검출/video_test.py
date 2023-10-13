import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

#출처 : https://pinkwink.kr/1264
path = "test_video.mp4"
cap=cv2.VideoCapture(path)

fps = cap.get(cv2.CAP_PROP_FPS)
width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
codec = cv2.VideoWriter_fourcc(*'mp4v')
out=cv2.VideoWriter('output_2.mp4', codec, 30.0, (int(width),int(height)))
print(width,height)
# img=mpimg.imread("idontknow.png")
# img = (img * 255).astype(np.uint8)
# img2=img.copy()
# print('img type:',type(img),'dimension:',img.shape)

def grayscale(img):
    return cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# gray=grayscale(img)

def gaussian_blur(img,kernel_size):
    return cv2.GaussianBlur(img,(kernel_size, kernel_size),0)
kernel_size=5
# blur_gray=gaussian_blur(gray,kernel_size)
# blur_gray = (blur_gray * 255).astype(np.uint8)

def canny(img,low_threshold,high_threshold):
    return cv2.Canny(img,low_threshold,high_threshold)
low_threshold=50
high_threshold=200
# edges=canny(blur_gray,low_threshold,high_threshold)
# print(len(img.shape))



# print(imshape)


def region_of_interest(img,vertices):
    mask = np.zeros_like(gray)

    if len(img.shape) >2 :
        channel_count=img.shape[2]
        ignore_mask_color=(255,)*channel_count
    else:
        ignore_mask_color =255
    cv2.fillPoly(mask,vertices,ignore_mask_color)

    masked_image=cv2.bitwise_and(img,mask)
    return masked_image

# imshape=img.shape
# vertices=np.array([[(10,400),(120,120),(440,120),(490,400)]],dtype=np.int32)
# mask = region_of_interest(edges,vertices)

def draw_lines(img,lines,color=[255,0,0],thickness=5):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img,(x1,y1),(x2,y2),color,thickness)

def hough_lines(img,rho,theta,threshold,min_line_len,max_line_gap):
    lines =cv2.HoughLinesP(img,rho,theta,threshold,np.array([]), minLineLength=min_line_len,maxLineGap=max_line_gap)
    line_img=np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8) #왜3으로 해야하니 내 shape은 4던뎅
    draw_lines(line_img,lines)
    return line_img
rho=2
theta=np.pi/180
threshold=90
min_line_len=120
max_line_gap=150
# lines=hough_lines(mask,rho,theta,threshold,min_line_len,max_line_gap)

# def weighted_img(img,initial_img, a=0.8,b=1,lamda=0):
#     return cv2.addWeighted(initial_img,a,img,b,lamda)
# lines_edges=weighted_img(lines,img2,a=0.8,b=1,lamda=0)
# lines_edges=cv2.addWeighted(img2,0.8,lines,1,0)
while cap.isOpened():
    ret,img = cap.read()
    img = (img * 255).astype(np.uint8)
    img2=img.copy()
    if not ret:
        print("프레임 수신불가. 종료중")
        break

    gray=grayscale(img)
    blur_gray=gaussian_blur(gray,kernel_size)
    blur_gray = (blur_gray * 255).astype(np.uint8)
    edges=canny(blur_gray,50,200)
    mask=np.zeros_like(gray)

    if len(img.shape) >2 :
        channel_count=img.shape[2]
        ignore_mask_color=(255,)*channel_count
    else:
        ignore_mask_color =255
    imgshape=img.shape
    vertices=np.array([[(10,600),(120,120),(680,120),(730,600)]],dtype=np.int32)
    cv2.fillPoly(mask,vertices,ignore_mask_color)
    mask=region_of_interest(edges, vertices)

    lines=hough_lines(mask,rho,theta,threshold,min_line_len,max_line_gap)
    # line_edges=weighted_img(lines,img,0.8,1,0)
    lines_edges=cv2.addWeighted(img2,0.8,lines,1,0)
    
    cv2.imshow('frame1',lines_edges)
    out.write(lines_edges)

    if cv2.waitKey(25) == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
