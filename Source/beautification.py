import cv2
import numpy as np
import os
import math

face_cascade = cv2.CascadeClassifier('../Cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../Cascades/haarcascade_eye.xml')


parser = argparse.ArgumentParser()
parser.add_argument('-pic', type=str)
parser.add_argument('-pic_new',type=str)
args = parser.parse_args()

def square(x):
	return x * x

def distance_sqr(x1, y1, x2, y2):
	return square(x1-x2) + square(x2-y2)

def face_detector(image, cascade, cascade2):
	grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	equalImage = cv2.equalizeHist(grayImage)
	faces = cascade.detectMultiScale(equalImage, scaleFactor=1.3, minNeighbors=2)

	for(x,y,w,h) in faces:
		image1 = image[y:y+h, x:x+w]
		image_high = image1
		eyes = cascade2.detectMultiScale(image1)
		for (ex, ey, ew, eh) in eyes:
			center_x = ex + ew * 0.5
			center_y = ey + eh * 0.5
			eyes1 = image1[ey:ey+eh, ex:ex+ew]
			eyes2 = eyes1
			kernel_radius = min(ew, eh) * 0.4
			for r in range(eh):
				for c in range(ew):
					diff_x = c - ew*0.5
					diff_y = r - eh*0.5
					distance = math.sqrt(diff_x * diff_x + diff_y * diff_y)
					p_x = 0
					p_y = 0
					if distance <= kernel_radius:
						re = (1 - math.cos(distance / kernel_radius * 2 * math.pi)) * 2.5
						p_x = -diff_x * (re / kernel_radius)
						p_y = -diff_y * (re / kernel_radius)
					if p_x < 0 : 
						p_x  = 0
					if p_y < 0 : 
						p_y = 0
					eyes2[r,c] = eyes1[int(r + p_y),int(c + p_x)]
			image1[ey:ey+eh, ex:ex+ew] = eyes2	
		image_high1 = cv2.bilateralFilter(image_high, 15, 37, 37)
		image_high2 = image_high1 - image1 + 128 
		image_high3 = cv2.GaussianBlur(image_high1,(1, 1),0)
		image_high4 = image1 + 2 * image_high3 - 255
		final = image1 * 0.45 + image_high4 * 0.55
		c_x = x + w * 0.5
		c_y = y + h * 0.5
		radius = min(w, h) * 2
		image_high4 = image_high3
		for row in range(h):
			for col in range(w):
				diff_x = col - w * 0.5
				diff_y = col - h * 0.5
				distance = math.sqrt(square(col - w*0.5) + square(row - h*0.5))
				m_x = 0
				m_y = 0
				if distance <= radius:
					re = (1 - math.cos(distance / radius * 2 * math.pi)) * 2
					m_x = -diff_x * (re / radius)
					m_y = -diff_y * (re / radius)
				if m_x < 0:
					m_x = 0
				if m_y < 0:
					m_y = 0
				image_high4[row,col] = image_high3[int(row + m_y), int(col + m_x)]
		image[y:y+h, x:x+w] = image_high4
	return image

img = cv2.imread("PHOTO/"+args.pic+".jpg")
show_image = face_detector(img, face_cascade, eye_cascade)
cv2.imwrite('PHOTO/'+args.pic_new+'.jpg', show_image)



