import pyscreenshot as ImageGrab
import time

images_folder = "Dataset/5/"

for i in range (0,39):
	
	time.sleep(7)
	im = ImageGrab.grab(bbox=(80, 80, 208, 208)) # X1,Y1,X2,Y2
	print("saved....",i)
	im.save(images_folder+str(i)+'.png')
	print("clear screen now and redraw now...")
	