import multiprocessing as mp
from camera import read_stream
from ultralytics import YOLO


rtsp_links = []

 
print("Enter the number of cameras:")

no_cam = int(input())

for i in range(no_cam):
    print("Enter rtsp link of camera "+str(i+1))
    link = input()
    if len(link)<4: 
        link = int(link)
    rtsp_links.append(link)


processes = [mp.Process(target=read_stream, args=(stream,)) for stream in rtsp_links]


for process in processes:
    process.start()

for process in processes:
    process.join()




