import cv2

def draw_bbox(img,bbox_list,classes,names):
    for x,bbox in enumerate(bbox_list):
        start_point = (int(bbox[0]),int(bbox[1]))
        end_point = (int(bbox[2]),int(bbox[3]))
        cv2.rectangle(img, start_point, end_point, (255,0,0), 2)
        cv2.putText(img, names[classes[x]], (start_point[0] - 2, start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    
    return img