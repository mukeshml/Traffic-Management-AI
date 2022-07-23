import cv2
import numpy as np
from tracker import *
# Initialize Tracker
tracker = EuclideanDistTracker()

middle_line_position = 325
up_line_position = middle_line_position - 15
down_line_position = middle_line_position + 15


font_color = (0, 255, 255)
font_size = 0.5
font_thickness = 2

# List for store vehicle count information
temp_up_list = []
temp_down_list = []
up_list = [0, 0, 0, 0]
down_list = [0, 0, 0, 0]

# class index for our required detection classes
required_class_index = [2, 3, 5, 7]

detected_classNames = []

nms_thres = 0.2
thres = 0.6
# img = cv2.imread('car.jpg')
cap= cv2.VideoCapture('lane5.mp4')
cap1 = cv2.VideoCapture('lane4.mp4')


# cap.set(3,640)
# cap.set(4,480)
# cap= cv2.VideoCapture(0)

classnames = []
classfile = 'coco.names'

with open(classfile, 'rt') as f:
    classnames = f.read().rstrip('\n').split('\n')

# print(classnames)

configpath ='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weights,configpath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def find_center(x, y, w, h):
    x1=int(w/2)
    y1=int(h/2)
    cx = x+x1
    cy=y+y1
    return cx, cy


def count_vehicle(box_id, img):
    x, y, w, h, id, index = box_id

    # Find the center of the rectangle for detection
    center = find_center(x, y, w, h)
    ix, iy = center

    # Find the current position of the vehicle
    if (iy > up_line_position) and (iy < middle_line_position):

        if id not in temp_up_list:
            temp_up_list.append(id)

    elif iy < down_line_position and iy > middle_line_position:
        if id not in temp_down_list:
            temp_down_list.append(id)

    elif iy < up_line_position:
        if id in temp_down_list:
            temp_down_list.remove(id)
            up_list[index] = up_list[index] + 1

    elif iy > down_line_position:
        if id in temp_up_list:
            temp_up_list.remove(id)
            down_list[index] = down_list[index] + 1

    # Draw circle in the middle of the rectangle
    cv2.circle(img, center, 2, (0, 0, 255), -1)  # end here
    # print(up_list, down_list)



def main():
    while True:
        success, img=cap.read()
        success1, img1=cap1.read()
        img = cv2.resize(img, (600,400))
        img1 = cv2.resize(img1, (600,400))
        ih, iw, channels = img.shape
        classids, confs, bbox = net.detect(img, confThreshold=thres)
        detection=[]
        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1, -1)[0])
        confs = list(map(float, confs))

        indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_thres)
        # print(indices)

        #
        # boxes_ids = tracker.update(detection)
        # for box_id in boxes_ids:
        #     count_vehicle(box_id, img)


        # Draw the crossing lines

        cv2.line(img, (0, middle_line_position), (iw, middle_line_position), (255, 0, 255), 1)
        cv2.line(img, (0, up_line_position), (iw, up_line_position), (0, 0, 255), 1)
        cv2.line(img, (0, down_line_position), (iw, down_line_position), (0, 0, 255), 1)



        for i in indices:

            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(img, (x, y), (x + w, h + y), color=(0, 255, 0), thickness=2)
            cv2.putText(img, classnames[classids[i] - 1], (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)


            names = classnames[classids[i] - 1]
            detected_classNames.append(names)
            detection.append([x, y, w, h, required_class_index.index(classids[i]-1)])


        boxes_ids = tracker.update(detection)
        for box_id in boxes_ids:

            count_vehicle(box_id, img)


        cv2.line(img, (0, middle_line_position), (iw, middle_line_position), (255, 0, 255), 2)
        cv2.line(img, (0, up_line_position), (iw, up_line_position), (0, 0, 255), 2)
        cv2.line(img, (0, down_line_position), (iw, down_line_position), (0, 0, 255), 2)

        # Draw counting texts in the frame
        cv2.putText(img, "  ", (110, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)

        cv2.putText(img, "Car:        " + str(up_list[0]) + "     " , (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Motorbike:  " + str(up_list[1]) + "     " , (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Bus:        " + str(up_list[2]) + "     " , (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Truck:      " + str(up_list[3]) + "     " ,(20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)

        # Show the frames
        cv2.imshow('Output', img)

        if cv2.waitKey(1) == ord('q'):
            break


        # for classid, confidence, box in zip(classids.flatten(), confs.flatten(), bbox):
        #     cv2.rectangle(img, box, color =(0,255,0), thickness=2)
        #     cv2.rectangle(img1, box, color=(0, 255, 0), thickness=2)
        #     cv2.putText(img, classnames[classid-1],(box[0]+10,box[1]+30),
        #                 cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0), 2)
        #     cv2.putText(img1, classnames[classid - 1], (box[0] + 10, box[1] + 30),
        #                 cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)





    cv2.imshow("Test 1", img)
    # cv2.imshow("Test 2", img1)

    cv2.waitKey(1)

if __name__ == '__main__':
    main()
    # cv2.destroyAllWindows()