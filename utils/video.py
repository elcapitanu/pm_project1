import numpy as np
import cv2

def init(output_file, frame_rate, frame_width, frame_height):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, frame_rate, (frame_width, frame_height))
    return out

def update(out, t, width, height, pred, real, fov = False, landmarks = True):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    offset = 100
    multiplier = 50

    #time
    cv2.putText(frame, str(t), (frame.shape[1] - 100, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,  (255, 255, 255), 2, lineType=cv2.LINE_AA)
    #axis
    cv2.line(frame, (5*offset, 0), (5*offset, width), (255, 255, 255), 2)
    cv2.line(frame, (0, height-offset), (width, height-offset), (255, 255, 255), 2)
    
    #landmarks
    if landmarks:
        cv2.circle(frame, (0*multiplier+5*offset, height-0*multiplier-offset), 25, (0, 255, 0), -1)
        cv2.circle(frame, (10*multiplier+5*offset, height-0*multiplier-offset), 25, (0, 255, 0), -1)
    #robot
    cv2.circle(frame, (int(pred[0]*multiplier)+5*offset, height-int(pred[1]*multiplier)-offset), 25, (0, 0, 255), -1)
    cv2.circle(frame, (int(real[0]*multiplier)+5*offset, height-int(real[1]*multiplier)-offset), 25, (255, 0, 0), -1)
    #heading
    cv2.arrowedLine(frame, (int(pred[0]*multiplier)+5*offset, height-int(pred[1]*multiplier)-offset), (int((pred[0]+np.cos(pred[2]))*multiplier)+5*offset, height-int((pred[1]+np.sin(pred[2]))*multiplier)-offset), (255, 255, 255), thickness=2, tipLength=0.2)
    cv2.arrowedLine(frame, (int(real[0]*multiplier)+5*offset, height-int(real[1]*multiplier)-offset), (int((real[0]+np.cos(real[2]))*multiplier)+5*offset, height-int((real[1]+np.sin(real[2]))*multiplier)-offset), (255, 255, 255), thickness=2, tipLength=0.2)
    #fov
    if fov:
        overlay = np.zeros(frame.shape, dtype=np.uint8)
        hdg = np.degrees(real[2])
        if (hdg < 0):
            hdg += 360
        cv2.ellipse(overlay, (int(real[0]*multiplier)+5*offset, height-int(real[1]*50)-offset), (2000, 2000), 360-hdg, 315, 360, (255, 255, 255), thickness=-1)
        cv2.ellipse(overlay, (int(real[0]*multiplier)+5*offset, height-int(real[1]*50)-offset), (2000, 2000), 360-hdg, 0, 45, (255, 255, 255), thickness=-1)
        frame = cv2.addWeighted(frame, 1, overlay, 0.9, 0)

    #legend
    cv2.putText(frame, "Pred", (frame.shape[1] - 200, 110), cv2.FONT_HERSHEY_SIMPLEX, 1,  (255, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.circle(frame, (frame.shape[1] - 250, 100), 25, (0, 0, 255), -1)
    cv2.putText(frame, "Real", (frame.shape[1] - 200, 210), cv2.FONT_HERSHEY_SIMPLEX, 1,  (255, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.circle(frame, (frame.shape[1] - 250, 200), 25, (255, 0, 0), -1)
    if landmarks:
        cv2.putText(frame, "Landmarks", (frame.shape[1] - 200, 310), cv2.FONT_HERSHEY_SIMPLEX, 1,  (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.circle(frame, (frame.shape[1] - 250, 300), 25, (0, 255, 0), -1)

    out.write(frame)
    return out

def update_2(out, t, width, height, pred, real, fov, landmarks, beacons):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    offset = 100
    multiplier = 50

    #time
    cv2.putText(frame, str(t), (frame.shape[1] - 100, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,  (255, 255, 255), 2, lineType=cv2.LINE_AA)
    #axis
    cv2.line(frame, (5*offset, 0), (5*offset, width), (255, 255, 255), 2)
    cv2.line(frame, (0, height-offset), (width, height-offset), (255, 255, 255), 2)
    
    #landmarks
    if landmarks:
        for beacon in beacons:
            cv2.circle(frame, (int(beacon[0])*multiplier+5*offset, height-int(beacon[1])*multiplier-offset), 25, (0, 255, 0), -1)
            cv2.circle(frame, (int(beacon[1])*multiplier+5*offset, height-int(beacon[1])*multiplier-offset), 25, (0, 255, 0), -1)
    #robot
    cv2.circle(frame, (int(pred[0]*multiplier)+5*offset, height-int(pred[1]*multiplier)-offset), 25, (0, 0, 255), -1)
    cv2.circle(frame, (int(real[0]*multiplier)+5*offset, height-int(real[1]*multiplier)-offset), 25, (255, 0, 0), -1)
    #heading
    cv2.arrowedLine(frame, (int(pred[0]*multiplier)+5*offset, height-int(pred[1]*multiplier)-offset), (int((pred[0]+np.cos(pred[2]))*multiplier)+5*offset, height-int((pred[1]+np.sin(pred[2]))*multiplier)-offset), (255, 255, 255), thickness=2, tipLength=0.2)
    cv2.arrowedLine(frame, (int(real[0]*multiplier)+5*offset, height-int(real[1]*multiplier)-offset), (int((real[0]+np.cos(real[2]))*multiplier)+5*offset, height-int((real[1]+np.sin(real[2]))*multiplier)-offset), (255, 255, 255), thickness=2, tipLength=0.2)
    #fov
    if fov:
        overlay = np.zeros(frame.shape, dtype=np.uint8)
        hdg = np.degrees(real[2])
        if (hdg < 0):
            hdg += 360
        cv2.ellipse(overlay, (int(real[0]*multiplier)+5*offset, height-int(real[1]*50)-offset), (2000, 2000), 360-hdg, 315, 360, (255, 255, 255), thickness=-1)
        cv2.ellipse(overlay, (int(real[0]*multiplier)+5*offset, height-int(real[1]*50)-offset), (2000, 2000), 360-hdg, 0, 45, (255, 255, 255), thickness=-1)
        frame = cv2.addWeighted(frame, 1, overlay, 0.9, 0)

    #legend
    cv2.putText(frame, "Pred", (frame.shape[1] - 200, 110), cv2.FONT_HERSHEY_SIMPLEX, 1,  (255, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.circle(frame, (frame.shape[1] - 250, 100), 25, (0, 0, 255), -1)
    cv2.putText(frame, "Real", (frame.shape[1] - 200, 210), cv2.FONT_HERSHEY_SIMPLEX, 1,  (255, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.circle(frame, (frame.shape[1] - 250, 200), 25, (255, 0, 0), -1)
    if landmarks:
        cv2.putText(frame, "Landmarks", (frame.shape[1] - 200, 310), cv2.FONT_HERSHEY_SIMPLEX, 1,  (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.circle(frame, (frame.shape[1] - 250, 300), 25, (0, 255, 0), -1)

    out.write(frame)
    return out


def export(out):
    out.release()
    return