import numpy as np
import cv2

def init(output_file, frame_rate, frame_width, frame_height):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, frame_rate, (frame_width, frame_height))
    return out

def update(out, width, height, x, y, theta, fov = False, beacons = True):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    offset = 100
    multiplier = 50
    #axis
    cv2.line(frame, (5*offset, 0), (5*offset, width), (255, 255, 255), 2)
    cv2.line(frame, (0, height-offset), (width, height-offset), (255, 255, 255), 2)
    #beacons
    if beacons:
        cv2.circle(frame, (0*multiplier+5*offset, height-0*multiplier-offset), 25, (0, 255, 0), -1)
        cv2.circle(frame, (10*multiplier+5*offset, height-0*multiplier-offset), 25, (0, 255, 0), -1)
    #robot
    cv2.circle(frame, (int(x*multiplier)+5*offset, height-int(y*multiplier)-offset), 25, (0, 0, 255), -1)
    #heading
    cv2.arrowedLine(frame, (int(x*multiplier)+5*offset, height-int(y*multiplier)-offset), (int((x+np.cos(theta))*multiplier)+5*offset, height-int((y+np.sin(theta))*multiplier)-offset), (255, 255, 255), thickness=2, tipLength=0.2)

    if fov:
        overlay = np.zeros(frame.shape, dtype=np.uint8)
        hdg = np.degrees(theta)
        if (hdg < 0):
            hdg += 360
        cv2.ellipse(overlay, (int(x*multiplier)+5*offset, height-int(y*50)-offset), (2000, 2000), 360-hdg, 315, 360, (255, 255, 255), thickness=-1)
        cv2.ellipse(overlay, (int(x*multiplier)+5*offset, height-int(y*50)-offset), (2000, 2000), 360-hdg, 0, 45, (255, 255, 255), thickness=-1)
        frame = cv2.addWeighted(frame, 1, overlay, 0.9, 0)

    out.write(frame)
    return out

def export(out):
    out.release()
    return