import cv2


# Mostra i landmarks sul volto
def coordinate_draw(frame, land, frame_h, frame_w):
    x = int(land.x * frame_w)
    y = int(land.y * frame_h)
    cv2.circle(frame, (x, y), 3, (0, 255, 255))
    return x, y


def coordinate(landmark, frame_h, frame_w):
    x = int(landmark.x * frame_w)
    y = int(landmark.y * frame_h)
    return x, y


# Restituisce l'apertura degli occhi o della bocca in base al parametro passato
def width(y1, y2):
    return y1 - y2
