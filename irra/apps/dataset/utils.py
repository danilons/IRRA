import cv2
import base64


def image2base64(image):
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer)