import math
import numpy as np
from PIL import Image

IMG_SIZE = 384

def parse_points_list(string):
    if string == "[]":
        return []
    return [parse_points(x) for x in string[1:-2].split('], ')][0]

def parse_points(string):
    string = string.replace('[', "")
    string = string.replace('(', "")
    string = string.replace(')', "")
    lst = [x for x in string.split(', ')]
    # return pairs of tuples
    points = []
    for i in range(0, len(lst), 2):
        x = math.floor(float(lst[i]))
        y = math.floor(float(lst[i+1]))
        points.append((x, y))
    return points

def open_image(filename):
    img = Image.open(filename)
    img = img.convert(mode="RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), resample=Image.NEAREST)
    img = np.asarray(img)
    img = img/255.0
    img = img.astype(np.float32)
    return img