import cv2 as cv
import numpy as np
from google.colab.patches import cv2_imshow
import math

BODY_PARTS = { "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13,
                  "Background": 18 }

# POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
#                ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
#                ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
#                ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]

# Right ARM - Straight
POSE_PAIRS = [ 
  ["RShoulder", "RElbow"],
  ["RElbow", "RWrist"], 
]

width = 368
height = 368
inWidth = width
inHeight = height

net = cv.dnn.readNetFromTensorflow("graph_opt.pb")
thr = 0.2

points = []
pointsForCalc = []

def poseDetector(frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    pointsForCalc = []
    
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :14, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    assert(len(BODY_PARTS) == out.shape[1])

    
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thr else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 2)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            pointsForCalc.append(points[idFrom])
            pointsForCalc.append(points[idTo])

    
    t, _ = net.getPerfProfile()
    
    return frame, pointsForCalc

    """
Straight Line Function
Returns true if three points in a straight line or not
We can use this or either wen can use the angle calculation method to detect a straignt line
"""
print("Solve function is Running...!")
def solve(self, coordinates):
  (x0, y0), (x1, y1) = coordinates[0], coordinates[1]
  for i in range(2, len(coordinates)):
    x, y = coordinates[i]
    if (x0 - x1) * (y1 - y) != (x1 - x) * (y0 - y1):
      return False
    return True 

def drawCoordinatesOnImage(imgName, input, output, pointsForCalc):
  font = cv.FONT_HERSHEY_SIMPLEX
  for pointValue in pointsForCalc:
    x = pointValue[0]
    y = pointValue[1]
    cv.putText(input, str(x) + ',' +str(y), (x,y), font,0.5, (0, 255, 0), 2)

  self = pointsForCalc[2]
  coordinates = []
  coordinates.append(pointsForCalc[0])
  coordinates.append(pointsForCalc[1])

  #print(solve(self,coordinates))

  cv2_imshow(output)

def lengthSquare(X, Y):
    xDiff = X[0] - Y[0]
    yDiff = X[1] - Y[1]
    return xDiff * xDiff + yDiff * yDiff
     
def calculateAngle(A, B, C):
     
    # Square of lengths be a2, b2, c2
    a2 = lengthSquare(B, C)
    b2 = lengthSquare(A, C)
    c2 = lengthSquare(A, B)
 
    # length of sides be a, b, c
    a = math.sqrt(a2);
    b = math.sqrt(b2);
    c = math.sqrt(c2);
 
    # From Cosine law
    alpha = math.acos((b2 + c2 - a2) /
                         (2 * b * c));
    beta = math.acos((a2 + c2 - b2) /
                         (2 * a * c));
    gamma = math.acos((a2 + b2 - c2) /
                         (2 * a * b));
 
    # Converting to degree
    alpha = alpha * 180 / math.pi;
    beta = beta * 180 / math.pi;
    gamma = gamma * 180 / math.pi;
 
    # printing all the angles
    #print("alpha : %f" %(alpha))
    #print("beta : %f" %(beta))
    #print("gamma : %f" %(gamma))

    return beta

def armTest(input):
  output, pointsForCalc = poseDetector(input)
  drawCoordinatesOnImage(imgName, input, output, pointsForCalc);

  A, B, C = pointsForCalc[0], pointsForCalc[2], pointsForCalc[3]
  D, E, F = pointsForCalc[4], pointsForCalc[6], pointsForCalc[7]
  rAngle = calculateAngle(A, B, C);
  lAngle = calculateAngle(D, E, F);

  if(rAngle<180 and rAngle+5>=180):
    print("Passed the Right arm test! ARM angle: %f" %rAngle)
  if(rAngle>180 and rAngle-5<=180):
    print("Passed the Right arm test! ARM angle: %f" %rAngle)
  if(rAngle==180):
    print("Right ARM Perfectly Passed")
  else:
    print("Right Arm test failed. Arms are too much angled %f" %rAngle)


  if(lAngle<180 and lAngle+5>=180):
    print("Passed the Left arm test! ARM angle: %f" %lAngle)
  if(lAngle>180 and lAngle-5<=180):
    print("Passed the Left arm test! ARM angle: %f" %lAngle)
  if(lAngle==180):
    print("Left Arm Perfectly Passed")
  else:
    print("Left Arm test failed. Arms are too much angled %f" %lAngle)

#Main Method
imgName = 'body1.jpg'
imgPath = "media/images/"+imgName # body1.jpg body2.jpg body3.pn body4.jpg body5.jpg

input = cv.imread(imgPath)
armTest(input)