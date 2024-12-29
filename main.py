import cv2
import depthai as dai
from calc import HostSpatialsCalc
from utility import recMask, rescale
import numpy as np
import time
from ultralytics import YOLO

conf_threshold = 0.75
model_path = "best.pt"
detect = True
count = 0


# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
camRgb = pipeline.create(dai.node.ColorCamera)


# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
# monoLeft.setFps(30)
# monoRight.setFps(30)

stereo.initialConfig.setConfidenceThreshold(255)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(False)

camRgb.setPreviewSize(600, 600)
camRgb.setInterleaved(False)
# camRgb.setFps(30)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")
stereo.depth.link(xoutDepth.input)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("disp")
stereo.disparity.link(xoutDepth.input)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
camRgb.preview.link(xoutRgb.input)

# file = open(r"data.csv", "a")
# file.write("-----------------------------------------------------------------------------------------------------------------------------------------------------------------\n")
# file.write("time, x, y, z\n")

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    depthQueue = device.getOutputQueue(name="depth")
    dispQ = device.getOutputQueue(name="disp")
    rgbQueue = device.getOutputQueue("rgb", 4, False)

    hostSpatials = HostSpatialsCalc(device)
    x = 200
    y = 300
    step = 3
    delta = 5
    hostSpatials.setDeltaRoi(delta)

    model = YOLO(model_path)

    prev_time = time.time()
    while True:
        depthData = depthQueue.get()
        rgbData = rgbQueue.get()
        if depthData is None or rgbData is None:
            continue

        rgbFrame = rgbData.getCvFrame()
        # rgbFrame = cv2.GaussianBlur(rgbFrame, (7,7), 0)
        if detect:
            results = model.predict(rgbFrame)
            for r in results:
                cv2.imshow("Balls", rgbFrame)
                if r.boxes is None:
                    continue

                for box in r.boxes:
                    if float(box.conf) < conf_threshold:
                        continue

                    bx = box.cpu().xywh.numpy()
                    imgH, imgW = r.orig_shape

                    if bx.shape[0] < 0:
                        continue
                    bbox = bx[0]
                    x, y, w, h = bbox
                    cv2.rectangle(rgbFrame, (int(x-w/2), int(y-h/2)),
                                  (int(x+w/2), int(y+h/2)), (0, 255, 0), 2)
                    detect = False

                    # Initialize Tracker
                    tracker = cv2.TrackerCSRT_create()
                    bbox_tracker = (
                        int(x - w / 2), int(y - h / 2), int(w), int(h))
                    tracker.init(rgbFrame, bbox_tracker)
                    break
        else:
            cv2.imshow("Balls", rgbFrame)
            success, bbox = tracker.update(rgbFrame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(rgbFrame, (x, y),
                              (x + w, y + h), (0, 255, 0), 2)
                count += 1
                if count > 100:
                    detect = True
                    count = 0
        # Calculate spatial coordiantes from depth frame
        spatials, centroid = hostSpatials.calc_spatials(
            depthData, (int(x), int(y)))
        # file.write(str(time.time() - prev_time))
        # prev_time = time.time()
        # file.write(', ')
        # file.write(str(spatials['x']))
        # file.write(', ')
        # file.write(str(spatials['y']))
        # file.write(', ')
        # file.write(str(spatials['z']))
        # file.write('\n')

        # Get disparity frame for nicer depth visualization
        disp = dispQ.get().getFrame()
        disp = (disp * (255 / stereo.initialConfig.getMaxDisparity())
                ).astype(np.uint8)
        disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

        # Show the frame
        cv2.imshow("depth", disp)
        cv2.waitKey(1)
