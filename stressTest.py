from common import *

# Initialize the apriltags detector
detector = DetectorClass(K=K, tagSize=apriltagParams['tagSize'], families=apriltagParams['family'])
map = Map()

def detectTags(frame):
    tags = detector.detect(frame)
    map.handleDetections(tags)
    drawTags(frame, K, tags)
    return frame


@timeit
def loadImages(imgPaths, maxFrames=None):
    for i, imgPath in enumerate(imgPaths[:3]):
        frame = cv2.imread(imgPath) # Read image from file
        detectTags(frame)

        if maxFrames is not None:
            if i >= maxFrames: break


@timeit
def loadVideo(videoPath, maxFrames=None):
    cap = cv2.VideoCapture(videoPath)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    size = (600, 300)
    writer = cv2.VideoWriter(f'{OUTPUT_FOLDER}/{path.basename(videoPath)[:-4]}.avi', fourcc, 20.0, size)

    i = 1
    while cap.isOpened():
        ret, frame = cap.read()
        
        # if frame is read correctly ret is True
        if not ret: break
        handledFrame = detectTags(frame)

        handledFrame = cv2.resize(handledFrame, size)
        writer.write(handledFrame)

        i += 1
        if maxFrames is not None:
            if i >= maxFrames: break
    cap.release()
    writer.release()

print('Loading frames and detecting trajectory...')
maxFrames = None
frames = loadVideo(VIDEO_PATHS[1], maxFrames)
# frames = loadImages(IMG_PATHS[:3], maxFrames)

FPS = 20 # FPS
fixedFrameOffset = None
visualizationOptions = {'FPS': FPS, 'fixedFrameOffset': fixedFrameOffset}
vis = Visualize(map, **visualizationOptions)

vis.play()