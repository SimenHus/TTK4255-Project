from common import *

# Initialize the apriltags detector
detector = DetectorClass(K=K, tagSize=apriltagParams['tagSize'], families=apriltagParams['family'])
map = Map()

@timeit
def detectTags(frames):
    for frame in frames:
        tags = detector.detect(frame)
        map.handleDetections(tags)

        drawTags(frame, K, tags)
    return frames


print('Loading frames...')
frames = loadVideo(VIDEO_PATHS[1])
# frames = loadImages(IMG_PATHS[:3])

print('Detecting tags in frames...')
frames = detectTags(frames)
FPS = 20 # FPS
manualPlayback = False # Whether to control when moving to next frame

roll = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0]
])
fixedFrameOffset = Pose(roll)
visualizationOptions = {'FPS': FPS, 'manualPlayback': manualPlayback,
                        'fixedFrameOffset': fixedFrameOffset, 'resizeImg': (300, 600)}
vis = Visualize(map, frames, **visualizationOptions)

vis.play()