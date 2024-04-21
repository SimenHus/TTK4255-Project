
from common.transformation import Pose
from collections import OrderedDict


class Map:
    landmarks = OrderedDict() # Detected landmarks and their pose seen from the fixed frame
    trajectory = [] # Store camera pose over time
    
    def handleDetections(self, tags):
        detections = OrderedDict() # id: pose
        for tag in tags:
            id, R, t = int(tag.tag_id), tag.pose_R, tag.pose_t.reshape((3,))
            if id == 48: continue
            Tsc = Pose(R, t) # Transformation matrix from detection to camera frame
            detections[id] = Tsc # Store pose of detected landmark
        
        # Exists if in detections and landmarks
        existingLandmarksDetected = list(detections.keys() & self.landmarks.keys())

        # New if in one but not the other
        # Note: This means any landmark that is in self.landmarks but not in detections will appear.
        # Needs to be handled properly
        newLandmarksDetected = list(detections.keys() ^ self.landmarks.keys())

        # Check if the fixed frame (tag 0) is detected
        fixedFrameDetected = (0 in detections.keys())

        Tct = None # Transformation matrix from camera to fixed frame
        if fixedFrameDetected: # Fixed frame detected in image
            Tct = detections[0].inv
        elif len(existingLandmarksDetected) > 0: # If no fixed frame, check for any other existing landmark
            someLandmark = existingLandmarksDetected[0]
            Tst = self.landmarks[someLandmark]
            Tcs = detections[someLandmark].inv
            Tct = Tst@Tcs
        
        if Tct is None: # No know reference detected
            self.trajectory.append(self.trajectory[-1])
            return 
        
        self.trajectory.append(Tct)
        for id in newLandmarksDetected: # Update landmark storage with new landmarks
            if id not in detections.keys(): continue # Handle new landmark as mentioned above
            Tsc = detections[id]
            Tst = Tct@Tsc
            self.landmarks[id] = Tst