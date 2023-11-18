import  { useEffect, useRef, useState } from 'react';
import * as posenet from '@tensorflow-models/pose-detection';
import Webcam from 'react-webcam';
import * as tf from '@tensorflow/tfjs';


export const PoseDetection = () => {
  const webcamRef = useRef(null);
  // eslint-disable-next-line no-undef
  const [poses, setPoses] = useState([]);

  useEffect(() => {
    const runPoseDetection = async () => {
      // Create PoseNet detector with the specified configuration
      await tf.ready();
      const detectorConfig = {
        architecture: 'ResNet50',
        outputStride: 16,
        inputResolution: { width: 257, height: 200 },
        quantBytes: 2,
      };

      const detector = await posenet.createDetector(posenet.SupportedModels.PoseNet, detectorConfig);

      // Continuously detect poses
      const detectPoses = async () => {
        if (webcamRef.current && webcamRef.current.video.readyState === 4) {
          // Get video properties
          const video = webcamRef.current.video;
          const videoWidth = 257;
          const videoHeight = 200;

          // Set video size for PoseNet
          video.width = videoWidth;
          video.height = videoHeight;

          // Estimate poses
          const pose = await detector.estimatePoses(video, {
            flipHorizontal: false,
          });

          // Log the pose data
          console.log(pose);
          setPoses(pose);
        }

        // Request the next animation frame
        requestAnimationFrame(detectPoses);
      };

      detectPoses();
    };

    runPoseDetection();
  }, []);

  return (
    <div style={{width:600}}>
      <Webcam
        ref={webcamRef}
        mirrored={false}
        style={{
          marginLeft: 'auto',
          marginRight: 'auto',
          left: 0,
          right: 0,
          textAlign: 'center',
          zIndex: 9,
          width: 640,
          height: 480,
        }}
      />
     <pre style={{ width: 600, height: 600, overflow: 'auto', wordWrap: 'break-word' }}>
       {JSON.stringify(poses, null, 2)}
     </pre>
    </div>
  );
};
