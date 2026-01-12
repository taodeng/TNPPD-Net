# Qualitative Video Results

This repository provides qualitative results of our model evaluated on real thyroid puncture videos, showing frame-by-frame puncture state prediction.

## Files

- pred.mp4  
  Prediction results at normal playback speed.
  https://github.com/taodeng/TNPPD-Net/blob/main/Qualitative%20video%20demo/pred.mp4

https://github.com/user-attachments/assets/2a41e6c3-76f8-4e59-81c8-74512b545163.mp4


https://github.com/user-attachments/assets/38cc2630-e0fe-4d1b-b471-746a428813ac



<video width="800" height="450" controls autoplay muted loop>
  <source src="https://github.com/taodeng/TNPPD-Net/blob/main/Qualitative%20video%20demo/pred.mp4" type="video/mp4">
</video>
<video src="https://github.com/taodeng/TNPPD-Net/blob/main/Qualitative%20video%20demo/pred.mp4" controls="controls" width="500" height="300"></video>
- pred_slow.mp4  
  Slow-motion version of pred.mp4 (1/10x speed) for easier frame-level inspection.
  
  <video src="pred_slow.mp4" controls="controls" width="500" height="300"></video>
  

## On-Screen Information

Each frame contains the following overlays:

- Frame: current frame index  
- State: puncture state predicted by the model  
- FPS: real-time inference speed, dynamically varying according to per-frame computation time

## Prediction Visualization

- Green indicates the predicted state matches the physician’s annotation.  
- Red indicates the predicted state does not match the physician’s annotation.

## Runtime Environment

- GPU: NVIDIA RTX 3090 (single GPU)  
- Inference mode: frame-by-frame, real-time processing

## Notes

- The model is trained on a static-frame dataset and applied to continuous puncture videos.  
- These videos are provided for qualitative demonstration.  
- Construction of a dedicated thyroid puncture video dataset with frame-level annotations and the development of temporal models are ongoing and planned as future work.
