**Vehicle Detection Project**

[//]: # (Image References)
[image1]: ./results/overview_training_images.png
[image2]: ./results/overview_features.png
[image3]: ./results/sliding_windows.jpg
[image4]: ./results/overview_still_images.jpg
[image5]: ./results/overview_still_images.png
[image6]: ./results/labels_map.png
[image7]: ./results/output_bboxes.png
[video1]: ./results/project_video.mp4

---

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The implementation can be found in `Utils.py`

![alt text][image1]


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.



####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).



###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The final implantation of the sliding window search can be found in the `__detect_vehicles_frame()` method of the
`VehicleDetection` class.


The overlap was chosen to be 0.25. This means that it takes 4 steps to make a complete step.
Further the search region was limited to an area relevant for vehicles starting at height 400px and going to
656px.

This is visualized in the image below with 96x96 windows shown in green and 64x64 are annotated in blue.

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The pipeline works quite well on still images.
It detects the vehicles in the images on your lane and even is capable of detecting images on the other
traffic lanes.

Some false positives occur but their number is rather low.

![alt text][image4]

---

### Video Implementation

Training

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./results/project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

- Incompatible image data types caused problems at the beginning.
The training images where provided in png and read in floating point representation while the
test images and video frames hat a UInt8 representation. This showed that carefully looking
at the data before applying machine learning approaches is very important.
- Calculating the hog features for the whole image and not for the search window not only improved
speed but improved the results as the cut-out images showed a lot of discontinuities.
- Comparing results is complicated if you don't have a ground truth available. Optimizing the parameters can
only be done by viewing and manually checking the results. This shows the need for tons of labeled data.
- The detection of vehicles in the image using a SVC is not quite stable. It takes a lot of effort to overcome
the shortcomings like heatmap and
- In the future the results of the sequential frames should not only be low passed but predicted using
e.g. a Kalman filter to improve the results. Transforming the bounding boxes to a birds eye view
(perspective transform as in lane finding project) makes the tracking and the results independent of
image distortions.