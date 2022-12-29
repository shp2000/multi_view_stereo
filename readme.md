# Implementation two-view stereo and multi-view stereo algorithms

Implementing `two_view_stereo`:

- Rectify the views.
- Using ssd, sad and zncc kernels, compute the disparity map.
- L-R consistency check.
- Reconstruct point cloud using disparity map and  multi-pair aggregation.

Implementing `plane_sweep_stereo`:

- Sweep a series of imaginary depth planes (fronto-parallel with the reference view) across a range of candidate depths and project neighboring views onto the imaginary depth plane and back onto the reference view via a computed collineation.
- Compute cost map
- Construct the cost volume by repeating the aforementioned steps across each of the depth planes to be swept over and stacking each resulting cost map along the depth axis.
- To extract a depth map from the cost volume, choose a depth candidate by taking the argmax of the costs across the depths at each pixel.



