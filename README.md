## Computer Vision Navigation Feasibility Study
#### Melanie Stich
##### Department of Mechanical and Aerospace Engineering, University of California, Davis
A computer vision-based navigation feasibility study consisting of two navigation algorithms are presented to determine whether computer vision can be used to safely navigate a small semi-autonomous inspection satellite. Utilizing stereoscopic image-sensors and computer vision, the relative attitude determination and the relative distance determination algorithms estimate the inspection satellite's relative proximity in relation to its host spacecraft. An algorithm needed to calibrate the stereo camera system is presented, and this calibration method is discussed. These relative navigation algorithms are tested in NASA Johnson Space Center's simulation software, EDGE, using a rendered model of the International Space Station to serve as the host spacecraft. The relative attitude determination algorithm (decomposeHomography.py, stereo_calibrate.py) and the relative distance determination algorithms (ocr.py, find_rectangles.py, stereo_calibrate.py), along with the necessary input images, are provided.
