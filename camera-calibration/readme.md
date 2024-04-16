# COMP3317 Computer Vison
## Camera Calibration

This program performs camera calibration using Python. The main features implemented include:

### Estimate plane-to-plane projectivities
- Solved using linear least squares optimization.

### Generate 3D coordinates correspondence
- Use the estimated projectivities to generate 3D coordinates correspondence.

### QR Decomposition
- Decompose the camera projection matrix into product of a 3×3 camera calibration matrix K and a 3×4 matrix [R T] composed of the rigid body motion of the camera.

### Estimate an Essential Matrix
- Solve the relative rotation and translation between two views and estimate the essential matrix.
