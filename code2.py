import numpy as np
import cv2
import glob
import yaml
import os


# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*9, 3), np.float32)
square_size = 15  # tamanho real do quadrado em mm
objp[:, :2] = np.mgrid[0:9, 0:9].T.reshape(-1, 2) * square_size

# Arrays to store object points and image points from all the images.
objpoints = []  # 3D point in real world space
imgpoints = []  # 2D points in image plane.

# Load images from specified directory
images = glob.glob(r'/home/jardel/Desktop/camera_calibration/gopro9/*.JPG')

# Create a results directory to save corrected images
results_dir = '/home/jardel/Desktop/camera_calibration/results-gopro'
os.makedirs(results_dir, exist_ok=True)

found = 0
for fname in images:
    img = cv2.imread(fname)  # Capture frame-by-frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 9), None)
    
    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)  # Every loop objp is the same, in 3D.
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9, 9), corners2, ret)
        found += 1
        
        # Resize the image for visualization
        img_resized = cv2.resize(img, (1080, 720))  # Ajuste as dimensões conforme necessário
        cv2.imshow('img', img_resized)
        cv2.waitKey(500)

# Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Transform the matrix and distortion coefficients to writable lists
data = {
    'camera_matrix': np.asarray(mtx).tolist(),
    'dist_coeff': np.asarray(dist).tolist()
}

# Save calibration data to a file
with open("calibration_matrix-gopro.yaml", "w") as f:
    yaml.dump(data, f)

# Apply undistortion to the images and display original and corrected images
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    
    # Undistort the image using the calibration parameters
    h, w = img.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, new_camera_mtx)
    
    # Resize images for side-by-side visualization
    img_resized = cv2.resize(img, (640, 480))  # Resize original
    dst_resized = cv2.resize(dst, (640, 480))  # Resize corrected
    
    # Concatenate the original and corrected images horizontally
    combined = np.hstack((img_resized, dst_resized))
    
    # Display the combined image
    cv2.imshow('Original vs Corrected', combined)
    cv2.waitKey(500)
    
    # Save the corrected image
    corrected_image_name = os.path.join(results_dir, f'corrected_{idx}.png')
    cv2.imwrite(corrected_image_name, dst)

print("Number of images used for calibration: ", found)

# When everything done, release the capture
cv2.destroyAllWindows()
