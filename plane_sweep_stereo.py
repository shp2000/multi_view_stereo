import numpy as np
import cv2


EPS = 1e-8


def backproject_corners(K, width, height, depth, Rt):
    """
    Backproject 4 corner points in image plane to the imaginary depth plane using intrinsics and extrinsics
    
    Hint:
    depth * corners = K @ T @ y, where y is the output world coordinates and T is the 4x4 matrix of Rt (3x4)

    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        width -- width of the image
        heigh -- height of the image
        depth -- depth value of the imaginary plane
    Output:
        points -- 2 x 2 x 3 array of 3D coordinates of backprojected points, here 2x2 correspinds to 4 corners
    """

    points = np.array(
        (
            (0, 0, 1),
            (width, 0, 1),
            (0, height, 1),
            (width, height, 1),
        ),
        dtype=np.float32,
    ).reshape(2, 2, 3)

    """ YOUR CODE HERE
    """
   
    Rt = np.array(Rt)
    
    t = np.reshape(Rt[:,3],(3,1))
    points = points.reshape(4,3)
    dkpo =  np.array(depth * np.linalg.inv(K)@(points.T))
    points = np.array((np.linalg.inv(Rt[:,:3])@ (dkpo - t)).T)
    points = points.reshape((2,2,3))
    


    """ END YOUR CODE
    """
    return points


def project_points(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    
    Hint:
    Z * projections = K @ T @ p, where p is the input points and projections is the output, T is the 4x4 matrix of Rt (3x4)
    
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- points_height x points_width x 3 array of 3D points
    Output:
        projections -- points_height x points_width x 2 array of 2D projections
    """
    """ YOUR CODE HERE
    """
    points = np.array(points)
    row = points.shape[0]
    col = points.shape[1]
    points = (points.reshape(row*col,3)).T                                                  
                                                                   
    ones = np.ones(row*col)
    points = np.vstack((points, ones))
    points = K@Rt@points
    points = points/points[2,:] 
    points = (points[:2,:]).T
    points = points.reshape(row, col,2)

    """ END YOUR CODE
    """
    return points


def warp_neighbor_to_ref(
    backproject_fn, project_fn, depth, neighbor_rgb, K_ref, Rt_ref, K_neighbor, Rt_neighbor
):
    """
    Warp the neighbor view into the reference view
    via the homography induced by the imaginary depth plane at the specified depth

    Make use of the functions you've implemented in the same file (which are passed in as arguments):
    - backproject_corners
    - project_points

    Also make use of the cv2 functions:
    - cv2.findHomography
    - cv2.warpPerspective
    
    ! Note, when you use cv2.warpPerspective, you should use the shape (width, height), NOT (height, width)
    
    Hint: you should do the follows:
    1.) apply backproject_corners on ref view to get the virtual 3D corner points in the virtual plane
    2.) apply project_fn to project these virtual 3D corner points back to ref and neighbor views
    3.) use findHomography to get teh H between neighbor and ref
    4.) warp the neighbor view into the reference view

    Input:
        backproject_fn -- backproject_corners function
        project_fn -- project_points function
        depth -- scalar value of the depth at the imaginary depth plane
        neighbor_rgb -- height x width x 3 array of neighbor rgb image
        K_ref -- 3 x 3 camera intrinsics calibration matrix of reference view
        Rt_ref -- 3 x 4 camera extrinsics calibration matrix of reference view
        K_neighbor -- 3 x 3 camera intrinsics calibration matrix of neighbor view
        Rt_neighbor -- 3 x 4 camera extrinsics calibration matrix of neighbor view
    Output:
        warped_neighbor -- height x width x 3 array of the warped neighbor RGB image
    """

    height, width = neighbor_rgb.shape[:2]
    

    """ YOUR CODE HERE
    """
    

    vircor = backproject_fn(K_ref, width, height, depth, Rt_ref)
    
    virref1 = project_fn(K_neighbor, Rt_neighbor, vircor)
    virref2 =project_fn(K_ref, Rt_ref, vircor)
    
    vircor = vircor.reshape(-1,2)
    print(vircor.shape)
    virref1 = virref1.reshape(-1,2)
    print(virref1.shape)
    virref2 = virref2.reshape(-1,2)
   
    H, _ = cv2.findHomography(virref1, virref2)
    
    warped_neighbor = cv2.warpPerspective(neighbor_rgb, H, (width, height))


    """ END YOUR CODE
    """
    return warped_neighbor


def zncc_kernel_2D(src, dst):
    """
    Compute the cost map between src and dst patchified images via the ZNCC metric

    IMPORTANT NOTES:
    - Treat each RGB channel separately but sum the 3 different zncc scores at each pixel

    - When normalizing by the standard deviation, add the provided small epsilon value,
    EPS which is included in this file, to both sigma_src and sigma_dst to avoid divide-by-zero issues

    Input:
        src -- height x width x K**2 x 3
        dst -- height x width x K**2 x 3
    Output:
        zncc -- height x width array of zncc metric computed at each pixel
    """
    assert src.ndim == 4 and dst.ndim == 4
    assert src.shape[:] == dst.shape[:]

    """ YOUR CODE HERE
    """
    srcm = np.mean(src, axis = 2)
    dstm = np.mean(dst, axis = 2)
    srcsd = np.std(src, axis = 2)
    dstsd = np.std(dst, axis = 2)
    
    zncc= np.zeros((src.shape[0], src.shape[1], src.shape[3]))
    for i in range(src.shape[3]):
        for j in range(src.shape[0]):
            for k in range(src.shape[1]):
                zncc[j][k][i] = np.sum(np.multiply(src[j,k,:,i]-srcm[j][k][i], dst[j,k,:,i]-dstm[j][k][i]))/(np.multiply(srcsd[j][k][i], dstsd[j][k][i])+EPS)

    zncc = np.sum(zncc, axis = 2)

    """ END YOUR CODE
    """

    return zncc  # height x width


def backproject(dep_map, K):
    """
    Backproject image points to 3D coordinates wrt the camera frame according to the depth map

    Input:
        K -- camera intrinsics calibration matrix
        dep_map -- height x width array of depth values
    Output:
        points -- height x width x 3 array of 3D coordinates of backprojected points
    """
    _u, _v = np.meshgrid(np.arange(dep_map.shape[1]), np.arange(dep_map.shape[0]))

    """ YOUR CODE HERE
    """
    
    
    xyz_cam = np.zeros((dep_map.shape[0], dep_map.shape[1], 3), dtype=np.float64)
    x, y = np.meshgrid(np.arange(dep_map.shape[1]), np.arange(dep_map.shape[0]))
    xyz_cam[:,:,0] = ((x - K[0][2])*dep_map)/K[0][0]
    xyz_cam[:,:,1] = ((y - K[1][2])*dep_map)/K[1][1]
    xyz_cam[:,:,2] = dep_map

    """ END YOUR CODE
    """
    return xyz_cam
