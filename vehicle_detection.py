import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split

### Parameter for training
color_space = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

### Parameter for detecting vehicles
ystart = 400
ystop = 656
scale = 1.5
cells_per_step = 2

### Detection results of last n frames
history_bbox_list = []
n_history = 2

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def train():
    # Read in dataset
    cars = glob.glob('training_data/vehicles/GTI_*/*.png') + glob.glob('training_data/vehicles/KITTI_extracted/*.png')
    notcars = glob.glob('training_data/non-vehicles/Extras/*.png') + glob.glob('training_data/non-vehicles/GTI/*.png')

    car_features = extract_features(cars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    return svc, X_scaler

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_candidate_cars(img, color_space, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, cells_per_step, spatial_size, hist_bins):
    
    img_tosearch = img[ystart:ystop,:,:]
    
    # apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    else: ctrans_tosearch = np.copy(img_tosearch)
    
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    #cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    # Create an empty list to receive positive detection windows
    hot_windows = []

    # Iterate over all windows
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            # If positive (prediction == 1) then save the window
            if test_prediction == 1:
                # Append window position to list
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                hot_windows.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart))) 
                
    return hot_windows

# Transform the detected boxes into heatmaps and apply thresholding to get the final detection
def detect_cars(image, candidate_cars_boxes):

    def add_heat(heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap# Iterate through list of bboxes
        
    def apply_threshold(heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

    def get_labeled_bboxes(labels):
        # Initialize a list to append bounding boxes to
        bbox_list = []
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Append the box to the box list
            bbox_list.append(bbox)
        # Return the bounding box list
        return bbox_list

    # Add heat to each box in box list
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    heat = add_heat(heat, candidate_cars_boxes)
        
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 2)

    # Visualize the heatmap when displaying    
    heat = np.clip(heat*10, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heat, cv2.COLORMAP_HOT)

    # Find final boxes from heatmap using label function
    from scipy.ndimage.measurements import label
    labels = label(heatmap)
    bbox_list = get_labeled_bboxes(labels)

    return bbox_list, heatmap

def rect_overlap(rect1, rect2):
    '''Overlapping rectangles overlap both horizontally & vertically
    '''
    
    def range_overlap(a_min, a_max, b_min, b_max):
        '''Neither range is completely greater than the other
        '''
        return (a_min <= b_max) and (b_min <= a_max)

    return range_overlap(rect1[0][0], rect1[1][0], rect2[0][0], rect2[1][0]) and range_overlap(rect1[0][1], rect1[1][1], rect2[0][1], rect2[1][1])

def history_check(history_bbox_list, n_history, final_boxes):
    checked_final_boxes = []
    for final_box in final_boxes:
        for bbox_list in history_bbox_list:
            for bbox in bbox_list:
                # found an overlap in a history frame
                if rect_overlap(final_box, bbox):
                    break
            else:
                # didn't find any overlap in a history frame
                # exit from the loop, causing the outer else to not be called
                break
        else:
            # find overlap in all history frame
            checked_final_boxes.append(final_box)
    
    return checked_final_boxes

def save_image(image, filename, suffix):
    filename = filename.replace('\\', '/')
    splitted_folder_file = filename.split('/')
    splitted_file_ext = splitted_folder_file[1].split('.')
    write_name = splitted_folder_file[0] + '/' + splitted_file_ext[0] + '_' + suffix + '.' + splitted_file_ext[1]
    cv2.imwrite(write_name, image)

if __name__ == '__main__':

    # # Do training and produce the model
    # clf, X_scaler = train()

    # # Save the model
    # with open('model.pickle', 'wb') as pickle_file:
    #     pickle.dump((clf, X_scaler), pickle_file)

    # Load the model
    with open('model.pickle', 'rb') as pickle_file:
        clf, X_scaler = pickle.load(pickle_file)
    
    # # Apply to test image
    # tests = glob.glob('test_images/test?.jpg')
    # for test_filename in tests:
    #     # Read in the image
    #     image = cv2.cvtColor(cv2.imread(test_filename), cv2.COLOR_BGR2RGB)
    #     # Apply svm to get boxes with positive prediction
    #     candidate_boxes = []
    #     candidate_boxes.extend(find_candidate_cars(image, color_space, ystart, ystop, 0.75, clf, X_scaler, orient, pix_per_cell, cell_per_block, 2, spatial_size, hist_bins))
    #     candidate_boxes.extend(find_candidate_cars(image, color_space, ystart, ystop, 1., clf, X_scaler, orient, pix_per_cell, cell_per_block, 2, spatial_size, hist_bins))
    #     candidate_boxes.extend(find_candidate_cars(image, color_space, ystart, ystop, 1.5, clf, X_scaler, orient, pix_per_cell, cell_per_block, 2, spatial_size, hist_bins))
    #     candidate_boxes.extend(find_candidate_cars(image, color_space, ystart, ystop, 2., clf, X_scaler, orient, pix_per_cell, cell_per_block, 2, spatial_size, hist_bins))
    #     # Draw the boxes and save image
    #     candidate_image = draw_boxes(np.copy(image), candidate_boxes)
    #     save_image(cv2.cvtColor(candidate_image, cv2.COLOR_RGB2BGR), test_filename, 'candidate_boxes')
    #     # Find the final detection boxes using heatmap technique
    #     final_boxes, heatmap = detect_cars(image, candidate_boxes)
    #     # Draw the boxes and save the image
    #     final_image = draw_boxes(np.copy(image), final_boxes)
    #     save_image(cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR), test_filename, 'final_boxes')
    #     # Save the heatmap image
    #     save_image(heatmap, test_filename, 'heatmap')

    # Pipeline for video
    def detect_vehicles(image):
        # Apply svm to get boxes with positive prediction
        candidate_boxes = []
        candidate_boxes.extend(find_candidate_cars(image, color_space, ystart, ystop, 0.75, clf, X_scaler, orient, pix_per_cell, cell_per_block, 2, spatial_size, hist_bins))
        candidate_boxes.extend(find_candidate_cars(image, color_space, ystart, ystop, 1., clf, X_scaler, orient, pix_per_cell, cell_per_block, 2, spatial_size, hist_bins))
        candidate_boxes.extend(find_candidate_cars(image, color_space, ystart, ystop, 1.5, clf, X_scaler, orient, pix_per_cell, cell_per_block, 2, spatial_size, hist_bins))
        candidate_boxes.extend(find_candidate_cars(image, color_space, ystart, ystop, 2., clf, X_scaler, orient, pix_per_cell, cell_per_block, 2, spatial_size, hist_bins))
        # Find the final detection boxes using heatmap technique
        final_boxes, heatmap = detect_cars(image, candidate_boxes)
        # Check with history, will accept bounding box if it overlaps with last n detections
        checked_final_boxes = history_check(history_bbox_list, n_history, final_boxes)
        if len(history_bbox_list) == 2:
            history_bbox_list.pop(0)
        history_bbox_list.append(final_boxes)
        # Draw the boxes and save the image
        final_image = draw_boxes(np.copy(image), checked_final_boxes)
        return final_image

    # Apply to test videos
    from moviepy.editor import VideoFileClip

    video = VideoFileClip("test_video.mp4")
    detected_vehicles_video = video.fl_image(detect_vehicles)
    detected_vehicles_video.write_videofile("test_video_detected.mp4", audio=False)

    # video = VideoFileClip("project_video.mp4")
    # detected_vehicles_video = video.fl_image(detect_vehicles)
    # detected_vehicles_video.write_videofile("project_video_detected.mp4", audio=False)
