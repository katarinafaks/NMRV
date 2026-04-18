
import time
import cv2
import matplotlib.pyplot as plt

from sequence_utils import VOTSequence
from corr_tracker import CorrelationParams,CorrelationTracker


# set the path to directory where you have the sequences
dataset_path = 'workspace-dir/sequences' # TODO: set to the dataset path on your disk
sequence = 'gymnastics'  # choose the sequence you want to test
seq_name = sequence



# visualization and setup parameters
win_name = 'Tracking window'
reinitialize = True
show_gt = True
video_delay = 15
font = cv2.FONT_HERSHEY_PLAIN
# create sequence object
sequence = VOTSequence(dataset_path, sequence)

import numpy as np
alphas = np.linspace(0.01, 0.9,15)
# sigmas = np.linspace(0.5,5,30)
# sigmas = [0.5,5]
for alpha in alphas:
    avg_o = []
    avg_o_full = []
    # top_left = []

    init_frame = 0
    n_failures = 0
    # create parameters and tracker objects
    # parameters = CorrelationParams()
    tracker = CorrelationTracker()

    time_all = 0

    # initialize visualization window
    sequence.initialize_window(win_name)
    # tracking loop - goes over all frames in the video sequence
    frame_idx = 0
    while frame_idx < sequence.length():
        img = cv2.imread(sequence.frame(frame_idx))
        # initialize or track
        if frame_idx == init_frame:
            # initialize tracker (at the beginning of the sequence or after tracking failure)
            t_ = time.time()
            tracker.initialize(img, sequence.get_annotation(frame_idx, type='rectangle'))
            tracker.alpha = alpha
            
            time_all += time.time() - t_
            predicted_bbox = sequence.get_annotation(frame_idx, type='rectangle')
        else:
            # track on current frame - predict bounding box
            t_ = time.time()
            predicted_bbox = tracker.track(img)
            time_all += time.time() - t_

        # calculate overlap (needed to determine failure of a tracker)
        gt_bb = sequence.get_annotation(frame_idx, type='rectangle')
        o = sequence.overlap(predicted_bbox, gt_bb)        
        avg_o.append(o)
        avg_o_full.append(o)
        
        # top_left_diff = np.linalg.norm([abs(predicted_bbox[0] - gt_bb[0]), abs(predicted_bbox[1] - gt_bb[1])])
        # top_left.append(top_left_diff)

        # draw ground-truth and predicted bounding boxes, frame numbers and show image
        if show_gt:
            sequence.draw_region(img, gt_bb, (0, 255, 0), 1)
        sequence.draw_region(img, predicted_bbox, (0, 0, 255), 2)
        sequence.draw_text(img, '%d/%d' % (frame_idx + 1, sequence.length()), (25, 25))
        sequence.draw_text(img, 'Fails: %d' % n_failures, (25, 55))
        sequence.show_image(img, video_delay)

        if o > 0 or not reinitialize:
            # increase frame counter by 1
            frame_idx += 1
            
        else:
            # fail_img = img.copy()
            # sequence.draw_region(fail_img, gt_bb, (0, 255, 0), 2)
            # sequence.draw_region(fail_img, predicted_bbox, (0, 0, 255), 2)
            
            # filename = f"report/figures/failure_car_frame{frame_idx}.jpg"
            # cv2.imwrite(filename, fail_img)
            # print(f"Saved failure image to {filename}")
            # cv2.imshow('Failure analysis', fail_img)
            # cv2.waitKey(0)
            # print('Overlap at this frame: %.3f' % o)
            # print('Average overlap until failure: %.3f' % (sum(avg_o) / len(avg_o)))
            
            # increase frame counter by 5 and set re-initialization to the next frame
            frame_idx += 5
            init_frame = frame_idx
            n_failures += 1
            
            avg_o = []
            

    
    print('Tracking speed: %.1f FPS' % (sequence.length() / time_all))
    print('Tracker failed %d times' % n_failures)
    if avg_o:
        print('Average overlap: %.3f' % (sum(avg_o) / len(avg_o ))) 
    if avg_o_full:
        print('Average overlap (full sequence): %.3f' % (sum(avg_o_full) / len(avg_o_full))) 
    # save the average overlap and tracking speed to a text file with sequence name in the filename
    # but don't overwrite the file if it already exists, but append to it
    with open(f'tests_/{seq_name}_alpha_results.txt', 'a') as f:
        f.write(f'Alpha: {tracker.alpha}, Sigma: {tracker.sigma}, Lambda: {tracker.lambda_}\n')
        f.write(f'Tracking speed: {sequence.length() / time_all:.1f} FPS\n')
        f.write(f'Tracker failed {n_failures} times\n') 
        if avg_o_full:
            f.write(f'Average overlap (full sequence): {sum(avg_o_full) / len(avg_o_full):.3f}\n') 
        f.write('\n')
        



