import time
import os
import cv2
from sequence_utils import VOTSequence
# from ncc_tracker_example import NCCTracker, NCCParams
# from ms_tracker import MeanShiftTracker, MSParams
from particle_tracker import ParticleParams, ParticleTracker

# set the path to directory where you have the sequences
dataset_path = 'vot14'  # TODO: set to the dataset path on your disk
# Automatically get all sequence names (folders only, skip files like list.txt)
seq_names = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]

for seq_name in seq_names:
    print(f'Running tracker on sequence: {seq_name}')
    avg_o = []
    avg_o_full = []
    # visualization and setup parameters
    win_name = 'Tracking window'
    reinitialize = True
    show_gt = True
    video_delay = 15
    font = cv2.FONT_HERSHEY_PLAIN

    # create sequence object
    sequence = VOTSequence(dataset_path, seq_name)
    init_frame = 0
    n_failures = 0
    # create parameters and tracker objects
    # parameters = NCCParams()
    # tracker = NCCTracker(parameters)
    # parameters = MSParams()
    # tracker = MeanShiftTracker(parameters)
    parameters = ParticleParams()
    tracker = ParticleTracker(parameters)

    time_all = 0
    # initialize visualization window
    sequence.initialize_window(win_name)
    # tracking loop - goes over all frames in the video sequence
    frame_idx = 0
    while frame_idx < sequence.length():
        img = cv2.imread(sequence.frame(frame_idx))
        particles = None
        # initialize or track
        if frame_idx == init_frame:
            # initialize tracker (at the beginning of the sequence or after tracking failure)
            t_ = time.time()
            tracker.initialize(img, sequence.get_annotation(frame_idx, type='rectangle'))
            time_all += time.time() - t_
            predicted_bbox = sequence.get_annotation(frame_idx, type='rectangle')
        else:
            # track on current frame - predict bounding box
            t_ = time.time()
            track_result = tracker.track(img)
            time_all += time.time() - t_
            # First 4 elements are the bounding box [x, y, w, h]
            predicted_bbox = track_result[:4]
            # 5th element contains the particles matrix [N x 4]
            particles = track_result[4]

        # calculate overlap (needed to determine failure of a tracker)
        gt_bb = sequence.get_annotation(frame_idx, type='rectangle')
        o = sequence.overlap(predicted_bbox, gt_bb)
        avg_o.append(o)
        avg_o_full.append(o)

        # draw ground-truth and predicted bounding boxes, frame numbers and show image
        if show_gt:
            sequence.draw_region(img, gt_bb, (0, 255, 0), 1)
        # 1. Draw particles if they exist
        if particles is not None:
            for p in particles:
                px, py = int(p[0]), int(p[1])
                cv2.circle(img, (px, py), 1, (255, 0, 0), -1) # Blue dots for particles
        sequence.draw_region(img, predicted_bbox, (0, 0, 255), 2)
        sequence.draw_text(img, '%d/%d' % (frame_idx + 1, sequence.length()), (25, 25))
        sequence.draw_text(img, 'Fails: %d' % n_failures, (25, 55))
        sequence.show_image(img, video_delay)

        if o > 0 or not reinitialize:
            # increase frame counter by 1
            frame_idx += 1
        else:
            print('Average overlap until failure: %.3f' % (sum(avg_o) / len(avg_o)))
            frame_idx += 5
            init_frame = frame_idx
            n_failures += 1
            avg_o = []

    print('Tracking speed: %.1f FPS' % (sequence.length() / time_all))
    print('Tracker failed %d times' % n_failures)
    if avg_o:
        print('Average overlap: %.3f' % (sum(avg_o) / len(avg_o))) 
    if avg_o_full:
        print('Average overlap (full sequence): %.3f' % (sum(avg_o_full) / len(avg_o_full)))

    # write avg_overlap (full), tracking speed and number of failures to a text file
    results_filename = f'results_200.txt'
    with open(results_filename, 'a') as f:
        f.write(f'Sequence: {seq_name}\n')
        f.write(f'Average overlap (full sequence): {sum(avg_o_full) / len(avg_o_full):.3f}\n')
        f.write(f'Tracking speed: {sequence.length() / time_all:.1f} FPS\n')
        f.write(f'Number of failures: {n_failures}\n')





