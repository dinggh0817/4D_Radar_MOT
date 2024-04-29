'''
Support functions used by PMBM filters
'''
import numpy as np
import matplotlib.pyplot as plt
import powerboxes as pb

def gen_filter_model(parameters, classification="default"):

    filter_model = {}  # filter_model is the dictionary which has all the corresponding parameters of the generated filter_model

    birth_rate = parameters[classification]['birth_rate']
    p_S = parameters[classification]['p_s']
    p_D = parameters[classification]['p_d']
    clutter_rate = parameters[classification]['clutter_rate']
    extraction_thr = parameters[classification]['extraction_thr']
    ber_thr = parameters[classification]['ber_thr']
    poi_pruning_thr = parameters[classification]['poi_pruning_thr']
    eB_thr = parameters[classification]['eB_thr']

    detected_ellipsoidal_gating_threshold = parameters[classification]['detected_ellipsoidal_gating_threshold']
    undetected_ellipsoidal_gating_threshold = parameters[classification]['undetected_ellipsoidal_gating_threshold']
    max_gating_euclidean_distance = parameters[classification]['max_gating_euclidean_distance']
    new_birth_minimum_distance = parameters[classification]["new_birth_minimum_distance"]

    mean_target_dimension = parameters[classification]['mean_target_dimension']

    velocity_prior_deviation = parameters[classification]['velocity_prior_deviation']
    meas_deviation = parameters[classification]['meas_deviation']

    acceleration_deviation = parameters[classification]['acceleration_deviation']
    turn_rate_noise_deviation = np.array(parameters[classification]['turn_rate_noise_deviation'])

    if parameters[classification]['model'] == 'CV':
        dim = 2
        dim_of_state = 4
    elif parameters[classification]['model'] == 'CT':
        dim = 2
        dim_of_state = 5
    else:
        raise KeyError("Motion model not defined.")
    filter_model['dim'] = dim # Space dimension
    filter_model['dim_of_state'] = dim_of_state # Space dimension
    filter_model['H_k'] = np.concatenate([np.eye(dim),np.zeros([dim,dim_of_state-dim])],axis=1, dtype=np.float64)  # Observation filter_model matrix.
    filter_model['meas_deviation'] = np.array([meas_deviation[0]/180*np.pi, meas_deviation[1]])
    filter_model['acceleration_deviation'] = acceleration_deviation
    filter_model['turn_rate_noise_deviation'] = turn_rate_noise_deviation
    
    # Measurements parameters
    filter_model['p_S'] = p_S   # Probability of target survival (prob_death = 1 - prob_survival).
    filter_model['p_D'] = p_D   # Probability of detection
    filter_model['birth_rate'] = birth_rate

    # Initial state covariance
    filter_model['velocity_prior_deviation'] = velocity_prior_deviation

    filter_model['position_total_covariance'] = mean_target_dimension[0]*mean_target_dimension[1] * np.eye(2)


    # Gating and measurement partition parameters
    filter_model['detected_ellipsoidal_gating_threshold'] = detected_ellipsoidal_gating_threshold
    filter_model['undetected_ellipsoidal_gating_threshold'] = undetected_ellipsoidal_gating_threshold
    filter_model['max_gating_euclidean_distance'] = max_gating_euclidean_distance
    filter_model['new_birth_minimum_distance'] = new_birth_minimum_distance

    # PMBM filter pruning 
    filter_model['maximum_number_of_global_hypotheses'] = 1     # Maximum number of hypotheses(MBM components)
    filter_model['T_pruning_MBM'] = ber_thr   # Threshold for pruning multi-Bernoulli mixtures weights.
    filter_model['T_pruning_Pois'] = poi_pruning_thr   # Threshold for pruning PHD of the Poisson component.
    filter_model['eB_threshold'] = eB_thr
 
    # Clutter(False alarm) parameters
    x_range = [-50, 50 ]  # X range of measurements
    y_range = [-50, 50]  # Y range of measurements
    A = (x_range[1] - x_range[0])*(y_range[1]-y_range[0])   # Size of area
    filter_model['clutter_intensity'] = clutter_rate/A  # Generate clutter/false alarm intensity (clutter intensity lambda_c = lambda_t/A)
    filter_model['xrange'] = x_range
    filter_model['yrange'] = y_range
    filter_model['uniform_weight'] = 1/A
    filter_model['clutter_rate'] = clutter_rate

    # Set the option for state extraction step
    filter_model['state_extraction_option'] = 1
    filter_model['eB_estimation_threshold'] = extraction_thr

    return filter_model

def get_meas_bboxes(bbox_hwlxyza_at_current_frame, bbox_score_at_current_frame, score_thld=0, nms_iou_thld=0.5):
    num_bbox = len(bbox_hwlxyza_at_current_frame)
    Z_k = {}
    if num_bbox > 0:
        bbox_hwlxyza_at_current_frame = np.array(bbox_hwlxyza_at_current_frame)
        bbox_score_at_current_frame = np.array(bbox_score_at_current_frame)
        # Remove low score boxes
        if_high_score = np.array([score >= score_thld for score in bbox_score_at_current_frame])
        idx_mapping = np.arange(num_bbox)[if_high_score]

        # Non-maximum supression
        effective_bbox_idx = non_maximum_supression(bbox_hwlxyza_at_current_frame[if_high_score,:], bbox_score_at_current_frame[if_high_score], nms_iou_thld)
        effective_bbox_idx = idx_mapping[effective_bbox_idx]

        Z_k["hwlxyza"] = bbox_hwlxyza_at_current_frame[effective_bbox_idx, :].T
        Z_k["score"] = bbox_score_at_current_frame[effective_bbox_idx]
    else:
        Z_k["hwlxyza"] = np.zeros([7,0])
        Z_k["score"] = np.zeros(0)
        effective_bbox_idx = []

    return Z_k, effective_bbox_idx

def non_maximum_supression(bbox_hwlxyza, bbox_score, iou_thld=0.5):
    num_bbox = len(bbox_hwlxyza)
    if num_bbox==0:
        return []

    order = sorted(range(num_bbox), key=lambda i:bbox_score[i], reverse=True)
    keep_idx = []
    bbox_cxcylwa = [np.hstack([bbox[3:5], bbox[2], bbox[1], bbox[-1]/np.pi*180]).reshape(1,-1) for bbox in bbox_hwlxyza]
    while order:
        i = order.pop(0)
        keep_idx.append(i)
        for j in order:
            # Rotated IOU metric function. Boxes are in the cxcylwa format
            bbox_iou = pb.rotated_iou_distance(bbox_cxcylwa[i], bbox_cxcylwa[j])
            if (1 - bbox_iou[0,0]) > iou_thld:
                order.remove(j)
    return keep_idx
