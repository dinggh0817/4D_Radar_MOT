'''
Support functions used by PMBM filters
'''
import numpy as np
import matplotlib.pyplot as plt
import powerboxes as pb

def gen_filter_model(parameters, classification="default"):

    filter_model = {}  # filter_model is the dictionary which has all the corresponding parameters of the generated filter_model

    birth_rate = parameters[classification]['birth_rate']
    extent_model = parameters[classification]['extent_model']
    p_S = parameters[classification]['p_s']
    p_D = parameters[classification]['p_d']
    clutter_rate = parameters[classification]['clutter_rate']
    extraction_thr = parameters[classification]['extraction_thr']
    ber_thr = parameters[classification]['ber_thr']
    poi_pruning_thr = parameters[classification]['poi_pruning_thr']
    eB_thr = parameters[classification]['eB_thr']
    eta = parameters[classification]['eta']
    tau = parameters[classification]['tau']
    prior_DOF = parameters[classification]['prior_DOF']

    detected_ellipsoidal_gating_threshold = parameters[classification]['detected_ellipsoidal_gating_threshold']
    undetected_ellipsoidal_gating_threshold = parameters[classification]['undetected_ellipsoidal_gating_threshold']
    max_gating_euclidean_distance = parameters[classification]['max_gating_euclidean_distance']
    new_birth_minimum_distance = parameters[classification]["new_birth_minimum_distance"]
    min_cluster_elements = parameters[classification]['min_cluster_elements']
    max_cluster_elements = parameters[classification]['max_cluster_elements']
    max_possible_target_number = parameters[classification]['max_possible_target_number']
    max_cluster_size = parameters[classification]['max_cluster_size']

    mean_target_dimension = parameters[classification]['mean_target_dimension']
    prior_extent2 = parameters[classification]['prior_extent2']

    velocity_prior_deviation = parameters[classification]['velocity_prior_deviation']
    meas_deviation = parameters[classification]['meas_deviation']

    acceleration_deviation = parameters[classification]['acceleration_deviation']
    turn_rate_noise_deviation = np.array(parameters[classification]['turn_rate_noise_deviation'])
    rotate_angle_noise_deviation = parameters[classification]['rotate_angle_noise_deviation']

    dbscan_max_distance = parameters[classification]['dbscan_max_distance']
    dbscan_min_distance = parameters[classification]['dbscan_min_distance']
    dbscan_distance_grid = parameters[classification]['dbscan_distance_grid']
    kinematic_prior_covariance = np.diag(parameters[classification]['kinematic_prior_covariance'])

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
    # filter_model['R_k'] = meas_deviation**2 * np.eye(2) # Measurement noise covariance matrix
    
    # Measurements parameters
    filter_model['p_S'] = p_S   # Probability of target survival (prob_death = 1 - prob_survival).
    filter_model['p_D'] = p_D   # Probability of detection
    filter_model['birth_rate'] = birth_rate

    # Initial state covariance
    filter_model['velocity_prior_deviation'] = velocity_prior_deviation
    filter_model['kinematic_prior_covariance'] = kinematic_prior_covariance

    # Gamma initial parameters
    mean_measurements = parameters[classification]["MEAN_MEASUREMENTS"]
    var_measurements = parameters[classification]["VARIANCE_MEASUREMENTS"]
    # Calculate alpha, beta by method of moments
    alphaGamma_new_birth = mean_measurements**2 / var_measurements
    betaGamma_new_birth = mean_measurements / var_measurements
    filter_model['alphaGamma_new_birth'] = alphaGamma_new_birth
    filter_model['betaGamma_new_birth'] = betaGamma_new_birth
    filter_model['mean_measurements'] = mean_measurements
    filter_model['var_measurements'] = var_measurements

    # GGIW predict parameters
    filter_model['eta'] = eta
    filter_model['tau'] = tau
    filter_model['prior_DOF'] = prior_DOF

    filter_model['prior_extent2'] = prior_extent2
    if extent_model=='GGIW':
        filter_model['prior_extent1'] = np.array([
            [(mean_target_dimension[0]/2)**2 * (prior_extent2 - 3), 0],
            [0, (mean_target_dimension[1]/2)**2 * (prior_extent2 - 3)]
        ])
    else:
        raise KeyError("Unsupported extent model.")

    filter_model['position_total_covariance'] = mean_target_dimension[0]*mean_target_dimension[1] * np.eye(2)
    filter_model['mean_target_dimension'] = mean_target_dimension

    filter_model['rotate_angle_noise_deviation'] = rotate_angle_noise_deviation

    # Gating and measurement partition parameters
    filter_model['detected_ellipsoidal_gating_threshold'] = detected_ellipsoidal_gating_threshold
    filter_model['undetected_ellipsoidal_gating_threshold'] = undetected_ellipsoidal_gating_threshold
    filter_model['max_gating_euclidean_distance'] = max_gating_euclidean_distance
    filter_model['new_birth_minimum_distance'] = new_birth_minimum_distance
    filter_model['min_cluster_elements'] = min_cluster_elements
    filter_model['max_cluster_elements'] = max_cluster_elements
    filter_model['max_possible_target_number'] = max_possible_target_number
    filter_model["max_cluster_size"] = max_cluster_size

    # PMBM filter pruning 
    filter_model['maximum_number_of_global_hypotheses'] = 5     # Maximum number of hypotheses(MBM components)
    filter_model['T_pruning_MBM'] = ber_thr   # Threshold for pruning multi-Bernoulli mixtures weights.
    filter_model['T_pruning_Pois'] = poi_pruning_thr   # Threshold for pruning PHD of the Poisson component.
    filter_model['eB_threshold'] = eB_thr

    # Clustering parameters
    filter_model['dbscan_max_distance'] = dbscan_max_distance
    filter_model['dbscan_min_distance'] = dbscan_min_distance
    filter_model['dbscan_distance_grid'] = dbscan_distance_grid
 
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

def gen_measurement_matrix_for_radar(point_cloud_at_current_frame, snr_thr=-60, vel_thr=0.1, score_thr=0, dim=2, classification="car", cluster_by_bbox=False, voxelize_size=0):
    if "classification" in point_cloud_at_current_frame:
        if_in_this_class = np.array([classification in point_class for point_class in point_cloud_at_current_frame["classification"]])
    else:
        if_in_this_class = np.ones(len(point_cloud_at_current_frame["position"])).astype(bool)

    if ("velocity_r_compensated" in point_cloud_at_current_frame) and ("power" in point_cloud_at_current_frame):
        if_effective = (np.abs(np.array(point_cloud_at_current_frame["velocity_r_compensated"])) >= vel_thr) *\
            (np.array(point_cloud_at_current_frame["power"]) >= snr_thr)
    elif ("power" in point_cloud_at_current_frame):
        if_effective = (np.array(point_cloud_at_current_frame["power"]) >= snr_thr)
    elif ("RCS" in point_cloud_at_current_frame):
        if_effective = (np.array(point_cloud_at_current_frame["RCS"]) >= snr_thr)
    else:
        if_effective = np.ones(len(point_cloud_at_current_frame["position"])).astype(bool)

    if ("point_score" in point_cloud_at_current_frame):
        if_high_score = np.zeros(len(point_cloud_at_current_frame["position"])).astype(bool)
        for i, score in enumerate(point_cloud_at_current_frame["point_score"]):
            for j, cls in enumerate(point_cloud_at_current_frame["classification"][i]):
                if (cls == classification) and (score[j] >= score_thr):
                    if_high_score[i] = True
        # if_high_score = np.array([np.max(score) >= score_thr for i, score in enumerate(point_cloud_at_current_frame["point_score"])])
    else:
        if_high_score = np.ones(len(point_cloud_at_current_frame["position"])).astype(bool)

    effective_points_idx = np.where(if_in_this_class * if_effective * if_high_score)[0]

    Z_k = {}
    if len(point_cloud_at_current_frame["position"]) > 0:
        Z_k['position'] = (np.array(point_cloud_at_current_frame["position"])[effective_points_idx, :].T)[0:dim, :]

        if "velocity_r_compensated" in point_cloud_at_current_frame:
            Z_k['velocity_r_compensated'] = (np.array(point_cloud_at_current_frame["velocity_r_compensated"])[effective_points_idx]).reshape(1, -1)
        else:
            Z_k['velocity_r_compensated'] = np.zeros((1, len(effective_points_idx)))

        if cluster_by_bbox:
            if "bbox_score" in point_cloud_at_current_frame:
                if_high_score = [score >= score_thr for score in point_cloud_at_current_frame["bbox_score"]]
                Z_k['bbox_mask'] = np.array(point_cloud_at_current_frame["bbox_mask"])[if_high_score][:, effective_points_idx]
            else:
                Z_k['bbox_mask'] = np.array(point_cloud_at_current_frame["bbox_mask"])[:, effective_points_idx]

        if ("point_score" in point_cloud_at_current_frame):
            Z_k['score'] = np.array([np.max(score) for score in point_cloud_at_current_frame["point_score"]])[effective_points_idx]
        else:
            Z_k['score'] = np.zeros((len(effective_points_idx)))


        if voxelize_size != 0:
            all_points_meas = Z_k["position"]
            meas_num = all_points_meas.shape[1]
            if meas_num > 0:
                all_points_velocity = Z_k["velocity_r_compensated"]
                all_points_score = Z_k["score"]
                x_grid = np.arange(np.floor(np.min(all_points_meas[0,:])), np.ceil(np.max(all_points_meas[0,:])) + 1, voxelize_size)
                y_grid = np.arange(np.floor(np.min(all_points_meas[1,:])), np.ceil(np.max(all_points_meas[1,:])) + 1, voxelize_size)
                x_voxel_idx = np.array([np.where(x < x_grid)[0][0] for x in all_points_meas[0,:]])
                y_voxel_idx = np.array([np.where(y < y_grid)[0][0] for y in all_points_meas[1,:]])
                voxel_idx = np.hstack([x_voxel_idx.reshape(-1,1), y_voxel_idx.reshape(-1,1)])
                unique_voxel = np.unique(voxel_idx, axis=0)
                unique_voxel_num = unique_voxel.shape[0]
                voxelized_meas = np.zeros([all_points_meas.shape[0], unique_voxel_num])
                voxelized_velocity = np.zeros([1, unique_voxel_num])
                voxelized_score = np.zeros([unique_voxel_num])
                if cluster_by_bbox:
                    all_points_bbox_mask = Z_k["bbox_mask"]
                    bbox_num = all_points_bbox_mask.shape[0]
                    voxelized_bbox_mask = np.zeros([bbox_num, unique_voxel_num])
                for unique_voxel_idx in np.arange(unique_voxel_num):
                    meas_idx = np.where(((unique_voxel[unique_voxel_idx,0] * np.ones(meas_num)) == voxel_idx[:,0])
                                        * (((unique_voxel[unique_voxel_idx,1] * np.ones(meas_num)) == voxel_idx[:,1])))[0]
                    voxelized_meas[:,unique_voxel_idx] = np.mean(all_points_meas[:,meas_idx], axis=1)
                    voxelized_velocity[:,unique_voxel_idx] = np.mean(all_points_velocity[:,meas_idx], axis=1)
                    voxelized_score[unique_voxel_idx] = np.max(all_points_score[meas_idx])
                    if cluster_by_bbox:
                        voxelized_bbox_mask[:,unique_voxel_idx] = np.any(all_points_bbox_mask[:,meas_idx], axis=1)
                Z_k['position'] = voxelized_meas
                Z_k['velocity_r_compensated'] = voxelized_velocity
                Z_k['score'] = voxelized_score
                if cluster_by_bbox:
                    Z_k['bbox_mask'] = voxelized_bbox_mask
    else:
        Z_k['position'] = np.zeros([dim,0])
        Z_k['velocity_r_compensated'] = np.zeros([1,0])
        Z_k['score'] = np.zeros([0])
        if cluster_by_bbox:
            Z_k['bbox_mask'] = np.zeros([0,0])


    return Z_k

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