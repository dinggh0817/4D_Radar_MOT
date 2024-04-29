"""
Run the GGIW-PMBM filter to perform extended object tracking

Config files are located in ./trackers/extended_object_trackers/GGIW_PMBM/ folder:
    -config.json: The tracking parameters for different datasets and scenes
"""
import ujson
import json
import numpy as np
from numpyencoder import NumpyEncoder
import os
from time import perf_counter

from trackers.extended_object_trackers.GGIW_PMBM.GGIW_PMBM_Tracker import GGIW_PMBM_Filter
from trackers.extended_object_trackers.utils.utils import *
from utils import *
from datetime import datetime
import multiprocessing
import argparse
from tqdm import tqdm
import evaluation.evaluation_HOTA.trackeval as trackeval
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.spatial.transform import Rotation as Rot


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="VOD")
    parser.add_argument(
        "--set_split", default="val",
        help="choose dataset split among [val] [test] [train]",
    )

    parser.add_argument(
        "--tracker_dir",
        default=os.path.join(get_code_path(), "trackers/extended_object_trackers/GGIW_PMBM")
    )
    parser.add_argument("--project_dir", default=get_code_path())

    parser.add_argument(
        "--dataset_dir",
        default=os.path.join(get_code_path(), "VOD_dataset")
    )
    parser.add_argument(
        "--dataset_frames_info_file",
        default=os.path.join(get_code_path(), "VOD_dataset", "frames_info.json")
    )
    parser.add_argument("--results_dir",
        default=os.path.join(get_code_path(), "results/GGIW_PMBM")
    )

    parser.add_argument("--parallel_process", default=4)
    parser.add_argument("--gen_eval_txt_files", default=True)

    args = parser.parse_args()
    return args


def main(process_token, cls="default", output_dir=None, if_plot=False, if_print_frame_info=True):
    args = parse_args()
    dataset_frames_info_file = args.dataset_frames_info_file
    config = os.path.join(args.tracker_dir, args.dataset +  "_config.json")

    if not (args.dataset in ["VOD", "TJ4D"]):
        raise KeyError("Incompatible dataset")
    if args.set_split in ["val", "train", "test"]:
        set_info = args.set_split
    else:
        raise KeyError("Wrong data version")

    # Read dataset frames info
    with open(dataset_frames_info_file, "rb") as f:
        frames_info = ujson.load(f)
    
    scenes_in_this_set = [x["scene_token"] for x in frames_info if x["set"]==set_info]

    # Read parameters
    with open(config, "r") as f:
        parameters = ujson.load(f)

    # Generate filter model based on the classification
    filter_model = gen_filter_model(parameters)

    # Iterate over different scenes
    for scene_idx, scene_token in enumerate(scenes_in_this_set):
        if scene_idx % args.parallel_process != process_token:
            continue

        with open(os.path.join(args.dataset_dir, "{}_dataset_info_{}_{}.json".format(args.dataset.lower(), args.set_split, scene_token)), "rb") as f:
            dataset_info = ujson.load(f)

        if if_plot:
            figure_output_dir = os.path.join(output_dir, str(scene_idx), cls)
            if os.path.exists(figure_output_dir):
                pass
            else:
                os.makedirs(figure_output_dir)

        print(">>> Start tracking under scene {}({}) process {}".format(scene_idx, scene_token, process_token))
        ordered_frames_in_this_scene = dataset_info["ordered_frame_info"]
        time_stamps_in_this_scene = dataset_info["time_stamp_info"]
        ego_info_in_this_scene = dataset_info["ego_position_info"]
        point_cloud_in_this_scene = dataset_info["radar_point_cloud_info"]
        ground_truth_bbox_in_this_scene = dataset_info["ground_truth_bboxes"]

        # Generate filter based on the filter model
        tracker = GGIW_PMBM_Filter(filter_model)

        # Initialize submission file
        class_sub = initiate_classification_submission_file(cls)
        estimates = []
        target_labels = np.zeros(0).astype(int)
        # Iterate over all frames in the current scene
        for frame_idx, frame_token in enumerate(ordered_frames_in_this_scene):
            if frame_idx == 0:
                pre_timestamp = time_stamps_in_this_scene[frame_idx]
            cur_timestamp = time_stamps_in_this_scene[frame_idx]
            time_lag = (cur_timestamp - pre_timestamp) / 1e6
            pre_timestamp = cur_timestamp

            # Get measurements at global coordinate
            if frame_token in point_cloud_in_this_scene.keys():
                point_cloud_at_current_frame = point_cloud_in_this_scene[frame_token]
            else:
                print("lacking point cloud information")
                break

            # Get gt bbox for plotting
            bbox_at_current_frame = []
            if frame_token in ground_truth_bbox_in_this_scene.keys():
                all_cls = list(ground_truth_bbox_in_this_scene[frame_token].keys())
                if cls == "default":
                    bboxes = [np.array(ground_truth_bbox_in_this_scene[frame_token][c]) for c in all_cls]
                else:
                    bboxes = [np.array(ground_truth_bbox_in_this_scene[frame_token][c]) for c in all_cls if c==cls]
                if len(bboxes)>0:
                    bbox_at_current_frame = np.vstack(bboxes)

            # Initialize submission file
            class_sub["results"][frame_token] = []

            Z_k = gen_measurement_matrix_for_radar(
                point_cloud_at_current_frame, parameters[cls]['snr_thld'], parameters[cls]['vel_thld'], dim=3, voxelize_size=parameters[cls]['VOXEL_SIZE']
            )
            if if_print_frame_info:
                print("Scene:{}  \tFrame:{}  \tPoints:{}".format(scene_idx, frame_token, Z_k['position'].shape[1]))
        
            sensor_state = np.array(ego_info_in_this_scene[frame_token])[0:2]
            # Measurement preprocessing and prediction
            ts = perf_counter()
            if frame_idx == 0:
                # Initial predict step
                filter_predicted, meas_preprocessed = tracker.predict_initial_step(Z_k, cluster_by_bbox=False, ego_position=sensor_state, use_cluster_size=True)
            else:
                # Predict step
                filter_predicted, meas_preprocessed = tracker.predict(
                    time_lag, filter_pruned, Z_k, cluster_by_bbox=False, ego_position=sensor_state, use_cluster_size=True
                )
            te = perf_counter()
            tt = te - ts

            if if_plot:
                plt.figure(figsize=(9,9))
                plt.scatter(Z_k['position'][0,:], Z_k['position'][1,:], s=2, c='y')
                plt.scatter(sensor_state[0], sensor_state[1], s=10, c='r')
                for idx_cluster in range(meas_preprocessed['number_of_unique_clusters']):
                    idx_meas_in_cluster = np.where(meas_preprocessed['unique_clusters'][idx_cluster])[0]
                    plt.scatter(meas_preprocessed['used_measurements'][0,idx_meas_in_cluster],
                                meas_preprocessed['used_measurements'][1,idx_meas_in_cluster], s=20, alpha=0.3)
                for idx_gt_bbox in range(len(bbox_at_current_frame)):
                    plt.plot(
                        np.array(bbox_at_current_frame[idx_gt_bbox])[0,[0,1,2,3,0]],
                        np.array(bbox_at_current_frame[idx_gt_bbox])[1,[0,1,2,3,0]],
                        'b-', linewidth=1
                    )
                plt.axis("equal")
                plt.xlim(sensor_state[0] - 50, sensor_state[0] + 50)
                plt.ylim(sensor_state[1] - 50, sensor_state[1] + 50)

            ts = perf_counter()
            # Update step
            if frame_idx == 0:
                filter_updated = tracker.update(
                    filter_predicted, meas_preprocessed, sensor_state
                )
            else:
                filter_updated = tracker.update(
                    filter_predicted, meas_preprocessed, sensor_state, filter_pruned
                )

            # Pruning
            filter_pruned = tracker.prune(filter_updated)

            # State extraction
            estimates.append(tracker.extractStates(filter_pruned))
            te = perf_counter()
            tt = tt + te - ts
            class_sub["run_time"][frame_token] = tt

            # NMS
            num_estimates = len(estimates[frame_idx]["id"])
            estimated_bbox_hwlxyza = []
            estimated_bbox_eB = []
            for idx in range(num_estimates):
                # Get the 2D bounding box in X-Y plane
                matX = estimates[frame_idx]["extent"][idx]
                eigenvalues_in_xy_plane, eigenvectors_in_xy_plane = eig(matX)
                eigenvalues_idx_ascend = np.argsort(eigenvalues_in_xy_plane)

                height = 1
                ### GGIW
                size_in_xy_plane = np.sqrt(eigenvalues_in_xy_plane[eigenvalues_idx_ascend]) * 2
                # Calculate rotation angle of 2D bounding box in X-Y plane
                rotation_angle_around_z_axis = np.arctan2(eigenvectors_in_xy_plane[1][eigenvalues_idx_ascend[1]], eigenvectors_in_xy_plane[0][eigenvalues_idx_ascend[1]])
                bbox_w = size_in_xy_plane[0]
                bbox_l = size_in_xy_plane[1]
                bbox_h = height
                hwlxyza = np.array([
                    bbox_h,
                    bbox_w,
                    bbox_l,
                    estimates[frame_idx]["mean"][idx][0],
                    estimates[frame_idx]["mean"][idx][1],
                    0,
                    rotation_angle_around_z_axis
                ])
                estimated_bbox_hwlxyza.append(hwlxyza)
                estimated_bbox_eB.append(estimates[frame_idx]["eB"][idx])
            keep_idx = non_maximum_supression(estimated_bbox_hwlxyza, estimated_bbox_eB, iou_thld=0.1)

            # Sort out the data structure
            bbox_corners = []
            for idx in keep_idx:
                this_target_label = estimates[frame_idx]["id"][idx]
                existing_target_num = len(target_labels)
                if existing_target_num == 0:
                    target_labels = np.hstack([target_labels, this_target_label])
                    target_ID = 0
                else:
                    target_ID = np.where(target_labels == this_target_label)[0]
                    if len(target_ID) == 0:
                        target_labels = np.hstack([target_labels, this_target_label])
                        target_ID = existing_target_num
                    elif len(target_ID) == 1:
                        target_ID = target_ID[0]
                    elif len(target_ID) > 1:
                        print("F")
                target_ID = str(target_ID)

                instance_info = {}
                instance_info["sample_token"] = str(frame_idx)
                instance_info["translation"] = [
                    estimates[frame_idx]["mean"][idx][0],
                    estimates[frame_idx]["mean"][idx][1],
                    0]
                instance_info["velocity"] = [estimates[frame_idx]["mean"][idx][2], estimates[frame_idx]["mean"][idx][3]]
                instance_info["tracking_id"] = cls + "_" + str(target_ID)
                instance_info["tracking_name"] = cls
                instance_info["tracking_score"] = 1
                instance_info["cluster_width_mean"] = estimates[frame_idx]["cluster_width_mean"][idx]
                instance_info["cluster_width_var"] = estimates[frame_idx]["cluster_width_var"][idx]
                instance_info["cluster_length_mean"] = estimates[frame_idx]["cluster_length_mean"][idx]
                instance_info["cluster_length_var"] = estimates[frame_idx]["cluster_length_var"][idx]
                instance_info["cluster_element_mean"] = estimates[frame_idx]["cluster_element_mean"][idx]
                instance_info["cluster_element_var"] = estimates[frame_idx]["cluster_element_var"][idx]
                instance_info["match_ratio"] = estimates[frame_idx]["match_ratio"][idx]

                center = np.reshape(instance_info['translation'], (3,1))
                """
                    Matrix-based Extent
                """
                # Get the 2D bounding box in X-Y plane
                matX = estimates[frame_idx]["extent"][idx]
                eigenvalues_in_xy_plane, eigenvectors_in_xy_plane = eig(matX)
                eigenvalues_idx_ascend = np.argsort(eigenvalues_in_xy_plane)

                height = 1
                ### GGIW
                size_in_xy_plane = np.sqrt(eigenvalues_in_xy_plane[eigenvalues_idx_ascend]) * 2
                # Calculate rotation angle of 2D bounding box in X-Y plane
                rotation_angle_around_z_axis = np.arctan2(eigenvectors_in_xy_plane[1][eigenvalues_idx_ascend[1]], eigenvectors_in_xy_plane[0][eigenvalues_idx_ascend[1]])
                rotation = Rot.from_rotvec(rotation_angle_around_z_axis * np.array([0,0,1]))
                bbox_w = size_in_xy_plane[0]
                bbox_l = size_in_xy_plane[1]
                bbox_h = height
                padding = 0 # Expand the bounding box to contain more points
                h_padding = 0 # Expand the bounding box to contain more points
                box_corners_at_origin = np.array([[bbox_l/2 + padding, bbox_w/2 + padding, 0 - h_padding],
                                                [bbox_l/2 + padding, -bbox_w/2 - padding, 0 - h_padding],
                                                [-bbox_l/2 - padding, -bbox_w/2 - padding, 0 - h_padding],
                                                [-bbox_l/2 - padding, bbox_w/2 + padding, 0 - h_padding],
                                                [bbox_l/2 + padding, bbox_w/2 + padding, bbox_h + h_padding],
                                                [bbox_l/2 + padding, -bbox_w/2 - padding, bbox_h + h_padding],
                                                [-bbox_l/2 - padding, -bbox_w/2 - padding, bbox_h + h_padding],
                                                [-bbox_l/2 - padding, bbox_w/2 + padding, bbox_h + h_padding]
                                                ])
                bbox_corners = rotation.as_matrix() @ box_corners_at_origin.T + center

                rotation = Rot.from_rotvec(rotation_angle_around_z_axis * np.array([0,0,1]))
                instance_info["size"] = np.array([bbox_h, bbox_w, bbox_l]).tolist()
                # instance_info["rotation"] = rotation.as_quat()[[1,2,3,0]].tolist()
                instance_info["rotation"] = rotation_angle_around_z_axis


                instance_info['vertices'] = bbox_corners[0:2,0:4].tolist()

                # Plot
                if if_plot:
                    plt.text(instance_info['translation'][0]+1, instance_info['translation'][1]+1, s=str(target_ID), fontweight='bold', fontsize=10)
                    plt.plot(
                        bbox_corners[0, [0,1,2,3,0]],
                        bbox_corners[1, [0,1,2,3,0]],
                        'r--', linewidth=1.5
                        )
                    plt.plot(
                        [instance_info['translation'][0], instance_info['velocity'][0] + instance_info['translation'][0]],
                        [instance_info['translation'][1], instance_info['velocity'][1] + instance_info['translation'][1]],
                        'r-', linewidth=3, alpha=0.6
                        )

                class_sub["results"][frame_token].append(instance_info)
            if if_plot:
                plt.axis("equal")
                plt.savefig(os.path.join(figure_output_dir,"result_{}.png".format(frame_idx)), dpi=250)
                plt.close()

        # Save the result for this scene
        with open(
            output_dir
            + "/{}_{}.json".format(scene_token, cls),
            "w",
        ) as f:
            json.dump(class_sub, f, cls=NumpyEncoder)

        print("<<< Done with scene {}({}) process {}".format(scene_idx, scene_token, process_token))


def eval(out_file_dir, cls):
    args = parse_args()

    if args.dataset == "VOD":
        file_name_digit = 5
    elif args.dataset == "TJ4D":
        file_name_digit = 6
    else:
        raise KeyError("incompatible dataset")

    if args.set_split in ["val", "train", "test"]:
        set_info = args.set_split
    else:
        raise KeyError("wrong data version")

    eval_result_output_dir = out_file_dir + "/eval_result"

    eval_config = {
        "USE_PARALLEL": False,
        "NUM_PARALLEL_CORES": 8,
        "BREAK_ON_ERROR": True,  # Raises exception and exits with error
        "RETURN_ON_ERROR": False,  # if not BREAK_ON_ERROR, then returns from function on error
        "LOG_ON_ERROR": os.path.join(eval_result_output_dir, "error_log.txt"),  # if not None, save any errors into a log file.
        "PRINT_RESULTS": True,
        "PRINT_ONLY_COMBINED": False,
        "PRINT_CONFIG": True,
        "TIME_PROGRESS": False,
        "DISPLAY_LESS_PROGRESS": False,
        "OUTPUT_SUMMARY": True,
        "OUTPUT_EMPTY_CLASSES": True,  # If False, summary files are not output for classes with no detections
        "OUTPUT_DETAILED": True,
        "PLOT_CURVES": True,
    }
    dataset_config = {
        "GT_FOLDER": os.path.join(args.dataset_dir, "kitti_eval_file"),  # Location of GT data
        "TRACKERS_FOLDER": os.path.join(out_file_dir, "evaluate"),  # Trackers location
        "OUTPUT_FOLDER": None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
        "TRACKERS_TO_EVAL": None,  # Filenames of trackers to eval (if None, all in folder)
        "CLASSES_TO_EVAL": cls,  # Valid: ['car', 'pedestrian']
        "SPLIT_TO_EVAL": set_info,  # Valid: 'training', 'val', 'training_minus_val', 'test'
        "INPUT_AS_ZIP": False,  # Whether tracker input files are zipped
        "PRINT_CONFIG": False,  # Whether to print current config
        "TRACKER_SUB_FOLDER": "data",  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
        "OUTPUT_SUB_FOLDER": eval_result_output_dir,  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
        "TRACKER_DISPLAY_NAMES": None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
        "PREMATCH_RESULT": True,    # If enable tracking result prematch
        "PREMATCH_THRESHOLD": 2,    # Euclidean distance threshold for result prematch
    }
    metrics_config = {"METRICS": ["HOTA", "CLEAR", "CLEAR_MT", "Identity"]}

    eval_file_dir = os.path.join(output_dir, "evaluate", "GGIW_PMBM", "data")
    visual_file_dir = os.path.join(output_dir, "k3d_visualize_file")

    if args.gen_eval_txt_files:
        if args.dataset == "VOD":
            # Read basic info of VOD
            with open(os.path.join(args.project_dir, "VOD_dataset", "frames_info.json"), "r") as f:
                scene_frame_info = ujson.load(f)
        elif args.dataset == "TJ4D":
            # Read basic info of TJ4D
            with open(os.path.join(args.project_dir, "TJ_dataset", "frames_info.json"), "r") as f:
                scene_frame_info = ujson.load(f)
        f.close()

        if not os.path.exists(eval_file_dir):
            os.makedirs(eval_file_dir)

        # Generate .txt tracking result files used by KITTI eval kit
        print("\n>>> Generating \.txt tracking result files used by KITTI eval kit...")
        pbar = tqdm(total=len(scene_frame_info))
        width_bbox = []
        length_bbox = []
        width_cluster = []
        length_cluster = []
        element_cluster = []
        total_run_time = 0
        total_frames = 0
        for scene_idx, scene_info in enumerate(scene_frame_info):
            scene_token = scene_info["scene_token"]

            # No annotations for the test set
            curr_set = scene_info["set"]
            # if curr_set == "test" or curr_set != set_info:
            if curr_set != set_info:
                pbar.update(1)
                continue

            with open(
                os.path.join(out_file_dir, "{}_{}.json".format(scene_token, "default")),
                "r",
            ) as f:
                result_dict = ujson.load(f)
            tracking_result = result_dict["results"]
            tracking_time = result_dict["run_time"]

            with open(os.path.join(args.dataset_dir, "{}_dataset_info_{}_{}.json".format(args.dataset.lower(), args.set_split, scene_token)), "rb") as f:
                dataset_info = ujson.load(f)

            # Read radar point cloud
            point_cloud_in_this_scene = dataset_info["radar_point_cloud_info"]

            # Read tracking results
            sample_tokens = [
                str(sample).zfill(file_name_digit)
                for sample in np.arange(
                    int(scene_info["begin"]), int(scene_info["end"]) + 1
                )
            ]

            # Assign unique IDs to objects in the current scene
            global_obj_IDs = {}
            obj_ID = 0
            # Generate KITTI annotation strings and save .txt files
            kitti_annotation_str = []
            for sample_idx, sample_token in enumerate(sample_tokens):
                total_frames += 1
                total_run_time += tracking_time[sample_token]
                # Get measurements at global coordinate
                if sample_token in point_cloud_in_this_scene.keys():
                    radar_points = np.array(point_cloud_in_this_scene[sample_token]["position"])
                else:
                    print("lacking point cloud information " + sample_token)
                    radar_points = []

                result = tracking_result[sample_token]
                # tracking_result_list.append(result)
                for obj_idx in range(len(result)):
                    this_ID = result[obj_idx]["tracking_id"]
                    if not(this_ID in global_obj_IDs):
                        global_obj_IDs[this_ID] = int(obj_ID)
                        obj_ID = obj_ID + 1

                for obj_idx in range(len(result)):
                    width_bbox.append(result[obj_idx]["size"][1])
                    length_bbox.append(result[obj_idx]["size"][2])
                    width_cluster.append(result[obj_idx]["cluster_width_mean"])
                    length_cluster.append(result[obj_idx]["cluster_length_mean"])
                    element_cluster.append(result[obj_idx]["cluster_element_mean"])

                    # No classification
                    class_of_target = 'car'

                    # # VOD
                    # is_car_targets = (result[obj_idx]["cluster_width_mean"] >= 0.6) and (result[obj_idx]["cluster_length_mean"] >= 1.2) \
                    #     and (result[obj_idx]["cluster_width_mean"] < 3) and (result[obj_idx]["cluster_length_mean"] < 4.5)
                    # is_pedestrian_targets = (result[obj_idx]["cluster_width_mean"] < 0.6) and (result[obj_idx]["cluster_length_mean"] < 0.6)
                    # is_cyclist_targets = (result[obj_idx]["cluster_width_mean"] < 1) and (result[obj_idx]["cluster_length_mean"] < 1.2) 
                    #     # and (result[obj_idx]["cluster_width_mean"] > 0.3) and (result[obj_idx]["cluster_length_mean"] > 0.5)
                    # if is_pedestrian_targets:
                    #     # class_of_target = 'pedestrian'
                    #     if (result[obj_idx]["cluster_element_mean"] > 1) and (result[obj_idx]["cluster_element_mean"] < 8):
                    #         if (result[obj_idx]["size"][1] <= 1.3) and (result[obj_idx]["size"][1] > 0.3 ) \
                    #             and (result[obj_idx]["size"][2] > 0.3) and (result[obj_idx]["size"][2] <= 1.5):
                    #             class_of_target = 'pedestrian'
                    # if is_cyclist_targets and (not (class_of_target == 'pedestrian')):
                    #     # class_of_target = 'cyclist'
                    #     if (result[obj_idx]["cluster_element_mean"] > 5) and (result[obj_idx]["cluster_element_mean"] < 10):
                    #         if (result[obj_idx]["size"][1] <= 1.2) and (result[obj_idx]["size"][1] > 0.3 ) \
                    #             and (result[obj_idx]["size"][2] > 0.3) and (result[obj_idx]["size"][2] <= 2):
                    #             class_of_target = 'cyclist'
                    # if is_car_targets and (not (class_of_target == 'pedestrian' or class_of_target == 'cyclist')):
                    #     # class_of_target = 'car'
                    #     if (result[obj_idx]["cluster_element_mean"] > 6) and (result[obj_idx]["cluster_element_mean"] < 200):
                    #         if (result[obj_idx]["size"][1] >= 0.8) and (result[obj_idx]["size"][2] >= 0.8):
                    #             class_of_target = 'car'

                    kitti_annotation_str.append(
                        "{time_idx} {obj_ID} {obj_class} {truncation} {occlusion} {alpha} {bb_left} {bb_top} {bb_right} {bb_bottom} {h} {w} {l} {x} {y} {z} {rot_y} {score}".format(
                            time_idx=sample_idx,
                            obj_ID = global_obj_IDs[result[obj_idx]["tracking_id"]],
                            obj_class=class_of_target,
                            truncation=0,
                            occlusion=0,
                            alpha=0,
                            bb_left=0,
                            bb_top=0,
                            bb_right=0,
                            bb_bottom=0,
                            h=result[obj_idx]["size"][0],
                            w=result[obj_idx]["size"][1],
                            l=result[obj_idx]["size"][2],
                            x=result[obj_idx]["translation"][0],
                            y=result[obj_idx]["translation"][1],
                            z=0,
                            rot_y=result[obj_idx]["rotation"],
                            score=1,
                        )
                    )
            with open(os.path.join(eval_file_dir, str(scene_idx).zfill(4) + ".txt"), "w") as f:
                f.write("\n".join(kitti_annotation_str))
            pbar.update(1)
        pbar.close()
        print("<<< Generated \.txt files in " + os.path.join(output_dir, "evaluate"))
        print("<<< Average FPS: {:.4f} ({} frames {:.4f} seconds).".format(total_frames/total_run_time, total_frames, total_run_time))

    # Run evaluation code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.Kitti2DBox(dataset_config)]
    metrics_list = []
    CLEAR_config = {"THRESHOLD":0.2}
    CLEAR_MT_config = {"THRESHOLD_LOW":0.5, "THRESHOLD_HIGH":0.9}
    Identity_config = {"THRESHOLD":0.2}
    for metric in [
        trackeval.metrics.HOTA(),
        # trackeval.metrics.CLEAR(CLEAR_config),
        trackeval.metrics.CLEAR_MT(CLEAR_MT_config),
        trackeval.metrics.Identity(Identity_config),
    ]:
        if metric.get_name() in metrics_config["METRICS"]:
            metrics_list.append(metric)
    if len(metrics_list) == 0:
        raise Exception("No metrics selected for evaluation")
    _, _, output_data = evaluator.evaluate(dataset_list, metrics_list)

    det_bbox_for_each_cls = {c:[] for c in cls}
    for seq in output_data["Kitti2DBox"]["GGIW_PMBM"]:
        this_seq_data = output_data["Kitti2DBox"]["GGIW_PMBM"][seq]
        for c in cls:
            dets = this_seq_data[c]["tracker_dets"]
            for det in dets:
                det_bbox_for_each_cls[c].append(np.atleast_2d(det[:,1:3]))

    bboxwl_for_each_cls = {k:np.vstack(det_bbox_for_each_cls[k]) for k in det_bbox_for_each_cls}

    plt.figure(figsize=(12,3), dpi=250)
    ax = plt.subplot(1,2,1)
    plt.hist([bboxwl_for_each_cls[c][:,0] for c in cls], bins=np.linspace(0,3,13), histtype='bar', alpha=1, label=cls)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(0,3)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.xlabel("Estimated Bounding Box Width/m", fontsize=18)
    plt.legend(fontsize=16)

    ax = plt.subplot(1,2,2)
    plt.hist([bboxwl_for_each_cls[c][:,1] for c in cls], bins=np.linspace(0,6,13), histtype='bar', alpha=1, label=cls)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(0,6)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.xlabel("Estimated Bounding Box Length/m", fontsize=18)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tp_bbox_histogram.png'))
    plt.close()


if __name__ == "__main__":
    args = parse_args()

    if not (args.dataset in ["VOD", "TJ4D"]):
        raise KeyError("incompatible dataset")

    if not (args.set_split in ["val", "train", "test"]):
        raise KeyError("wrong data version")

    # Create a folder for this experiment
    now = datetime.now()
    formated_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = os.path.join(args.results_dir, formated_time + '_' + args.dataset + '_' + args.set_split)
    if not (os.path.exists(output_dir)):
        os.makedirs(output_dir)

    # Read config
    with open(os.path.join(args.tracker_dir, args.dataset + "_config.json"), "rb") as f:
        config = ujson.load(f)
    # Save config to the output folder
    with open(output_dir + "/config.json", "w") as f:
        json.dump(config, f, cls=NumpyEncoder)

    # Run filter
    inputarguments = []
    for token in range(args.parallel_process):
        inputarguments.append(
            (token, "default", output_dir, False, False)
        )
    # Start processing information
    print("Tracking started.")
    pool = multiprocessing.Pool(processes=args.parallel_process)
    pool.starmap(main, inputarguments)
    pool.close()
    # For debugging
    # main(0, "default", output_dir, True, True)
    print("Tracking fininshed.")

    eval(output_dir, ["car", "pedestrian", "cyclist"])

