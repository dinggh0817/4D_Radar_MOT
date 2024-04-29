"""
Run the GNN-PMB filter to perform point target tracking. Use SMURF detection results as measurements.

Config files are located in ./trackers/point_object_trackers/GNN_PMB/ folder:
    -config.json: The tracking parameters for different datasets and scenes
"""
import ujson
import json
import numpy as np
from numpyencoder import NumpyEncoder
import os
from time import perf_counter

from trackers.point_object_trackers.GNN_PMB.GNN_PMB_Tracker import GNN_PMB_Filter
from trackers.point_object_trackers.utils.utils import *
from utils import *
from datetime import datetime
import multiprocessing
import argparse
from tqdm import tqdm
import evaluation.evaluation_HOTA.trackeval as trackeval
# import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('agg')
from scipy.spatial.transform import Rotation as Rot


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="VOD")
    parser.add_argument(
        "--set_split", default="val",
        help="choose dataset split among [val] [test] [train]",
    )
    parser.add_argument("--detection", default="SMURF")

    parser.add_argument(
        "--tracker_dir",
        default=os.path.join(get_code_path(), "trackers/point_object_trackers/GNN_PMB")
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
        default=os.path.join(get_code_path(), "results/SMURF_GNN_PMB")
    )

    parser.add_argument("--parallel_process", default=4)
    parser.add_argument("--gen_eval_txt_files", default=True)

    args = parser.parse_args()
    return args


def main(process_token, cls="default", output_dir=None, if_plot=False, if_print_frame_info=True):
    args = parse_args()
    dataset_frames_info_file = args.dataset_frames_info_file
    config = os.path.join(args.tracker_dir, args.dataset + "_config.json")
    # config = os.path.join(args.tracker_dir, args.dataset + "_SMURF_config.json")

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
    filter_model = gen_filter_model(parameters, cls)

    # Iterate over different scenes
    for scene_idx, scene_token in enumerate(scenes_in_this_set):
        if scene_idx % args.parallel_process != process_token:
            continue
        # if scene_idx != 18:
        #     continue

        with open(os.path.join(args.dataset_dir, "{}_dataset_info_{}_{}_{}.json".format(args.dataset.lower(), args.set_split, scene_token, args.detection.lower())), "rb") as f:
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
        detection_bbox_in_this_scene = dataset_info["ground_truth_bboxes"]
        detection_bbox_hwlxyza_in_this_scene = dataset_info["ground_truth_bboxes_hwlxyza"]
        if args.detection != "GT":
            detection_bbox_score_in_this_scene = dataset_info["bboxes_score"]
            with open(os.path.join(args.dataset_dir, "{}_dataset_info_{}_{}.json".format(args.dataset.lower(), args.set_split, scene_token)), "rb") as f:
                dataset_info = ujson.load(f)
            ground_truth_bbox_in_this_scene = dataset_info["ground_truth_bboxes"]

        # Generate filter based on the filter model
        tracker = GNN_PMB_Filter(filter_model)

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

            detection_bbox_at_current_frame = []
            detection_bbox_hwlxyza_at_current_frame = []
            detection_bbox_score_at_current_frame = []
            if frame_token in detection_bbox_in_this_scene.keys():
                all_cls = list(detection_bbox_in_this_scene[frame_token].keys())
                if cls == "default":
                    bboxes = [np.array(detection_bbox_in_this_scene[frame_token][c]) for c in all_cls]
                    bboxes_hwlxyza = [np.array(detection_bbox_hwlxyza_in_this_scene[frame_token][c]) for c in all_cls]
                    if args.detection != "GT":
                        bboxes_score = [np.array(detection_bbox_score_in_this_scene[frame_token][c]) for c in all_cls]
                    else:
                        bboxes_score = [np.ones(x.shape[0]) for x in bboxes]
                else:
                    bboxes = [np.array(detection_bbox_in_this_scene[frame_token][c]) for c in all_cls if c==cls]
                    bboxes_hwlxyza = [np.array(detection_bbox_hwlxyza_in_this_scene[frame_token][c]) for c in all_cls if c==cls]
                    if args.detection != "GT":
                        bboxes_score = [np.array(detection_bbox_score_in_this_scene[frame_token][c]) for c in all_cls if c==cls]
                    else:
                        bboxes_score = [np.ones(x.shape[0]) for x in bboxes]
                if len(bboxes)>0:
                    detection_bbox_at_current_frame = np.vstack(bboxes)
                    detection_bbox_hwlxyza_at_current_frame = np.vstack(bboxes_hwlxyza)
                    detection_bbox_score_at_current_frame = np.hstack(bboxes_score)

            # Initialize submission file
            class_sub["results"][frame_token] = []

            sensor_state = np.array(ego_info_in_this_scene[frame_token])[0:2]
            Z_k, effective_bbox_idx = get_meas_bboxes(
                detection_bbox_hwlxyza_at_current_frame, detection_bbox_score_at_current_frame, parameters[cls]['score_thld'], parameters[cls]['nms_iou_thld']
            )
            if if_print_frame_info:
                print("Scene:{}  \tFrame:{}  \tPoints:{}".format(scene_idx, frame_token, Z_k['position'].shape[1]))

            if if_plot:
                plt.figure(figsize=(9,9))
                plt.scatter(sensor_state[0], sensor_state[1], s=10, c='r')
                for idx_gt_bbox in range(len(bbox_at_current_frame)):
                    plt.plot(
                        np.array(bbox_at_current_frame[idx_gt_bbox])[0,[0,1,2,3,0]],
                        np.array(bbox_at_current_frame[idx_gt_bbox])[1,[0,1,2,3,0]],
                        'k-', linewidth=1
                    )
                if args.detection != "GT":
                    for idx_det_bbox in range(len(detection_bbox_at_current_frame)):
                        if idx_det_bbox in effective_bbox_idx:
                            plt.plot(
                                np.array(detection_bbox_at_current_frame[idx_det_bbox])[0,[0,1,2,3,0]],
                                np.array(detection_bbox_at_current_frame[idx_det_bbox])[1,[0,1,2,3,0]],
                                'b', linewidth=3, alpha=0.5
                            )
                        else:
                            plt.plot(
                                np.array(detection_bbox_at_current_frame[idx_det_bbox])[0,[0,1,2,3,0]],
                                np.array(detection_bbox_at_current_frame[idx_det_bbox])[1,[0,1,2,3,0]],
                                'b:', linewidth=1
                            )
                        plt.text(
                            np.array(detection_bbox_at_current_frame[idx_det_bbox])[0,0],
                            np.array(detection_bbox_at_current_frame[idx_det_bbox])[1,0],
                            s="{:.2f}".format(detection_bbox_score_at_current_frame[idx_det_bbox]), c='b', fontstyle='italic', fontsize=10)
                plt.axis("equal")
        
            # Measurement preprocessing and prediction
            ts = perf_counter()
            if frame_idx == 0:
                # Initial predict step
                filter_predicted, meas_preprocessed = tracker.predict_initial_step(Z_k, ego_position=sensor_state)
            else:
                # Predict step
                filter_predicted, meas_preprocessed = tracker.predict(time_lag, filter_pruned, Z_k, ego_position=sensor_state)

            # Update step
            if frame_idx == 0:
                filter_updated = tracker.update(filter_predicted, meas_preprocessed, sensor_state)
            else:
                filter_updated = tracker.update(filter_predicted, meas_preprocessed, sensor_state, filter_pruned)

            # Pruning
            filter_pruned = tracker.prune(filter_updated)

            # State extraction
            estimates.append(tracker.extractStates(filter_pruned))
            te = perf_counter()
            tt = te - ts
            class_sub["run_time"][frame_token] = tt

            # Sort out the data structure
            bbox_corners = []
            for idx in range(len(estimates[frame_idx]["id"])):
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

                center = np.reshape(instance_info['translation'], (3,1))
                # Get the 2D bounding box in X-Y plane
                hwlza = estimates[frame_idx]["hwlza"][idx]
                rotation_angle_around_z_axis = hwlza[4]
                rotation = Rot.from_rotvec(rotation_angle_around_z_axis * np.array([0,0,1]))
                bbox_h = hwlza[0]
                bbox_w = hwlza[1]
                bbox_l = hwlza[2]
                box_corners_at_origin = np.array([
                    [bbox_l/2,  bbox_w/2,  -bbox_h/2],
                    [bbox_l/2,  -bbox_w/2, -bbox_h/2],
                    [-bbox_l/2, -bbox_w/2, -bbox_h/2],
                    [-bbox_l/2, bbox_w/2,  -bbox_h/2],
                    [bbox_l/2,  bbox_w/2,  bbox_h/2],
                    [bbox_l/2,  -bbox_w/2, bbox_h/2],
                    [-bbox_l/2, -bbox_w/2, bbox_h/2],
                    [-bbox_l/2, bbox_w/2,  bbox_h/2]
                ])
                bbox_corners = rotation.as_matrix() @ box_corners_at_origin.T + center

                rotation = Rot.from_rotvec(rotation_angle_around_z_axis * np.array([0,0,1]))
                instance_info["size"] = np.array(hwlza[0:3]).tolist()
                # instance_info["rotation"] = rotation.as_quat()[[1,2,3,0]].tolist()
                instance_info["rotation"] = rotation_angle_around_z_axis
                instance_info['vertices'] = bbox_corners[0:2,0:4].tolist()

                # Plot
                if if_plot:
                    plt.scatter(instance_info['translation'][0], instance_info['translation'][1], s=8, c='k', marker='x')
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
                    plt.text(
                        instance_info['velocity'][0] + instance_info['translation'][0],
                        instance_info['velocity'][1] + instance_info['translation'][1],
                        s="{:.2f}".format(np.sqrt(instance_info['velocity'][0]**2 + instance_info['velocity'][1]**2)), fontsize=10)

                class_sub["results"][frame_token].append(instance_info)
            if if_plot:
                plt.axis("equal")
                # plt.xlim(-50, 50)
                # plt.ylim(-50, 50)
                plt.savefig(os.path.join(figure_output_dir,"result_{}.png".format(frame_idx)), dpi=200)
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
        "PREMATCH_RESULT": False,    # If enable tracking result prematch
        "PREMATCH_THRESHOLD": 4,    # Euclidean distance threshold for result prematch
    }
    metrics_config = {"METRICS": ["HOTA", "CLEAR", "CLEAR_MT", "Identity"]}

    eval_file_dir = os.path.join(output_dir, "evaluate", "SMRUF_GNN_PMB", "data")
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

            tracking_result = {}
            tracking_time = {}
            for c in cls:
                with open(
                    os.path.join(out_file_dir, "{}_{}.json".format(scene_token, c)),
                    "r",
                ) as f:
                    result_dict = ujson.load(f)
                tracking_result[c] = result_dict["results"]
                tracking_time[c] = result_dict["run_time"]

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

                for c in cls:
                    total_run_time += tracking_time[c][sample_token]
                    result = tracking_result[c][sample_token]
                    # tracking_result_list.append(result)
                    for obj_idx in range(len(result)):
                        this_ID = result[obj_idx]["tracking_id"]
                        if not(this_ID in global_obj_IDs):
                            global_obj_IDs[this_ID] = int(obj_ID)
                            obj_ID = obj_ID + 1

                    for obj_idx in range(len(result)):
                        kitti_annotation_str.append(
                            "{time_idx} {obj_ID} {obj_class} {truncation} {occlusion} {alpha} {bb_left} {bb_top} {bb_right} {bb_bottom} {h} {w} {l} {x} {y} {z} {rot_y} {score}".format(
                                time_idx=sample_idx,
                                obj_ID = global_obj_IDs[result[obj_idx]["tracking_id"]],
                                obj_class=result[obj_idx]["tracking_name"],
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
    evaluator.evaluate(dataset_list, metrics_list)


if __name__ == "__main__":
    args = parse_args()

    if not (args.dataset in ["VOD", "TJ4D"]):
        raise KeyError("incompatible dataset")

    if not (args.set_split in ["val", "train", "test"]):
        raise KeyError("wrong data version")

    # Create a folder for this experiment
    now = datetime.now()
    formated_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = os.path.join(args.results_dir, formated_time + '_' + args.dataset + '_' + args.set_split + '_' + args.detection)
    if not (os.path.exists(output_dir)):
        os.makedirs(output_dir)

    # Read config
    with open(os.path.join(args.tracker_dir, args.dataset + "_config.json"), "rb") as f:
        config = ujson.load(f)
    # Save config to the output folder
    with open(output_dir + "/config.json", "w") as f:
        json.dump(config, f, cls=NumpyEncoder)

    classifications = ["car", "cyclist", "pedestrian"]

    # Run filter
    for classification in classifications:
        inputarguments = []
        for token in range(args.parallel_process):
            inputarguments.append(
                (token, classification, output_dir, False, False)
            )
        # Start processing information
        print("Tracking started.")
        pool = multiprocessing.Pool(processes=args.parallel_process)
        pool.starmap(main, inputarguments)
        pool.close()
        # For debugging
        # main(0, classification, output_dir, True, False)
        print(">>> {} tracking fininshed.".format(classification))

    eval(output_dir, classifications)