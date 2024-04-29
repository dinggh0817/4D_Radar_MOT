
from cv2 import threshold
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_metric import _BaseMetric
from .. import _timing
from .. import utils

class CLEAR_MT(_BaseMetric):
    """Class which implements the CLEAR metrics under multiple thresholds"""

    @staticmethod
    def get_default_config():
        """Default class config values"""
        default_config = {
            'THRESHOLD_LOW': 0.1,  # Similarity score threshold required for a TP match.
            'THRESHOLD_HIGH': 0.8,  # Similarity score threshold required for a TP match.
            'PRINT_CONFIG': True,  # Whether to print the config information on init. Default: False.
        }
        return default_config

    def __init__(self, config=None):
        super().__init__()
        main_integer_array_fields = ['CLR_TP', 'CLR_FN', 'CLR_FP', 'IDSW', 'MT', 'PT', 'ML', 'Frag']
        extra_integer_array_fields = ['CLR_Frames']
        self.integer_array_fields = main_integer_array_fields + extra_integer_array_fields
        main_float_array_fields = ['MOTA', 'MOTP', 'MODA', 'CLR_Re', 'CLR_Pr', 'MTR', 'PTR', 'MLR', 'sMOTA']
        extra_float_array_fields = ['CLR_F1', 'FP_per_frame', 'MOTAL', 'MOTP_sum']
        self.float_array_fields = main_float_array_fields + extra_float_array_fields
        self.fields = self.float_array_fields + self.integer_array_fields
        self.summed_fields = self.integer_array_fields + ['MOTP_sum']
        self.summary_fields = main_float_array_fields + main_integer_array_fields

        # Configuration options:
        self.config = utils.init_config(config, self.get_default_config(), self.get_name())
        self.array_labels= np.linspace(float(self.config['THRESHOLD_LOW']), float(self.config['THRESHOLD_HIGH']), 5)

    @_timing.time
    def eval_sequence(self, data):
        """Calculates CLEAR metrics for one sequence"""
        # Initialise results
        res = {}
        for field in self.fields:
            res[field] = np.zeros((len(self.array_labels)), dtype=float)

        # Return result quickly if tracker or gt sequence is empty
        if data['num_tracker_dets'] == 0:
            res['CLR_FN'] = data['num_gt_dets'] * np.ones((len(self.array_labels)), dtype=float)
            res['ML'] = data['num_gt_ids'] * np.ones((len(self.array_labels)), dtype=float)
            res['MLR'] = np.ones((len(self.array_labels)), dtype=float)
            return res
        if data['num_gt_dets'] == 0:
            res['CLR_FP'] = data['num_tracker_dets'] * np.ones((len(self.array_labels)), dtype=float)
            res['MLR'] = np.ones((len(self.array_labels)), dtype=float)
            return res

        for idx, threshold in enumerate(self.array_labels):
            # Variables counting global association
            num_gt_ids = data['num_gt_ids']
            gt_id_count = np.zeros(num_gt_ids)  # For MT/ML/PT
            gt_matched_count = np.zeros(num_gt_ids)  # For MT/ML/PT
            gt_frag_count = np.zeros(num_gt_ids)  # For Frag

            # Note that IDSWs are counted based on the last time each gt_id was present (any number of frames previously),
            # but are only used in matching to continue current tracks based on the gt_id in the single previous timestep.
            prev_tracker_id = np.nan * np.zeros(num_gt_ids)  # For scoring IDSW
            prev_timestep_tracker_id = np.nan * np.zeros(num_gt_ids)  # For matching IDSW

            # Calculate scores for each timestep
            idsw_window = np.zeros(5)
            for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
                # Deal with the case that there are no gt_det/tracker_det in a timestep.
                if len(gt_ids_t) == 0:
                    res['CLR_FP'][idx] += len(tracker_ids_t)
                    continue
                if len(tracker_ids_t) == 0:
                    res['CLR_FN'][idx] += len(gt_ids_t)
                    gt_id_count[gt_ids_t] += 1
                    continue

                # Calc score matrix to first minimise IDSWs from previous frame, and then maximise MOTP secondarily
                similarity = data['similarity_scores'][t]
                score_mat = (tracker_ids_t[np.newaxis, :] == prev_timestep_tracker_id[gt_ids_t[:, np.newaxis]])
                score_mat = 1000 * score_mat + similarity
                score_mat[similarity < threshold - np.finfo('float').eps] = 0

                # Hungarian algorithm to find best matches
                match_rows, match_cols = linear_sum_assignment(-score_mat)
                actually_matched_mask = score_mat[match_rows, match_cols] > 0 + np.finfo('float').eps
                match_rows = match_rows[actually_matched_mask]
                match_cols = match_cols[actually_matched_mask]

                matched_gt_ids = gt_ids_t[match_rows]
                matched_tracker_ids = tracker_ids_t[match_cols]

                # Calc IDSW for MOTA
                prev_matched_tracker_ids = prev_tracker_id[matched_gt_ids]
                is_idsw = (np.logical_not(np.isnan(prev_matched_tracker_ids))) & (
                    np.not_equal(matched_tracker_ids, prev_matched_tracker_ids))

                # idsw_window[0:-1] = idsw_window[1:]
                # idsw_window[-1] = np.sum(is_idsw)
                # if np.sum(idsw_window) > 3:
                #     print("time: " + str(t) + " IDS: " + str(np.sum(idsw_window)))

                res['IDSW'][idx] += np.sum(is_idsw)

                # Update counters for MT/ML/PT/Frag and record for IDSW/Frag for next timestep
                gt_id_count[gt_ids_t] += 1
                gt_matched_count[matched_gt_ids] += 1
                not_previously_tracked = np.isnan(prev_timestep_tracker_id)
                prev_tracker_id[matched_gt_ids] = matched_tracker_ids
                prev_timestep_tracker_id[:] = np.nan
                prev_timestep_tracker_id[matched_gt_ids] = matched_tracker_ids
                currently_tracked = np.logical_not(np.isnan(prev_timestep_tracker_id))
                gt_frag_count += np.logical_and(not_previously_tracked, currently_tracked)

                # Calculate and accumulate basic statistics
                num_matches = len(matched_gt_ids)
                res['CLR_TP'][idx] += num_matches
                res['CLR_FN'][idx] += len(gt_ids_t) - num_matches
                res['CLR_FP'][idx] += len(tracker_ids_t) - num_matches
                if num_matches > 0:
                    res['MOTP_sum'][idx] += sum(similarity[match_rows, match_cols])

            # Calculate MT/ML/PT/Frag/MOTP
            tracked_ratio = gt_matched_count[gt_id_count > 0] / gt_id_count[gt_id_count > 0]
            res['MT'][idx] = np.sum(np.greater(tracked_ratio, 0.8))
            res['PT'][idx] = np.sum(np.greater_equal(tracked_ratio, 0.2)) - res['MT'][idx]
            res['ML'][idx] = num_gt_ids - res['MT'][idx] - res['PT'][idx]
            res['Frag'][idx] = np.sum(np.subtract(gt_frag_count[gt_frag_count > 0], 1))
            res['MOTP'][idx] = res['MOTP_sum'][idx] / np.maximum(1.0, res['CLR_TP'][idx])

            res['CLR_Frames'][idx] = data['num_timesteps']

        # Calculate final CLEAR scores
        res = self._compute_final_fields(res)
        return res

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {}
        for field in self.summed_fields:
            res[field] = self._combine_sum(all_res, field)
        res = self._compute_final_fields(res)
        return res

    def combine_classes_det_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the detection values"""
        res = {}
        for field in self.summed_fields:
            res[field] = self._combine_sum(all_res, field)
        res = self._compute_final_fields(res)
        return res

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=False):
        """Combines metrics across all classes by averaging over the class values.
        If 'ignore_empty_classes' is True, then it only sums over classes with at least one gt or predicted detection.
        """
        res = {}
        for field in self.integer_fields:
            if ignore_empty_classes:
                res[field] = self._combine_sum(
                    {k: v for k, v in all_res.items() if v['CLR_TP'] + v['CLR_FN'] + v['CLR_FP'] > 0}, field)
            else:
                res[field] = self._combine_sum({k: v for k, v in all_res.items()}, field)
        for field in self.float_fields:
            if ignore_empty_classes:
                res[field] = np.mean(
                    [v[field] for v in all_res.values() if v['CLR_TP'] + v['CLR_FN'] + v['CLR_FP'] > 0], axis=0)
            else:
                res[field] = np.mean([v[field] for v in all_res.values()], axis=0)
        return res

    @staticmethod
    def _compute_final_fields(res):
        """Calculate sub-metric ('field') values which only depend on other sub-metric values.
        This function is used both for both per-sequence calculation, and in combining values across sequences.
        """
        num_gt_ids = res['MT'] + res['ML'] + res['PT']
        res['MTR'] = res['MT'] / np.maximum(1.0, num_gt_ids)
        res['MLR'] = res['ML'] / np.maximum(1.0, num_gt_ids)
        res['PTR'] = res['PT'] / np.maximum(1.0, num_gt_ids)
        res['CLR_Re'] = res['CLR_TP'] / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
        res['CLR_Pr'] = res['CLR_TP'] / np.maximum(1.0, res['CLR_TP'] + res['CLR_FP'])
        res['MODA'] = (res['CLR_TP'] - res['CLR_FP']) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
        res['MOTA'] = (res['CLR_TP'] - res['CLR_FP'] - res['IDSW']) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
        res['MOTP'] = res['MOTP_sum'] / np.maximum(1.0, res['CLR_TP'])
        res['sMOTA'] = (res['MOTP_sum'] - res['CLR_FP'] - res['IDSW']) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])

        res['CLR_F1'] = res['CLR_TP'] / np.maximum(1.0, res['CLR_TP'] + 0.5*res['CLR_FN'] + 0.5*res['CLR_FP'])
        res['FP_per_frame'] = res['CLR_FP'] / np.maximum(1.0, res['CLR_Frames'])
        safe_log_idsw = np.log10(res['IDSW']) if np.all(res['IDSW'] > 0) else res['IDSW']
        res['MOTAL'] = (res['CLR_TP'] - res['CLR_FP'] - safe_log_idsw) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
        return res

    def plot_single_tracker_results(self, table_res, tracker, cls, output_folder):
        """Create plot of results"""

        # Only loaded when run to reduce minimum requirements
        from matplotlib import pyplot as plt

        res = table_res['COMBINED_SEQ']
        styles_to_plot = ['r', 'b', 'g', 'b--', 'b:', 'g--', 'g:', 'm']
        for name, style in zip(self.float_array_fields, styles_to_plot):
            plt.plot(self.array_labels, res[name], style)
        plt.xlabel('threshold')
        plt.ylabel('score')
        plt.title(tracker + ' - ' + cls)
        plt.axis([self.config['THRESHOLD_LOW'], self.config['THRESHOLD_HIGH'], 0, 1])
        legend = []
        for name in self.float_array_fields:
            legend += [name + ' (' + str(np.round(np.mean(res[name]), 2)) + ')']
        plt.legend(legend, loc='upper right')
        out_file = os.path.join(output_folder, cls + '_plot_clear.pdf')
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        plt.savefig(out_file)
        plt.savefig(out_file.replace('.pdf', '.png'))
        plt.clf()
