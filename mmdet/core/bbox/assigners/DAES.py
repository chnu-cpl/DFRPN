
## 重新修改版本，在原版本中有一个center fliter需要计算anchor的 xy和中心值，这个值计算量比较大，耗时还没有用 因此把center_filter， anchors_xys删掉
import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class DAES(BaseAssigner):
    """For each gt box, assign k pos samples to it. The remaining samples are assigned with a neg label.

    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    Args:
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
        iou_calculator (str): The class of calculating bbox similarity, including BboxOverlaps2D and BboxDistanceMetric
        assign_metric (str): The metric of measuring the similarity between boxes.
        topk (int): assign k positive samples to each gt.
    """

    def __init__(self,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 gpu_assign_thr=-1,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 assign_metric='iou',
                 fun='none',
                 scores = True,
                 candidate_topk = 9,
                 center_radius = 2.5):
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.assign_metric = assign_metric
        self.fun = fun
        self.scores = scores
        self.candidate_topk = candidate_topk
        self.center_radius = center_radius
    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, or a semi-positive number. -1 means negative
        sample, semi-positive number is the index (0-based) of assigned gt.
        The assignment is done in following steps, the order matters.

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.

        Example:
            >>> self = RankingAssigner(0.5, 0.5)
            >>> bboxes = torch.Tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
            >>> gt_bboxes = torch.Tensor([[0, 0, 10, 9]])
            >>> assign_result = self.assign(bboxes, gt_bboxes)
            >>> expected_gt_inds = torch.LongTensor([1, 0])
            >>> assert torch.all(assign_result.gt_inds == expected_gt_inds)
        """
        # import pdb
        # pdb.set_trace()
        # assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
        #     gt_bboxes.shape[0] > self.gpu_assign_thr) else False
        assign_on_cpu = False
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = bboxes.device
            bboxes = bboxes.cpu()
            
            pred_bboxes = pred_bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()


        # 计算模型预测框与真实边界框的重叠度

        pred_overlaps = self.iou_calculator(gt_bboxes, bboxes, mode=self.assign_metric)
        # pred_overlaps = self.iou_calculator(gt_bboxes, pred_bboxes, mode='iou', fun=self.fun)
        # 计算真实框与要分配的边界框的重叠度
        overlaps = self.iou_calculator(gt_bboxes, bboxes, mode=self.assign_metric)

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(
                    bboxes, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(
                    gt_bboxes_ignore, bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        # 调用assign_wrt_ranking方法进行真正的分配
        assign_result = self.assign_wrt_ranking(overlaps, pred_overlaps, gt_labels)

        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result
 
    def assign_wrt_ranking(self, overlaps, pred_overlaps, gt_labels=None):
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)   

        # 1. assign -1 by default
        # 初始化分配结果矩阵 assigned_gt_inds，默认值为 -1
        assigned_gt_inds = overlaps.new_full((num_bboxes,),
                                             -1,
                                             dtype=torch.long)

        # 如果没有真实目标框或者没有边界框，返回空的分配结果
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes,),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        # 通过比较每个边界框与所有真实边界框的重叠，找到与每个边界框最匹配的真实边界框
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        assigned_gt_inds[(max_overlaps >= 0) & (max_overlaps < 0.3)] = 0 # pre-assign neg samples

        # 创建一个匹配矩阵，记录每个真实边界框与预测边界框的匹配情况
        matching_matrix = torch.zeros_like(overlaps, dtype=torch.uint8)
        # 限制候选框数量为self.candidate_topk或预测边界框的数量的最小值
        candidate_topk = min(self.candidate_topk, pred_overlaps.size(1))
        # 获取预测边界框中与真实边界框重叠最大的前 candidate_topk 个边界框及其对应的IoU
        topk_ious, _ = torch.topk(pred_overlaps, candidate_topk, dim=1)
        # calculate dynamic k for each gt
        # 计算每个真实框的动态 k 值，即与其重叠最大的前k个边界框，并确保最终结果大于1
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)

        # socre for topk scores assigner, different from topk assigner
        # 如果需要考虑分数信息
        if self.scores:
            for gt_idx in range(num_gts):
                # 找到与每个真实边界框重叠最大的前 k 个边界框
                gt_max_overlaps, _ = overlaps[gt_idx,:].topk(dynamic_ks[gt_idx], largest=True, sorted=True)
                max_overlap_inds = torch.zeros_like(overlaps[gt_idx,:], dtype=bool)
                # 将匹配的边界框标记为匹配状态
                for key in range(dynamic_ks[gt_idx]):
                    max_overlap_inds = max_overlap_inds | (overlaps[gt_idx,:] == gt_max_overlaps[key])
                matching_matrix[gt_idx, :][max_overlap_inds] = 1

        # 如果不考虑分数信息
        else:
            for gt_idx in range(num_gts):
                # 找到与每个真实边界框重叠最大的前 k 个边界框
                _, pos_idx = torch.topk(
                    overlaps[gt_idx, :], k=dynamic_ks[gt_idx])
                # 将匹配的边界框标记为匹配状态
                matching_matrix[gt_idx, :][pos_idx] = 1
            del pos_idx

        del topk_ious, dynamic_ks
        
        #如果有anchors被分配给了多个gt的处理方式, 如果有选择代价最小的那个gt作为分配样本。
        prior_match_gt_mask = matching_matrix.sum(0) > 1
        if prior_match_gt_mask.sum() > 0:
            iou_max, iou_argmax = torch.max(
                overlaps[:, prior_match_gt_mask], dim=0)
            matching_matrix[:, prior_match_gt_mask] *= 0
            matching_matrix[iou_argmax, prior_match_gt_mask] = 1

        # 计算匹配后的正样本掩码，找到与预测框匹配的真实边界框索引
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        matched_pred_ious = (matching_matrix *
                             overlaps).sum(0)[fg_mask_inboxes]

        # 更新分配结果，将匹配的真实边界框索引赋给相应的预测边界框
        assigned_gt_inds[fg_mask_inboxes] = matched_gt_inds + 1
       

        # 如果有真实标签，将分配的标签添加到结果里
        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, matched_pred_ious, labels=assigned_labels)


@BBOX_ASSIGNERS.register_module()
class DynamicTopkAssignerR(BaseAssigner):
    """For each gt box, assign k pos samples to it. The remaining samples are assigned with a neg label.

    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    Args:
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxesDynamicTopkAssignerR_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
        iou_calculator (str): The class of calculating bbox similarity, including BboxOverlaps2D and BboxDistanceMetric
        assign_metric (str): The metric of measuring the similarity between boxes.
        topk (int): assign k positive samples to each gt.
    """

    def __init__(self,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 gpu_assign_thr=-1,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 assign_metric='iou',
                 candidate_topk=10,
                 center_radius=2.5):
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.assign_metric = assign_metric
        self.candidate_topk = candidate_topk
        self.center_radius = center_radius

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, or a semi-positive number. -1 means negative
        sample, semi-positive number is the index (0-based) of assigned gt.
        The assignment is done in following steps, the order matters.

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.

        Example:
            >>> self = RankingAssigner(0.5, 0.5)
            >>> bboxes = torch.Tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
            >>> gt_bboxes = torch.Tensor([[0, 0, 10, 9]])
            >>> assign_result = self.assign(bboxes, gt_bboxes)
            >>> expected_gt_inds = torch.LongTensor([1, 0])
            >>> assert torch.all(assign_result.gt_inds == expected_gt_inds)
        """
        # assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
        #         gt_bboxes.shape[0] > self.gpu_assign_thr) else False
        assign_on_cpu = False
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = bboxes.device
            bboxes = bboxes.cpu()

            pred_bboxes = pred_bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()

        pred_overlaps = self.iou_calculator(bboxes, gt_bboxes, mode=self.assign_metric)
        # pred_overlaps = self.iou_calculator(valid_decoded_bbox, gt_bboxes, mode=self.assign_metric)
        overlaps = self.iou_calculator(bboxes, gt_bboxes, mode=self.assign_metric)

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(
                    bboxes, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(
                    gt_bboxes_ignore, bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            pred_overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        assign_result = self.assign_wrt_ranking(overlaps, pred_overlaps, gt_labels)

        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result

    def assign_wrt_ranking(self, overlaps, pred_overlaps, gt_labels=None):
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes,),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes,),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        assigned_gt_inds[(max_overlaps >= 0) & (max_overlaps < 0.3)] = 0  # pre-assign neg samples

        matching_matrix = torch.zeros_like(overlaps, dtype=torch.uint8)
        candidate_topk = min(self.candidate_topk, pred_overlaps.size(1))
        topk_ious, _ = torch.topk(pred_overlaps, candidate_topk, dim=1)
        # calculate dynamic k for each gt
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        topk = dynamic_ks.max()
        gt_max_overlaps, gt_argmax_overlaps = overlaps.topk(topk, dim=1, largest=True, sorted=True)

        for gt_idx in range(num_gts):
            _, pos_idx = torch.topk(
                overlaps[:, gt_idx], k=dynamic_ks[gt_idx])
            matching_matrix[:, gt_idx][pos_idx] = 1

        for gt_idx in range(num_gts):
            for j in range(dynamic_ks[gt_idx]):
                max_overlap_inds = overlaps[:, gt_idx] == gt_max_overlaps[j, gt_idx]
            matching_matrix[:, gt_idx][max_overlap_inds] = 1

        del topk_ious, dynamic_ks, pos_idx

        # 如果有anchors被分配给了多个gt的处理方式, 如果有选择代价最小的那个gt作为分配样本。
        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            iou_max, iou_argmax = torch.max(
                overlaps[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0
            matching_matrix[prior_match_gt_mask, iou_argmax] = 1
        fg_mask_inboxes = matching_matrix.sum(1) > 0
        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_pred_ious = (matching_matrix *
                             overlaps).sum(1)[fg_mask_inboxes]

        assigned_gt_inds[fg_mask_inboxes] = matched_gt_inds + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, matched_pred_ious, labels=assigned_labels)
