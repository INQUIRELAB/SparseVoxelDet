import contextlib

import pytest
import torch

import quality_aligned_loss as qal
import strict_loss


TRAINER_DIAGNOSTIC_KEYS = {
    "num_gt", "num_gt_with_candidates", "gt_zero_candidates", "dynamic_k_sum",
    "num_pos_raw", "quota_fill_ratio", "quota_deficit", "conflict_sites", "gt_zero_after_conflict",
    "multi_gt_samples", "multi_gt_gt_zero_assigned", "candidate_count_mean",
    "candidate_count_max", "classification_quality_target_mean",
    "classification_quality_target_max", "decoded_iou_target_mean",
    "decoded_iou_target_max",
}


def _outputs(indices, pred_boxes=None, cls=None, ctr=None, stride=4, requires_grad=False):
    indices = torch.tensor(indices, dtype=torch.long).reshape(-1, 3)
    n = len(indices)
    if pred_boxes is None:
        pred_boxes = []
        for _, y, x in indices.tolist():
            cx, cy = x * stride + stride / 2, y * stride + stride / 2
            pred_boxes.append([cx - 2, cy - 2, cx + 2, cy + 2])
    pred_boxes = torch.tensor(pred_boxes, dtype=torch.float32).reshape(-1, 4)
    centers_x = indices[:, 2].float() * stride + stride / 2
    centers_y = indices[:, 1].float() * stride + stride / 2
    distances = torch.stack(
        (
            centers_x - pred_boxes[:, 0],
            centers_y - pred_boxes[:, 1],
            pred_boxes[:, 2] - centers_x,
            pred_boxes[:, 3] - centers_y,
        ),
        dim=1,
    ).clamp_min(0.01)
    cls_logits = torch.tensor(cls if cls is not None else [0.0] * n, dtype=torch.float32).reshape(n, 1)
    ctr_logits = torch.tensor(ctr if ctr is not None else [0.0] * n, dtype=torch.float32).reshape(n, 1)
    box_ltrb = distances.log()
    if requires_grad:
        cls_logits.requires_grad_()
        box_ltrb.requires_grad_()
        ctr_logits.requires_grad_()
    return {
        "cls_logits": cls_logits,
        "box_ltrb": box_ltrb,
        "ctr_logits": ctr_logits,
        "indices_2d": indices,
        "spatial_2d": (32, 32),
    }


def _enabled(epoch=0):
    loss = qal.SparseVoxelDetLoss(
        task_aligned_enabled=True,
        task_aligned_alpha=1.0,
        task_aligned_beta=6.0,
        dynamic_k_topq=10,
        quality_bootstrap_epochs=2,
        nwd_weight=0.5,
        nwd_c=12.8,
    )
    loss.set_epoch(epoch)
    return loss


def _clone_outputs(outputs):
    cloned = {}
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            value = value.clone().detach()
            if key in {"cls_logits", "box_ltrb", "ctr_logits"}:
                value.requires_grad_()
            cloned[key] = value
        else:
            cloned[key] = value
    return cloned


def test_disabled_path_exact_outputs_and_input_gradients():
    outputs = _outputs(
        [[0, 2, 2], [0, 2, 5], [0, 8, 8]],
        pred_boxes=[[4, 4, 12, 12], [16, 4, 24, 12], [30, 30, 38, 38]],
        cls=[0.2, -0.4, 0.7],
        ctr=[0.1, -0.3, 0.5],
    )
    boxes = [torch.tensor([[4.0, 4.0, 12.0, 12.0], [16.0, 4.0, 24.0, 12.0]])]
    labels = [torch.ones(2, dtype=torch.long)]
    strict_outputs = _clone_outputs(outputs)
    disabled_outputs = _clone_outputs(outputs)
    expected = strict_loss.SparseVoxelDetLoss(nwd_weight=0.5)(strict_outputs, boxes, labels)
    actual = qal.SparseVoxelDetLoss(nwd_weight=0.5)(disabled_outputs, boxes, labels)
    assert expected.keys() == actual.keys()
    for key in expected:
        assert torch.equal(actual[key], expected[key]), key
    expected["loss"].backward()
    actual["loss"].backward()
    for key in ("cls_logits", "box_ltrb", "ctr_logits"):
        assert torch.equal(disabled_outputs[key].grad, strict_outputs[key].grad), key


def test_enabled_result_exposes_exact_trainer_diagnostic_names():
    outputs = _outputs([[0, 1, 1]], pred_boxes=[[0, 0, 12, 12]])
    boxes = [torch.tensor([[0.0, 0.0, 16.0, 16.0]])]
    result = _enabled(0)(outputs, boxes, [torch.ones(1, dtype=torch.long)])
    assert set(qal._DIAGNOSTIC_KEYS) == TRAINER_DIAGNOSTIC_KEYS
    assert TRAINER_DIAGNOSTIC_KEYS <= set(result)
    aliases = {"gt_count", "gt_with_candidates", "dynamic_quota_sum", "positives", "gt_unassigned_after_conflict", "multi_gt_unassigned_gts", "cls_quality_mean", "cls_quality_max"}
    assert not (aliases & set(result))


def test_every_strictly_inside_site_is_candidate_and_outside_is_not():
    outputs = _outputs([[0, 0, 0], [0, 10, 10], [0, 13, 13]])
    boxes = [torch.tensor([[0.0, 0.0, 48.0, 48.0]])]
    labels = [torch.ones(1, dtype=torch.long)]
    targets = qal.assign_quality_targets(outputs["indices_2d"], outputs["cls_logits"], outputs["box_ltrb"], boxes, labels)
    assert targets["candidate_count_max"].item() == 2
    assert targets["num_gt_with_candidates"].item() == 1
    assert targets["pos_mask"][2].item() is False


def test_zero_candidates_are_diagnostic_without_force_match():
    outputs = _outputs([[0, 0, 0], [0, 1, 1]])
    boxes = [torch.tensor([[100.0, 100.0, 120.0, 120.0]])]
    labels = [torch.ones(1, dtype=torch.long)]
    targets = qal.assign_quality_targets(outputs["indices_2d"], outputs["cls_logits"], outputs["box_ltrb"], boxes, labels)
    assert targets["pos_mask"].sum().item() == 0
    assert targets["gt_zero_candidates"].item() == 1
    assert targets["dynamic_k_sum"].item() == 0


def test_zero_active_sites_are_graph_connected_and_diagnostic():
    outputs = _outputs([], requires_grad=True)
    boxes = [torch.tensor([[0.0, 0.0, 16.0, 16.0]])]
    labels = [torch.ones(1, dtype=torch.long)]
    result = _enabled(0)(outputs, boxes, labels)
    assert result["num_pos_raw"].item() == 0
    assert result["gt_zero_candidates"].item() == 1
    assert result["reg_loss"].item() == 0
    assert result["quality_loss"].item() == 0
    result["loss"].backward()
    assert outputs["cls_logits"].grad is not None
    assert outputs["box_ltrb"].grad is not None
    assert outputs["ctr_logits"].grad is not None


def test_dynamic_k_keeps_one_candidate_when_all_decoded_ious_are_zero():
    outputs = _outputs([[0, 1, 1]])
    outputs["box_ltrb"].fill_(-100.0)
    boxes = [torch.tensor([[0.0, 0.0, 16.0, 16.0]])]
    labels = [torch.ones(1, dtype=torch.long)]
    targets = qal.assign_quality_targets(
        outputs["indices_2d"], outputs["cls_logits"], outputs["box_ltrb"], boxes, labels
    )
    assert targets["decoded_iou_target_max"].item() == 0
    assert targets["dynamic_k_sum"].item() == 1
    assert targets["num_pos_raw"].item() == 1


def test_dynamic_k_uses_floor_of_top_iou_sum():
    indices = [[0, 1, 1], [0, 1, 2], [0, 2, 1]]
    target = [0.0, 0.0, 16.0, 16.0]
    outputs = _outputs(indices, pred_boxes=[target, target, target], cls=[2.0, 1.0, 0.0])
    targets = qal.assign_quality_targets(outputs["indices_2d"], outputs["cls_logits"], outputs["box_ltrb"], [torch.tensor([target])], [torch.ones(1, dtype=torch.long)])
    assert targets["dynamic_k_sum"].item() == 3
    assert targets["num_pos_raw"].item() == 3
    assert targets["quota_fill_ratio"].item() == 1


def test_alignment_and_iou_targets_are_detached():
    outputs = _outputs([[0, 1, 1]], pred_boxes=[[0, 0, 12, 12]], cls=[0.5], requires_grad=True)
    boxes = [torch.tensor([[0.0, 0.0, 16.0, 16.0]])]
    labels = [torch.ones(1, dtype=torch.long)]
    targets = qal.assign_quality_targets(outputs["indices_2d"], outputs["cls_logits"], outputs["box_ltrb"], boxes, labels)
    assert not targets["cls_quality"].requires_grad
    assert not targets["iou_targets"].requires_grad
    cls_loss = strict_loss.BinaryQualityFocalLoss()(outputs["cls_logits"], targets["cls_quality"], targets["pos_mask"])
    cls_loss.backward()
    assert outputs["cls_logits"].grad is not None
    assert outputs["box_ltrb"].grad is None


def test_disjoint_two_gt_assignment():
    boxes = torch.tensor([[0.0, 0.0, 12.0, 12.0], [20.0, 0.0, 32.0, 12.0]])
    outputs = _outputs([[0, 1, 1], [0, 1, 6]], pred_boxes=boxes.tolist(), cls=[1.0, 1.0])
    targets = qal.assign_quality_targets(outputs["indices_2d"], outputs["cls_logits"], outputs["box_ltrb"], [boxes], [torch.ones(2, dtype=torch.long)])
    assert targets["num_pos_raw"].item() == 2
    assert targets["conflict_sites"].item() == 0
    assert targets["gt_zero_after_conflict"].item() == 0


def test_overlapping_conflict_is_one_site_one_gt_and_deterministic():
    boxes = torch.tensor([[0.0, 0.0, 20.0, 20.0], [4.0, 4.0, 16.0, 16.0]])
    outputs = _outputs([[0, 2, 2], [0, 1, 1]], pred_boxes=[[4, 4, 16, 16], [0, 0, 20, 20]], cls=[1.0, 1.0])
    first = qal.assign_quality_targets(outputs["indices_2d"], outputs["cls_logits"], outputs["box_ltrb"], [boxes], [torch.ones(2, dtype=torch.long)])
    second = qal.assign_quality_targets(outputs["indices_2d"], outputs["cls_logits"], outputs["box_ltrb"], [boxes], [torch.ones(2, dtype=torch.long)])
    assert first["conflict_sites"].item() >= 1
    assert torch.equal(first["assigned_gt"], second["assigned_gt"])
    assert first["pos_mask"].sum().item() <= len(outputs["indices_2d"])


def test_rejected_gt_continues_to_its_next_candidate():
    boxes = torch.tensor([[0.0, 0.0, 14.0, 12.0], [2.0, 0.0, 16.0, 12.0]])
    outputs = _outputs([[0, 1, 1], [0, 1, 2]], cls=[1.0, 0.0])
    outputs["box_ltrb"].fill_(-100.0)
    targets = qal.assign_quality_targets(
        outputs["indices_2d"], outputs["cls_logits"], outputs["box_ltrb"],
        [boxes], [torch.ones(2, dtype=torch.long)],
    )
    assert targets["dynamic_k_sum"].item() == 2
    assert targets["num_pos_raw"].item() == 2
    assert targets["quota_deficit"].item() == 0
    assert targets["gt_zero_after_conflict"].item() == 0
    assert set(targets["assigned_gt"].tolist()) == {0, 1}


def test_equal_area_conflict_uses_canonical_box_tie_break():
    boxes = torch.tensor([[2.0, 0.0, 14.0, 12.0], [0.0, 0.0, 12.0, 12.0]])
    outputs = _outputs([[0, 1, 1]], pred_boxes=[[4, 4, 8, 8]], cls=[1.0])
    labels = torch.ones(2, dtype=torch.long)
    original = qal.assign_quality_targets(
        outputs["indices_2d"], outputs["cls_logits"], outputs["box_ltrb"], [boxes], [labels]
    )
    permuted = qal.assign_quality_targets(
        outputs["indices_2d"], outputs["cls_logits"], outputs["box_ltrb"], [boxes.flip(0)], [labels.flip(0)]
    )
    expected = torch.tensor([0.0, 0.0, 12.0, 12.0])
    assert torch.equal(original["target_boxes"][0], expected)
    assert torch.equal(permuted["target_boxes"][0], expected)


def test_gt_row_permutation_preserves_semantic_targets():
    boxes = torch.tensor([[0.0, 0.0, 20.0, 20.0], [4.0, 4.0, 16.0, 16.0]])
    labels = torch.tensor([1, 1], dtype=torch.long)
    outputs = _outputs([[0, 2, 2], [0, 1, 1]], pred_boxes=[[4, 4, 16, 16], [0, 0, 20, 20]], cls=[1.0, 1.0])
    original = qal.assign_quality_targets(outputs["indices_2d"], outputs["cls_logits"], outputs["box_ltrb"], [boxes], [labels])
    permuted = qal.assign_quality_targets(outputs["indices_2d"], outputs["cls_logits"], outputs["box_ltrb"], [boxes.flip(0)], [labels.flip(0)])
    assert torch.equal(original["pos_mask"], permuted["pos_mask"])
    assert torch.equal(original["target_boxes"], permuted["target_boxes"])


def test_identical_candidate_two_gt_reports_quota_deficit():
    boxes = torch.tensor([[0.0, 0.0, 12.0, 12.0], [0.0, 0.0, 12.0, 12.0]])
    outputs = _outputs([[0, 1, 1]], pred_boxes=[[0, 0, 12, 12]], cls=[1.0])
    targets = qal.assign_quality_targets(outputs["indices_2d"], outputs["cls_logits"], outputs["box_ltrb"], [boxes], [torch.ones(2, dtype=torch.long)])
    assert targets["dynamic_k_sum"].item() == 2
    assert targets["num_pos_raw"].item() == 1
    assert targets["quota_deficit"].item() == 1
    assert targets["gt_zero_after_conflict"].item() == 1
    assert targets["multi_gt_gt_zero_assigned"].item() == 1


def test_multi_gt_zero_assigned_counts_zero_candidate_gts():
    boxes = torch.tensor([[100.0, 100.0, 120.0, 120.0], [0.0, 0.0, 12.0, 12.0]])
    outputs = _outputs([[0, 1, 1]], pred_boxes=[[0, 0, 12, 12]], cls=[1.0])
    targets = qal.assign_quality_targets(
        outputs["indices_2d"], outputs["cls_logits"], outputs["box_ltrb"],
        [boxes], [torch.ones(2, dtype=torch.long)],
    )
    assert targets["gt_zero_candidates"].item() == 1
    assert targets["gt_zero_after_conflict"].item() == 0
    assert targets["multi_gt_gt_zero_assigned"].item() == 1


def test_bootstrap_epochs_zero_one_two_match_manual_quality_targets():
    outputs = _outputs([[0, 1, 1]], pred_boxes=[[0, 0, 12, 12]], cls=[0.4])
    boxes = [torch.tensor([[0.0, 0.0, 16.0, 16.0]])]
    labels = [torch.ones(1, dtype=torch.long)]
    assigned = qal.assign_quality_targets(outputs["indices_2d"], outputs["cls_logits"], outputs["box_ltrb"], boxes, labels)
    quality = assigned["cls_quality"]
    assert 0 < quality.item() < 1
    losses = []
    for epoch, blend in ((0, 0.0), (1, 0.5), (2, 1.0)):
        result = _enabled(epoch)(outputs, boxes, labels)
        manual_target = torch.tensor([(1 - blend) + blend * quality.item()])
        manual = strict_loss.BinaryQualityFocalLoss()(outputs["cls_logits"], manual_target, assigned["pos_mask"])
        assert torch.allclose(result["cls_loss"], manual, atol=1e-6, rtol=0)
        assert result["bootstrap_blend"].item() == blend
        losses.append(result["cls_loss"].item())
    assert len(set(losses)) == 3


def test_perfect_and_offset_decoded_iou_quality_targets():
    box = torch.tensor([[0.0, 0.0, 16.0, 16.0]])
    perfect = _outputs([[0, 1, 1]], pred_boxes=box.tolist())
    offset = _outputs([[0, 1, 1]], pred_boxes=[[0, 0, 12, 12]])
    perfect_t = qal.assign_quality_targets(perfect["indices_2d"], perfect["cls_logits"], perfect["box_ltrb"], [box], [torch.ones(1, dtype=torch.long)])
    offset_t = qal.assign_quality_targets(offset["indices_2d"], offset["cls_logits"], offset["box_ltrb"], [box], [torch.ones(1, dtype=torch.long)])
    assert torch.allclose(perfect_t["iou_targets"], torch.ones(1), atol=1e-6, rtol=0)
    assert 0 < offset_t["iou_targets"].item() < 1


def test_regression_is_unweighted_decoded_box_mean():
    target = [0.0, 0.0, 20.0, 20.0]
    indices = [[0, 2, 2], [0, 1, 1], [0, 3, 3]]
    pred_boxes = [target, [0, 0, 19, 19], [0, 0, 10, 10]]
    outputs = _outputs(indices, pred_boxes=pred_boxes, cls=[2.0, 1.0, -2.0])
    boxes = [torch.tensor([target])]
    labels = [torch.ones(1, dtype=torch.long)]
    targets = qal.assign_quality_targets(outputs["indices_2d"], outputs["cls_logits"], outputs["box_ltrb"], boxes, labels)
    assert targets["num_pos_raw"].item() == 2
    pos = targets["pos_mask"]
    decoded = torch.exp(outputs["box_ltrb"][pos])
    pred = strict_loss.decode_ltrb_to_boxes(outputs["indices_2d"][pos], decoded)
    tgt = targets["target_boxes"][pos]
    manual = (0.5 * qal._aligned_giou_loss(pred, tgt) + 0.5 * strict_loss.nwd_loss_xyxy(pred, tgt, c=12.8)).mean()
    result = _enabled(2)(outputs, boxes, labels)
    assert torch.allclose(result["reg_loss"], manual, atol=1e-6, rtol=0)


def test_empty_positive_losses_are_graph_connected_and_finite():
    outputs = _outputs([[0, 0, 0]], pred_boxes=[[-1, -1, 3, 3]], requires_grad=True)
    boxes = [torch.tensor([[100.0, 100.0, 120.0, 120.0]])]
    labels = [torch.ones(1, dtype=torch.long)]
    result = _enabled(0)(outputs, boxes, labels)
    assert result["reg_loss"].item() == 0
    assert result["quality_loss"].item() == 0
    assert all(torch.isfinite(value).all() for value in result.values() if isinstance(value, torch.Tensor))
    result["loss"].backward()
    assert outputs["box_ltrb"].grad is not None
    assert outputs["ctr_logits"].grad is not None
    assert torch.count_nonzero(outputs["box_ltrb"].grad).item() == 0
    assert torch.count_nonzero(outputs["ctr_logits"].grad).item() == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cpu_autocast_matches_fp32_where_supported(dtype):
    outputs = _outputs([[0, 1, 1]], pred_boxes=[[0, 0, 12, 12]], cls=[0.5], ctr=[0.2], requires_grad=True)
    boxes = [torch.tensor([[0.0, 0.0, 16.0, 16.0]])]
    labels = [torch.ones(1, dtype=torch.long)]
    reference = _enabled(2)(_clone_outputs(outputs), boxes, labels)
    try:
        context = torch.autocast("cpu", dtype=dtype)
    except RuntimeError as error:
        pytest.skip(str(error))
    with contextlib.nullcontext():
        try:
            with context:
                result = _enabled(2)(outputs, boxes, labels)
        except RuntimeError as error:
            pytest.skip(str(error))
    for key in ("loss", "cls_loss", "reg_loss", "ctr_loss", "quality_loss"):
        assert torch.allclose(result[key], reference[key], atol=1e-6, rtol=0), key
    assert all(torch.isfinite(value).all() for value in result.values() if isinstance(value, torch.Tensor))
    result["loss"].backward()
    assert torch.isfinite(outputs["cls_logits"].grad).all()
    assert torch.isfinite(outputs["box_ltrb"].grad).all()
    assert torch.isfinite(outputs["ctr_logits"].grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_true_low_precision_prediction_tensors_are_finite(dtype):
    outputs = _outputs([[0, 1, 1]], pred_boxes=[[0, 0, 12, 12]], cls=[0.5], ctr=[0.2])
    for key in ("cls_logits", "box_ltrb", "ctr_logits"):
        outputs[key] = outputs[key].detach().to(dtype).requires_grad_()
        assert outputs[key].dtype == dtype
    boxes = [torch.tensor([[0.0, 0.0, 16.0, 16.0]])]
    labels = [torch.ones(1, dtype=torch.long)]
    result = _enabled(2)(outputs, boxes, labels)
    assert all(torch.isfinite(value).all() for value in result.values() if isinstance(value, torch.Tensor))
    result["loss"].backward()
    for key in ("cls_logits", "box_ltrb", "ctr_logits"):
        assert outputs[key].grad is not None
        assert torch.isfinite(outputs[key].grad).all()
