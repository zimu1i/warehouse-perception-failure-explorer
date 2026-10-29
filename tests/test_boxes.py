import torch

from wpfe.boxes import hflip_boxes

W = 100.0


def test_box_on_left_moves_to_right():
    out = hflip_boxes(torch.tensor([[10.0, 5.0, 30.0, 25.0]]), W)
    assert torch.allclose(out, torch.tensor([[70.0, 5.0, 90.0, 25.0]]))


def test_full_width_box_is_unchanged():
    box = torch.tensor([[0.0, 0.0, W, 50.0]])
    assert torch.allclose(hflip_boxes(box, W), box)


def test_flip_twice_is_identity():
    boxes = torch.tensor([[1.5, 2.0, 40.25, 9.0], [60.0, 0.0, 99.0, 80.0]])
    assert torch.allclose(hflip_boxes(hflip_boxes(boxes, W), W), boxes)


def test_result_is_valid_xyxy():
    out = hflip_boxes(torch.tensor([[10.0, 5.0, 30.0, 25.0], [0.0, 1.0, 1.0, 2.0]]), W)
    assert (out[:, 0] < out[:, 2]).all() and (out[:, 1] < out[:, 3]).all()


def test_y_is_untouched_and_input_not_modified():
    boxes = torch.tensor([[10.0, 5.0, 30.0, 25.0]])
    before = boxes.clone()
    out = hflip_boxes(boxes, W)
    assert torch.equal(boxes, before)
    assert torch.equal(out[:, [1, 3]], before[:, [1, 3]])


def test_empty():
    assert hflip_boxes(torch.zeros((0, 4)), W).shape == (0, 4)
