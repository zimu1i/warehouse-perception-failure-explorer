import torch

from wpfe.infer import predict_both_views


class BrightSquareDetector:
    """Stub detector: one 'pallet' box around all pixels brighter than 0.5."""

    def __call__(self, images):
        outs = []
        for im in images:
            ys, xs = torch.nonzero(im[0] > 0.5, as_tuple=True)
            box = torch.tensor([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], dtype=torch.float32)
            outs.append({"boxes": box, "labels": torch.tensor([1]), "scores": torch.tensor([0.9])})
        return outs


def test_flip_view_is_mapped_back_to_original_coordinates():
    image = torch.zeros(3, 40, 100)
    image[:, 10:20, 5:25] = 1.0  # off-center square, so a missing un-flip would show
    rows = predict_both_views(BrightSquareDetector(), image, "img_x", ["pallet"], "cpu")
    by_view = {r["view"]: r for r in rows}
    assert set(by_view) == {"orig", "flip"}
    for view in ("orig", "flip"):
        r = by_view[view]
        assert (r["x_min"], r["y_min"], r["x_max"], r["y_max"]) == (5.0, 10.0, 25.0, 20.0)
        assert r["class_name"] == "pallet" and r["image_id"] == "img_x"
