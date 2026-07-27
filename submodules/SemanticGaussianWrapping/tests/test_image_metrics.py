import torch

from evaluation import ImageMetricAccumulator, confusion_matrix, mean_iou


def test_confusion_and_iou_respect_ignore() -> None:
    target = torch.tensor([[0, 0, 1], [1, -1, 1]])
    prediction = torch.tensor([[0, 1, 1], [1, 0, 0]])
    matrix = confusion_matrix(prediction, target, 2)
    assert matrix.tolist() == [[1, 1, 0], [1, 2, 0]]
    score, classes = mean_iou(matrix)
    assert torch.allclose(classes, torch.tensor([1 / 3, 1 / 2], dtype=torch.float64))
    assert torch.allclose(score, torch.tensor(5 / 12, dtype=torch.float64))


def test_invalid_prediction_counts_as_false_negative() -> None:
    target = torch.tensor([0, 1])
    prediction = torch.tensor([-1, 1])
    matrix = confusion_matrix(prediction, target, 2)
    assert matrix.tolist() == [[0, 0, 1], [0, 1, 0]]
    score, classes = mean_iou(matrix)
    assert torch.allclose(classes, torch.tensor([0.0, 1.0], dtype=torch.float64))
    assert torch.allclose(score, torch.tensor(0.5, dtype=torch.float64))


def test_perfect_image_metrics() -> None:
    image = torch.rand(3, 16, 16)
    accumulator = ImageMetricAccumulator()
    accumulator.update_image(image, image)
    result = accumulator.compute()
    assert result["l1"] == 0.0
    assert result["ssim"] > 0.999


def test_all_ignored_semantics_serialize_as_null() -> None:
    accumulator = ImageMetricAccumulator()
    accumulator.update_semantic(torch.tensor([-1]), torch.tensor([-1]), 1)
    result = accumulator.compute()
    assert result["semantic_miou"] is None
    assert result["semantic_iou"] == [None]
