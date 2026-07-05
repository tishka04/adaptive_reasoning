import torch

from theory.m2.arc_lewm_model import (
    ARCLeWMModel,
    ActionConditionedPredictor,
    GridEncoder,
    action_to_id,
    encode_action_args,
    pad_crop_grid,
)


def test_grid_encoder_z_shape_stable():
    grid, mask = pad_crop_grid([[1, 2], [3, 4]], canvas_size=(8, 8))
    encoder = GridEncoder(latent_dim=32)

    z = encoder(grid.unsqueeze(0), mask.unsqueeze(0))

    assert z.shape == (1, 32)
    assert torch.isfinite(z).all()


def test_predictor_emits_next_latent_prediction():
    predictor = ActionConditionedPredictor(latent_dim=32)
    z_t = torch.randn(3, 32)
    action_ids = torch.tensor([action_to_id("ACTION6"), action_to_id("ACTION3"), action_to_id("RESET")])
    args = torch.stack(
        [
            encode_action_args({"x": 30, "y": 12}, canvas_size=(64, 64)),
            encode_action_args({}, canvas_size=(64, 64)),
            encode_action_args(None, canvas_size=(64, 64)),
        ]
    )

    z_hat = predictor(z_t, action_ids, args)

    assert z_hat.shape == (3, 32)
    assert torch.isfinite(z_hat).all()
    assert args[0, 0] > 0
    assert args[0, 1] > 0
    assert args[0, 2] == 1
    assert args[0, 3] == 1


def test_arc_lewm_model_forward_and_no_support_outputs():
    model = ARCLeWMModel(latent_dim=16)
    grid, mask = pad_crop_grid([[1, 2], [3, 4]], canvas_size=(8, 8))
    action_ids = torch.tensor([action_to_id("ACTION4")])
    args = encode_action_args({}, canvas_size=(8, 8)).unsqueeze(0)

    z_hat, z_t = model(grid.unsqueeze(0), mask.unsqueeze(0), action_ids, args)

    assert z_hat.shape == (1, 16)
    assert z_t.shape == (1, 16)
    assert all("support" not in key for key in model.state_dict())


def test_pad_crop_grid_uses_mask_for_variable_sizes():
    grid, mask = pad_crop_grid([[1, 2, 3], [4, 5, 6]], canvas_size=(3, 4))

    assert grid.shape == (3, 4)
    assert mask.sum().item() == 6
    assert grid[0, 2].item() == 3
    assert grid[2, 3].item() == 0
