import numpy as np
import pytest
import SimpleITK as sitk

from lungct_kit import (
    Volume,
    crop_patch,
    read_volume,
    resample,
    voxel_to_world,
    window_hu,
    world_to_voxel,
    write_volume,
)

from .conftest import make_scan

FLIP_X = np.diag([1.0, 1.0, -1.0])  # zyx order: x axis flipped


@pytest.mark.parametrize("direction", [None, FLIP_X])
def test_world_voxel_roundtrip(direction):
    vol, world = make_scan(center_zyx=(5, 20, 33), direction=direction)
    np.testing.assert_allclose(world_to_voxel(world, vol), (5, 20, 33), atol=1e-9)
    np.testing.assert_allclose(voxel_to_world(world_to_voxel(world, vol), vol), world)


@pytest.mark.parametrize("suffix", [".mha", ".nii.gz", ".npy"])
@pytest.mark.parametrize("direction", [None, FLIP_X])
def test_write_read_keeps_geometry(tmp_path, suffix, direction):
    vol, world = make_scan(direction=direction)
    back = read_volume(write_volume(vol, tmp_path / f"scan{suffix}"))
    np.testing.assert_array_equal(back.array, vol.array)
    np.testing.assert_allclose(back.spacing, vol.spacing, rtol=1e-6)
    np.testing.assert_allclose(back.origin, vol.origin, rtol=1e-6)
    np.testing.assert_allclose(back.direction, vol.direction, atol=1e-6)
    np.testing.assert_allclose(world_to_voxel(world, back), world_to_voxel(world, vol), atol=1e-4)


@pytest.mark.parametrize("direction", [None, FLIP_X])
def test_world_to_voxel_matches_simpleitk(tmp_path, direction):
    vol, world = make_scan(center_zyx=(7, 11, 41), direction=direction)
    img = sitk.ReadImage(str(write_volume(vol, tmp_path / "s.mha")))
    sitk_xyz = img.TransformPhysicalPointToContinuousIndex([float(w) for w in world])
    np.testing.assert_allclose(world_to_voxel(world, vol), sitk_xyz[::-1], atol=1e-6)


def test_volume_validation():
    with pytest.raises(ValueError):
        Volume(np.zeros((4, 4)), (1, 1, 1))
    with pytest.raises(ValueError):
        Volume(np.zeros((4, 4, 4)), (1, 0, 1))


def test_window_hu():
    out = window_hu(np.array([-2000, -1000, -300, 400, 3000]))
    np.testing.assert_allclose(out, [0, 0, 0.5, 1, 1])
    assert out.dtype == np.float32
    with pytest.raises(ValueError):
        window_hu(np.zeros(3), 10, 10)


def test_resample_preserves_extent_and_nodule():
    vol, world = make_scan()
    iso = resample(vol, 1.0)
    assert iso.shape == (60, 42, 42)
    np.testing.assert_allclose(iso.extent_mm, vol.extent_mm)
    assert iso.array.dtype == np.int16
    center = np.round(world_to_voxel(world, iso)).astype(int)
    assert iso.array[tuple(center)] > 0  # still inside the nodule
    with pytest.raises(ValueError):
        resample(vol, 0)


def test_crop_patch_inside_edge_and_outside():
    arr = np.arange(10 * 10 * 10, dtype=np.int16).reshape(10, 10, 10)
    p = crop_patch(arr, (5, 5, 5), 4)
    np.testing.assert_array_equal(p, arr[3:7, 3:7, 3:7])

    edge = crop_patch(arr, (0, 0, 0), (4, 4, 4))
    assert edge.shape == (4, 4, 4)
    assert (edge[:2] == -1024).all() and edge[2, 2, 2] == arr[0, 0, 0]

    far = crop_patch(arr, (100, 100, 100), 3)
    assert (far == -1024).all()
    with pytest.raises(ValueError):
        crop_patch(arr, (5, 5, 5), 0)
