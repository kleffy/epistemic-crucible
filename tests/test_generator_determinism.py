"""Generator determinism tests: package imports, config loading, seed determinism, world generation."""  # noqa: E501

import pathlib

import yaml


def test_package_imports():
    import crucible

    assert crucible.__version__ == "0.2.0"


def test_pytest_discovers_tests():
    assert True


def test_default_config_loads():
    config_path = pathlib.Path(__file__).parent.parent / "configs" / "default.yaml"
    assert config_path.exists(), f"Config not found: {config_path}"
    with config_path.open() as f:
        cfg = yaml.safe_load(f)
    assert "world" in cfg
    assert "logging" in cfg
    assert "generator" in cfg
    assert cfg["world"]["grid_size"] == 6
    assert cfg["world"]["seed"] == 42


def test_seed_determinism():
    from crucible.utils.seeding import make_python_rng, make_rng

    rng_a = make_rng(42)
    rng_b = make_rng(42)
    assert rng_a.integers(0, 1_000_000) == rng_b.integers(0, 1_000_000)

    py_a = make_python_rng(7)
    py_b = make_python_rng(7)
    assert py_a.randint(0, 1_000_000) == py_b.randint(0, 1_000_000)


def test_world_same_seed_identical():
    from crucible.utils.serialization import to_dict
    from crucible.world import generate_world

    w1 = generate_world(seed=42)
    w2 = generate_world(seed=42)
    assert to_dict(w1, public=False) == to_dict(w2, public=False)


def test_world_different_seeds_differ():
    from crucible.world import generate_world

    w1 = generate_world(seed=42)
    w2 = generate_world(seed=99)
    positions_1 = {o.visible.pos for o in w1.objects.values()}
    positions_2 = {o.visible.pos for o in w2.objects.values()}
    assert positions_1 != positions_2
