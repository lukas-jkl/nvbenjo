from nvbenjo.system_info import get_system_info


def test_get_system_info():
    system_info = get_system_info()
    assert isinstance(system_info, dict)
    assert "cpu" in system_info
    assert "gpus" in system_info
    assert "os" in system_info

    assert isinstance(system_info["gpus"], (dict, list))
    if isinstance(system_info["gpus"], list):
        for gpu in system_info["gpus"]:
            assert isinstance(gpu, dict)
            assert "name" in gpu
            assert "architecture" in gpu
            assert "memory" in gpu
            assert "cuda_capability" in gpu
            assert "driver" in gpu

    # check basic cpu info
    cpu_info = system_info["cpu"]
    assert isinstance(cpu_info, dict)
    assert "cores" in cpu_info
    assert "frequency" in cpu_info
    assert "model" in cpu_info
    assert "architecture" in cpu_info
