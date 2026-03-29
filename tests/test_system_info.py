from unittest.mock import patch, MagicMock

import pynvml

from nvbenjo.system_info import get_gpu_power_usage, get_system_info


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


@patch("nvbenjo.system_info.pynvml")
def test_get_gpu_power_usage(mock_pynvml):
    mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 150000
    assert get_gpu_power_usage(0) == 150000
    mock_pynvml.nvmlInit.assert_called_once()
    mock_pynvml.nvmlShutdown.assert_called_once()


@patch("nvbenjo.system_info.get_gpu_info", side_effect=pynvml.NVMLError_LibraryNotFound())
def test_get_system_info_no_nvidia_driver(mock_gpu):
    info = get_system_info()
    assert info["gpus"] == {}


@patch("nvbenjo.system_info.psutil")
@patch("nvbenjo.system_info.get_gpu_info", return_value=[])
@patch("nvbenjo.system_info.get_cpu_info", return_value={"brand_raw": "test", "arch_string_raw": "x86"})
def test_get_system_info_no_cpu_freq(mock_cpu, mock_gpu, mock_psutil):
    mock_psutil.virtual_memory.return_value = MagicMock(total=8_000_000_000)
    mock_psutil.cpu_count.return_value = 4
    del mock_psutil.cpu_freq  # simulate missing cpu_freq
    info = get_system_info()
    assert info["cpu"]["frequency"] == "0.00 GHz"
