from unittest.mock import patch

import pandas as pd
import pytest

from nvbenjo.plot import print_system_info, visualize_results


@pytest.fixture
def sample_results():
    return pd.DataFrame(
        {
            "model": ["m1", "m1", "m2", "m2"],
            "device": ["cpu", "cpu", "cpu", "cpu"],
            "batch_size": [1, 1, 1, 1],
            "runtime_options": ["opt1", "opt1", "opt1", "opt1"],
            "time_inference": [0.1, 0.2, 0.3, 0.4],
        }
    )


@patch("nvbenjo.plot.plt")
@patch("nvbenjo.plot.sns")
def test_visualize_results_summary_plot(mock_sns, mock_plt, sample_results, tmp_path):
    """Multiple models, single device triggers the summary plot (lines 55-71)."""
    visualize_results(sample_results, str(tmp_path), keys=["time_inference"])
    assert mock_plt.savefig.call_count == 3  # 2 per-model + 1 summary


@patch("nvbenjo.plot.plt")
@patch("nvbenjo.plot.sns")
def test_visualize_results_empty_model_device(mock_sns, mock_plt, tmp_path):
    """Model+device combo with no rows triggers continue (line 36)."""
    results = pd.DataFrame(
        {
            "model": ["m1"],
            "device": ["cpu"],
            "batch_size": [1],
            "runtime_options": ["opt1"],
            "time_inference": [0.1],
        }
    )
    # Two devices in data but m1 only has cpu rows → cuda combo is empty
    results = pd.concat(
        [
            results,
            pd.DataFrame(
                {
                    "model": ["m2"],
                    "device": ["cuda"],
                    "batch_size": [1],
                    "runtime_options": ["opt1"],
                    "time_inference": [0.2],
                }
            ),
        ]
    )
    visualize_results(results, str(tmp_path), keys=["time_inference"])
    # m1+cuda and m2+cpu are empty → continue. Only m1+cpu and m2+cuda produce plots.
    assert mock_plt.savefig.call_count == 2


@patch("nvbenjo.plot.console")
def test_print_system_info_no_gpus(mock_console):
    """No GPUs triggers the 'None' branch (line 106)."""
    system_info = {
        "os": {"system": "Linux", "node": "test", "release": "6.0", "version": "#1"},
        "cpu": {"model": "TestCPU", "architecture": "x86_64", "cores": {"physical": 4, "total": 8}},
        "gpus": [],
    }
    print_system_info(system_info)
    mock_console.print.assert_called_once()
