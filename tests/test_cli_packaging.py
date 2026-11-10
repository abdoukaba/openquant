import subprocess
import sys
import pandas as pd
import pytest
from openquant.cli import main


def test_cli_help_and_module_entrypoint():
    result = subprocess.run(
        [sys.executable, "-m", "openquant", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "backtest" in result.stdout


def test_installed_package_imports_outside_checkout(tmp_path):
    script = """
import importlib
import pkgutil
import openquant
for module in pkgutil.walk_packages(openquant.__path__, openquant.__name__ + '.'):
    importlib.import_module(module.name)
from openquant.cli import main
assert callable(main)
assert openquant.Backtest
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_offline_demo_creates_reconciling_report(tmp_path):
    assert main(["demo", "--out-dir", str(tmp_path)]) == 0
    frame = pd.read_csv(tmp_path / "performance.csv", index_col=0)
    pd.testing.assert_series_equal((1 + frame["return"]).cumprod(), frame.equity, check_names=False)
    assert (tmp_path / "tear_sheet.html").exists()


def test_cli_rejects_mismatched_weights_before_network():
    with pytest.raises(SystemExit) as error:
        main(
            [
                "backtest",
                "target-weights",
                "--tickers",
                "A",
                "B",
                "--weights",
                "1",
                "--start",
                "2024-01-01",
                "--end",
                "2024-02-01",
            ]
        )
    assert error.value.code == 2


def test_core_import_succeeds_without_optional_dependencies():
    script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'matplotlib', 'sklearn', 'pandas_market_calendars'}:
        raise ImportError('optional dependency not installed')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import openquant
assert openquant.Backtest
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
