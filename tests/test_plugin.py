from hydra._internal.config_search_path_impl import ConfigSearchPathImpl

from hydra_plugins.nvbenjo.searchpath_plugin import NvbenjoSearchPathPlugin


def _make_search_path():
    """Simulate the search path Hydra builds before plugins run."""
    search_path = ConfigSearchPathImpl()
    search_path.append(provider="hydra", path="pkg://hydra/conf")
    search_path.append(provider="main", path="pkg://nvbenjo/conf")
    return search_path


def test_search_path_user_before_builtin(monkeypatch):
    """User configs (CWD) should take priority over built-in package configs."""
    monkeypatch.setattr("sys.argv", ["nvbenjo", "-cn", "small.yaml"])

    search_path = _make_search_path()
    plugin = NvbenjoSearchPathPlugin()
    plugin.manipulate_search_path(search_path)

    providers = [el.provider for el in search_path.get_path()]
    # CWD must come before both "main" and "nvbenjo-plugin" (the built-in paths)
    assert providers.index("nvbenjo-user") < providers.index("main")
    assert providers.index("nvbenjo-user") < providers.index("nvbenjo-plugin")
    # But after Hydra's own internals
    assert providers.index("hydra") < providers.index("nvbenjo-user")
    # No -cd, so nvbenjo-user-cd should not be present
    assert "nvbenjo-user-cd" not in providers


def test_search_path_config_dir_before_builtin(monkeypatch):
    """When -cd is used (e.g. via _fix_config_path), that dir must also take priority."""
    monkeypatch.setattr("sys.argv", ["nvbenjo", "-cn", "small.yaml", "-cd", "/tmp/testcfg"])

    search_path = _make_search_path()
    plugin = NvbenjoSearchPathPlugin()
    plugin.manipulate_search_path(search_path)

    providers = [el.provider for el in search_path.get_path()]
    # -cd path must come before CWD, "main", and "nvbenjo-plugin"
    assert providers.index("nvbenjo-user-cd") < providers.index("nvbenjo-user")
    assert providers.index("nvbenjo-user-cd") < providers.index("main")
    assert providers.index("nvbenjo-user-cd") < providers.index("nvbenjo-plugin")
    # But after Hydra's own internals
    assert providers.index("hydra") < providers.index("nvbenjo-user-cd")
