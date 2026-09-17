from hydra._internal.config_search_path_impl import ConfigSearchPathImpl

from hydra_plugins.nvbenjo.searchpath_plugin import NvbenjoSearchPathPlugin


def _make_search_path():
    """Simulate the search path Hydra builds before plugins run."""
    search_path = ConfigSearchPathImpl()
    search_path.append(provider="hydra", path="pkg://hydra/conf")
    search_path.append(provider="main", path="pkg://nvbenjo/conf")
    return search_path


def _chdir_to_config_dir(monkeypatch, tmp_path):
    """CWD is only added to the search path when it actually holds configs."""
    (tmp_path / "myconfig.yaml").write_text("")
    monkeypatch.chdir(tmp_path)


def _providers(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", argv)
    search_path = _make_search_path()
    NvbenjoSearchPathPlugin().manipulate_search_path(search_path)
    return [el.provider for el in search_path.get_path()]


def test_search_path_user_before_builtin(monkeypatch, tmp_path):
    """User configs (CWD) should take priority over built-in package configs."""
    _chdir_to_config_dir(monkeypatch, tmp_path)
    providers = _providers(monkeypatch, ["nvbenjo", "-cn", "small.yaml"])

    # CWD must come before both "main" and "nvbenjo-plugin" (the built-in paths)
    assert providers.index("nvbenjo-user") < providers.index("main")
    assert providers.index("nvbenjo-user") < providers.index("nvbenjo-plugin")
    # But after Hydra's own internals
    assert providers.index("hydra") < providers.index("nvbenjo-user")
    # No -cd, so nvbenjo-user-cd should not be present
    assert "nvbenjo-user-cd" not in providers


def test_search_path_config_dir_before_builtin(monkeypatch, tmp_path):
    """When -cd is used (e.g. via _fix_config_path), that dir must also take priority."""
    _chdir_to_config_dir(monkeypatch, tmp_path)
    providers = _providers(monkeypatch, ["nvbenjo", "-cn", "small.yaml", "-cd", "/tmp/testcfg"])

    # -cd path must come before CWD, "main", and "nvbenjo-plugin"
    assert providers.index("nvbenjo-user-cd") < providers.index("nvbenjo-user")
    assert providers.index("nvbenjo-user-cd") < providers.index("main")
    assert providers.index("nvbenjo-user-cd") < providers.index("nvbenjo-plugin")
    # But after Hydra's own internals
    assert providers.index("hydra") < providers.index("nvbenjo-user-cd")


def test_search_path_skips_cwd_without_configs(monkeypatch, tmp_path):
    """A CWD without configs is left off the search path, so Hydra does not walk it."""
    monkeypatch.chdir(tmp_path)
    providers = _providers(monkeypatch, ["nvbenjo", "-cn", "small.yaml"])

    assert "nvbenjo-user" not in providers
    assert providers.index("main") < providers.index("nvbenjo-plugin")


def test_search_path_config_dir_without_cwd_configs(monkeypatch, tmp_path):
    """-cd must still take priority when CWD holds no configs."""
    monkeypatch.chdir(tmp_path)
    providers = _providers(monkeypatch, ["nvbenjo", "-cn", "small.yaml", "-cd", "/tmp/testcfg"])

    assert "nvbenjo-user" not in providers
    assert providers.index("nvbenjo-user-cd") < providers.index("main")
    assert providers.index("nvbenjo-user-cd") < providers.index("nvbenjo-plugin")
    assert providers.index("hydra") < providers.index("nvbenjo-user-cd")
