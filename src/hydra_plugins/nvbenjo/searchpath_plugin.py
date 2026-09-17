import os
import sys

from hydra.core.config_search_path import ConfigSearchPath, SearchPathQuery
from hydra.plugins.search_path_plugin import SearchPathPlugin


def _holds_configs(directory: str) -> bool:
    # NOTE: Hydra enumerates every config group on the search path (e.g. to render
    #       --help), which means walking the whole directory tree below each entry.
    #       we only put the dir on search path that actually looks like a config dir
    try:
        with os.scandir(directory) as entries:
            return any(
                entry.is_file() and not entry.name.startswith(".") and entry.name.endswith((".yml", ".yaml"))
                for entry in entries
            )
    except OSError:
        return False


class NvbenjoSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="nvbenjo-plugin", path="pkg://nvbenjo/conf")
        # Insert CWD just before the @hydra.main config_path (provider="main")
        # so user configs take priority over built-in ones,
        # but after Hydra's internal paths (to avoid shadowing hydra internals)
        cwd = os.getcwd()
        cd_anchor = "main"
        if _holds_configs(cwd):
            search_path.prepend(
                provider="nvbenjo-user",
                path=f"file://{cwd}",
                anchor=SearchPathQuery(provider="main"),
            )
            cd_anchor = "nvbenjo-user"
        # Hydra's -cd/--config-dir is added at the end of the search path (low priority),
        # which means built-in configs with the same name shadow user configs.
        # Fix this by also prepending the -cd path at high priority.
        # Prepend before nvbenjo-user so an explicit path beats CWD.
        config_dir = self._get_config_dir()
        if config_dir is not None:
            search_path.prepend(
                provider="nvbenjo-user-cd",
                path=f"file://{config_dir}",
                anchor=SearchPathQuery(provider=cd_anchor),
            )

    @staticmethod
    def _get_config_dir() -> "str | None":
        for flag in ("-cd", "--config-dir"):
            if flag in sys.argv:
                idx = sys.argv.index(flag) + 1
                if idx < len(sys.argv):
                    return os.path.abspath(sys.argv[idx])
        return None
