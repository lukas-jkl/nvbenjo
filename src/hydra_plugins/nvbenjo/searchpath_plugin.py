import os
import sys

from hydra.core.config_search_path import ConfigSearchPath, SearchPathQuery
from hydra.plugins.search_path_plugin import SearchPathPlugin


class NvbenjoSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="nvbenjo-plugin", path="pkg://nvbenjo/conf")
        # Insert CWD just before the @hydra.main config_path (provider="main")
        # so user configs take priority over built-in ones,
        # but after Hydra's internal paths (to avoid shadowing hydra internals)
        search_path.prepend(
            provider="nvbenjo-user",
            path=f"file://{os.getcwd()}",
            anchor=SearchPathQuery(provider="main"),
        )
        # Hydra's -cd/--config-dir is added at the end of the search path (low priority),
        # which means built-in configs with the same name shadow user configs.
        # Fix this by also prepending the -cd path at high priority.
        # Prepend before nvbenjo-user so an explicit path beats CWD.
        config_dir = self._get_config_dir()
        if config_dir is not None:
            search_path.prepend(
                provider="nvbenjo-user-cd",
                path=f"file://{config_dir}",
                anchor=SearchPathQuery(provider="nvbenjo-user"),
            )

    @staticmethod
    def _get_config_dir() -> "str | None":
        for flag in ("-cd", "--config-dir"):
            if flag in sys.argv:
                idx = sys.argv.index(flag) + 1
                if idx < len(sys.argv):
                    return os.path.abspath(sys.argv[idx])
        return None
