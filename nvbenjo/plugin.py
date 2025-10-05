from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class NvbenjoSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="nvbenjo-plugin", path="pkg://nvbenjo/conf")
        # allow from current directory
        search_path.append(provider="nvbenjo-user", path="file://${hydra.runtime.cwd}")
