from __future__ import annotations

import importlib.metadata
from typing import Any

project = "community_hierarchy"
copyright = "2026, Maximilian Jerdee"
author = "Maximilian Jerdee"
version = release = importlib.metadata.version("community-hierarchy")

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx.ext.viewcode",
]

source_suffix = [".rst", ".md"]
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
    "ARCHITECTURE.md",   # included by design.md
]

html_theme = "furo"

html_theme_options: dict[str, Any] = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/maxjerdee/community-hierarchy",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
    "source_repository": "https://github.com/maxjerdee/community-hierarchy",
    "source_branch": "main",
    "source_directory": "docs/",
}

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
]
myst_heading_anchors = 3

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
]
# NumPy-style type descriptions that are not classes: literal choices ({"a", "b"}),
# ranges ("float in [0, 1]", "int >= 1"), and the words "optional", "sequence", ...
nitpick_ignore_regex = [
    ("py:class", r'.*["{}\[\]<>=].*'),
    ("py:class", r".* .*"),
    ("py:class", r"\d+(\.\d+)?"),
    ("py:class", r"optional|sequence|array-like|default|callable|iterable|hashable|path-like|or|"
                 r"colormap|colors|collection|n_nodes|shape"),
    ("py:class", r"pandas\.core\..*"),
]
autodoc_type_aliases = {"Result": "community_hierarchy.Result"}
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "nx.Graph": "networkx.Graph", "nx.DiGraph": "networkx.DiGraph",
    "nx.MultiDiGraph": "networkx.MultiDiGraph",
    "pd.DataFrame": "pandas.DataFrame", "np.ndarray": "numpy.ndarray",
}

always_document_param_types = True
