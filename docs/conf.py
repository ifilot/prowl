"""Build the documentation without importing or compiling PROWL."""

from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parent.parent
metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]

project = "PROWL"
author = "PROWL contributors"
copyright = "2026, PROWL contributors"
release = metadata["version"]
version = release

extensions = ["myst_parser", "sphinx_design", "sphinx.ext.mathjax"]
source_suffix = {".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]
myst_heading_anchors = 3
nitpicky = True

html_theme = "pydata_sphinx_theme"
html_title = "PROWL documentation"
html_logo = "_static/brand/prowl-owl-wordmark.png"
html_favicon = "_static/brand/prowl-owl-symbol.png"
html_static_path = ["_static"]
html_css_files = ["prowl.css"]
html_context = {"default_mode": "light"}
html_theme_options = {
    "logo": {"alt_text": "PROWL — documentation home"},
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher"],
    "navbar_persistent": ["search-button"],
    "header_links_before_dropdown": 5,
    "show_nav_level": 2,
    "navigation_depth": 3,
    "show_prev_next": True,
    "show_toc_level": 2,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
}
html_sidebars = {"index": []}
html_show_sourcelink = False
html_copy_source = False
