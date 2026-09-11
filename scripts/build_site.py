"""
Assemble the static site in docs/ from site/.

Run from the repo root after scripts/export_site_data.py:

    python scripts/build_site.py

docs/ is what GitHub Pages serves. It contains three files:

    index.html    the page (site/index.dc.html with a real <title> and no editor-only head)
    support.js    the Claude Design page runtime, which renders the template and loads React
    mmm-data.js   the generated model outputs

No bundler is involved, so a refit is: fit_model.py -> export_site_data.py -> build_site.py -> commit.
"""

import os
import shutil

SRC = "site"
OUT = "docs"


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    page = open(os.path.join(SRC, "index.dc.html"), encoding="utf-8").read()
    # The design editor's head is minimal; give the hosted page a title and description.
    page = page.replace(
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n<script src="./support.js"></script>',
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<title>MMM Sandbox</title>\n'
        '<meta name="description" content="A Bayesian media mix model, tested on data with a known answer, that turns channel effects into a budget recommendation with honest uncertainty.">\n'
        '<script src="./support.js"></script>',
    )
    assert "<title>MMM Sandbox</title>" in page, "head replacement failed; site/index.dc.html head changed?"
    open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(page)
    for name in ("support.js", "mmm-data.js"):
        shutil.copyfile(os.path.join(SRC, name), os.path.join(OUT, name))
    print("built docs/: " + ", ".join(f"{n} ({os.path.getsize(os.path.join(OUT, n)) // 1024} KB)" for n in ("index.html", "support.js", "mmm-data.js")))


if __name__ == "__main__":
    main()
