import os
from pathlib import Path
import re
import sys
from typing import Any, Literal
from bs4 import BeautifulSoup
from bs4.element import *

"""
The idea of this script is to extract text from html while ensuring that mathematical content is
given in a mostly consistent way. The problem is that the internet has an inconsistently observed standard for
representing mathematical expressions.
For popular sites like wikipedia, planet-math and math-overflow, it's going to be better to extract the mathematical content directly from their API.
The hope is that this script will work for other sites.

There are 3 types of mathematical expressions that we support.
In all cases, if there is mathematical content, it should be returned between dollars (ideally double-dollars if it's block math)
and represented as LaTeX. There are also going to be problems with macros, but let's keep it simple for now.

## MathOverflow

Has the form
```
<span class="math-container">${latex content}$</span>
```

## PlanetMath

Uses mathml:

```
<math ... alttext="{latex content}">...</math>
```

## Wikipedia

Also uses mathml, but makes use of the `semantics > annotation`.

```
<math ... alttext="{latex content}">...
    <semantics> ...

        <annotation encoding="application/x-tex">{latex content}</annotation>
    </semantics>
</math>
```

Wikipedia also uses spans

```
The case <span class="texhtml"><i>a</i> = −1</span> leads to ...
```

These spans can contain unicode too, so they are not always valid LaTeX, so I think it's best to keep them
as text (`"The case a = -1 leads to ..."`) rather than trying to convert to valid latex.

So the procedure for extracting math text from math  is:
- Look for an element matching `math annotation[encoding="application/x-tex"]`, use that.
- Otherwise, look at the `alttext` attribute of the `math` tag.

We also need to figure out whether these math elements are block or inline.
This is done by inspecting the `display` attribute on the `math` element.

Unfortunately, wikipedia doesn't use this convention, instead display style is implicit from document structure and ad-hoc CSS.
The latex content is wrapped in a `{\displaystyle}` latex tag, so I think the right thing to do is just return
the latex content with this extra tag forcing display style.

In the MathOverflow case, we need only look for dollar signs.
I think this will cover most cases involving mathjax or katex, which are scanning for these dollars.
In this case we can behave the same as `get_text`.

"""


def get_latex_of_math_element(m: Tag) -> tuple[Literal["inline", "block"], str]:
    display: Any = m.get("display", "inline")
    assert display in ["inline", "block"]
    ann = m.select_one('annotation[encoding="application/x-tex"]')
    if ann is not None:
        latex = ann.get_text().strip()
    else:
        latex = m.get("alttext")
        if latex is not None:
            assert isinstance(latex, str)
            latex = latex.strip()
        else:
            raise NotImplementedError(f"No latex found for {m}")
    # wikipedia prefixes all latex with `{\displaystyle ...}` so this just strips that.
    d = re.search(r"^\s*{\\displaystyle\s(.*)}\s*$", latex, flags=re.DOTALL)
    if d is not None:
        # display = "block"
        latex = d.group(1)
        # in this case, if the parent is a `<p>` then we guess it's inline math.
        if m.find_parent("p") is not None:
            display = "inline"
        else:
            display = "block"
    return display, latex


def get_math_text(
    element: BeautifulSoup,
    separator="",
    strip=False,
):
    """Get all child strings of this PageElement, concatenated using the
    given separator.

    :param separator: Strings will be concatenated using this separator.

    :param strip: If True, strings will be stripped before being
        concatenated.

    :param types: A tuple of NavigableString subclasses. Any
        strings of a subclass not found in this list will be
        ignored. Although there are exceptions, the default
        behavior in most cases is to consider only NavigableString
        and CData objects. That means no comments, processing
        instructions, etc.

    :return: A string.
    """

    for m in element.find_all("math"):
        display, latex = get_latex_of_math_element(m)
        if display == "block":
            text = f"$${latex}$$"
        else:
            text = f"${latex}$"
        m.replace_with(text)

    return element.get_text(separator=separator, strip=strip)


def main():
    """main method, for demoing `get_text()` on some examples."""
    root = Path("./proof-pile-v2/web")
    example_html_path = root / "example_html"
    example_text_path = root / "example_text"
    example_text_path.mkdir(exist_ok=True)
    sources = {}
    for fle in example_html_path.iterdir():
        with fle.open("rt") as f:
            name = fle.stem
            sources[name] = {"html": f.read()}

    for key, source in sources.items():
        soup = BeautifulSoup(source["html"], "lxml")
        soup.get_text()
        text = get_math_text(soup)
        # make output more human-friendly
        text = "\n".join(x.strip() for x in text.split("\n") if x.strip() != "")
        p = example_text_path / f"{key}.txt"
        with p.open("wt") as f:
            f.write(text)


if __name__ == "__main__":
    main()
