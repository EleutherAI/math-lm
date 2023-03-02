import os
import sys
from bs4 import BeautifulSoup
from bs4.element import *


def _all_math_strings(element, strip=False,):
    """Yield all strings of certain classes, possibly stripping them.

    This makes it easy for NavigableString to implement methods
    like get_text() as conveniences, creating a consistent
    text-extraction API across all PageElements.

    :param strip: If True, all strings will be stripped before being
        yielded.

    :param types: A tuple of NavigableString subclasses. If this
        NavigableString isn't one of those subclasses, the
        sequence will be empty. By default, the subclasses
        considered are NavigableString and CData objects. That
        means no comments, processing instructions, etc.

    :yield: A sequence that either contains this string, or is empty.

    """
    types = (NavigableString, CData)

    # Do nothing if the caller is looking for specific types of
    # string, and we're of a different type.
    #
    # We check specific types instead of using isinstance(self,
    # types) because all of these classes subclass
    # NavigableString. Anyone who's using this feature probably
    # wants generic NavigableStrings but not other stuff.
    my_type = type(element)
    if types is not None:
        if isinstance(types, type):
            # Looking for a single type.
            if my_type is not types:
                return
        elif my_type not in types:
            # Looking for one of a list of types.
            return

    value = element
    if strip:
        value = value.strip()
    if len(value) > 0:
        yield value

def get_math_text(element, separator="", strip=False,):
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
    return separator.join([s for s in _all_math_strings(element,
                strip)])

def main(): 
    """main method, for demoing `get_text()` on some examples. 
    """
    example_path = "example_html/"
    sources = {}
    for fle in os.listdir(example_path):
        with open(os.path.join(example_path, fle)) as f:
            name = fle[:fle.index(".")]
            sources[name] = {"html": f.read()}

    for key in sources: 
        soup = BeautifulSoup(sources[key]["html"], "lxml")
        text = get_math_text(soup)
        print(text)
        with open(os.path.join("example_text", key + ".txt"), "w") as f:
            f.write(text)

if __name__=="__main__":
    main()
