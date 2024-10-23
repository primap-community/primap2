---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---
# Log messages

Many primap2 functions emit log messages, which have an associated severity.
The severities we use are shown in the table.

| severity | used for                                                                         | default |
|----------|----------------------------------------------------------------------------------|---------|
| debug    | useful for understanding what functions do internally                            | ✗       |
| info     | noteworthy information during normal processing                                  | ✓       |
| warning  | problems which are not necessarily fatal, but should be acknowledged by the user | ✓       |
| error    | problems which need to be solved by the user                                     | ✓       |

As noted, by default `debug` messages are not shown, all other messages are shown.

## Changing what is shown

As said, by default `debug` messages are not shown, as you can see here:

```{code-cell} ipython3
import primap2
import sys

from loguru import logger

logger.debug("This message will not be shown")
logger.info("This message will be shown")
```

To change this, remove the standard logger and add a new logger:

```{code-cell} ipython3
logger.remove()
logger.add(sys.stderr, level="DEBUG")

logger.debug("Now you see debug messages")
logger.info("You still also see info messages")
```

Instead of showing more, you can also show less:

```{code-cell} ipython3
logger.remove()
logger.add(sys.stderr, level="WARNING")

logger.debug("You don't see debug messages")
logger.info("You also don't see info messages")
logger.warning("But you do see all warnings")
```

## Advanced usage

It is also possible to log to a file or add more information to the logs. See the
[loguru documentation](https://loguru.readthedocs.io/) for details.
