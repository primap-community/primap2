(development-reference)=
# Development

PRIMAP2 is developed as free software on github, and you are welcome to participate!
To make the development process smooth, we use a couple of tools and standards also
known from other Python projects. In this section, we will describe these tools as well
as the internal structure of the PRIMAP2 library to help you get started.

## Quickstart

Here's how to set up a local environment for `primap2` development and start developing
in a nutshell.

1. **Clone the git repo locally.**
   If you use an IDE like [pycharm](https://www.jetbrains.com/de-de/pycharm/) or
   [Visual Studio Code](https://code.visualstudio.com/), use your IDE to check out
   your fork from github. For pycharm, use `git -> Clone…` or
   `VCS -> Check out from version control`. Alternatively, if you are not using an
   IDE, clone using the git command line:
   ```shell
    $ git clone git@github.com:primap-community/primap2.git
   ```

2. **Create the virtual environment.**
   To separate your environment used for developing PRIMAP2 from your system python,
   create a virtual environment. After cloning, pycharm will automatically offer
   creating a virtual environment, just accept. Alternatively, you can use the command
   line:
    ```shell
   $ cd primap2/
   $ make virtual-environment
    ```

3. **Install pre-commit hooks.**
   For static analysis tools and for enforcing a common code style, we use git hooks
   which are automatically executed before every commit. To install them for yourself,
   execute in the command line:
   ```shell
   $ make install-pre-commit
   ```
   You can immediately try the checks using:
   ```shell
   $ make lint
   ```

4. **Create a branch.**
   It is best practice to do all of your development in git branches, which you can
   easily submit for inclusion into PRIMAP2 later. In pycharm, open the git tab,
   right-click on the local `main` branch and select `New branch from selected…`.
   Alternatively, you can use the git command line:
   ```shell
   $ git checkout -b name-of-your-bugfix-or-feature
   ```
   Now you can make your changes locally.

5. **Run tests and format.**
   Whenever you feel like it during development, and especially before committing, you
   can run the test suite of PRIMAP2. It consists of two parts: static analysis and
   formatting (called linting) and units tests. To run the static analysis and format
   your code, run:
   ```shell
    $ make lint
   ```
   If static analysis finds an error or an inconsistency, this will be highlighted in
   red in the terminal. If the formatting changes your code, it will also be highlighted
   in red. If you are unsure if any action from your side is necessary to pass linting,
   just run `make lint` twice in a row. If the second run highlights anything in red,
   you have to fix it yourself.

   To run the unit tests, run in the terminal:
   ```shell
    $ make test
   ```
   or use the function provided by your IDE for pytest. If any tests fail, this will
   be shown.

6. **Commit your changes.**
   To commit your changes using pycharm, select the `Commit` tab, select the changed
   files, provide a commit message and use `Commit and Push…`. To do the same in the
   terminal, run:
   ```shell
    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature
   ```
   Note that the commit will fail if your code does not pass `make lint`. You have
   to fix all issues listed by `make lint` before committing. You can try committing
   twice in a row to see if the linting could fix all issues by itself or if you have
   to fix something yourself.

7. **Submit a pull request.**
   Visit `primap2` on GitHub and submit your branch as a pull request.

That's it! For more details for each particular topic, keep reading.

## Branches and Pull Requests

We use the [GitHub flow](https://guides.github.com/introduction/flow/) to integrate
changes in PRIMAP2. The basic idea is that "packages" of changes are developed in
branches, and integrated into the main PRIMAP2 branch using pull requests. Since only
members of the primap-community team on GitHub can create branches directly in the PRIMAP2
repository, developers who are not members of the primap-community team should create a fork
and then create branches in their own fork.

Members of the primap-community team can also push directly to the main PRIMAP2 branch, which
should only be used for small self-contained changes.
If in doubt, use a branch and send a pull request.

## Code format

[<img src="https://img.shields.io/badge/code%20style-black-000000.svg">](https://github.com/psf/black)

We use the `black` code format standard (via the [`ruff`](https://docs.astral.sh/ruff/)
linter), which is also enforced by our CI pipeline and pre-commit hooks, so you *will*
use that standard. Don't worry about it, though, it  all happens automatically, just
running `make lint` will apply the standard.
We also follow [PEP8](https://www.python.org/dev/peps/pep-0008/), so use CamelCase for
classes and  lowercase_with_underscores for functions and arguments. "Hide" functions
which are not (yet) meant to be part of the public API using a leading ``_``, etc.

If a part of the code should *not* follow our usual code style (because you are somewhat
dubiously building ASCII art in Python or whatever), use
[the fmt on/off directive](https://github.com/psf/black#the-black-code-style) so ruff
will ignore that part.

We target Python version 3.10 and later, so using
[f-strings](https://docs.python.org/3/tutorial/inputoutput.html#tut-f-strings) is fine
and generally preferable to old-style format strings.

Please use [type annotations](https://realpython.com/lessons/type-hinting/) where
appropriate to facilitate static type checking and state your expectations explicitly
for other developers and users. Please also document your code, see the section below.

## Linting

We use [`pre-commit`](https://pre-commit.com/) to catch smaller and larger errors before
they are committed. All the configured checks and fixes are listed in the
`.pre-commit-config.yaml` file, the most interesting ones in daily development are:

- `check-ast`: parses all python files and errors if the syntax is not valid.
- `check-merge-conflict`: emits an error if it finds unresolved merge conflicts.
- `ruff`: static analysis for unused imports and variables etc.
  Sometimes, it is unavoidable to trigger ruff errors, in that case add a comment of
  the form `# noqa: E501` at the end of the offending line (using the error code that
  ruff reports).
- `ruff format`: source code formatting.

At any time, you can run all the checks using:
```shell
$ make lint
```

Checks are also automatically run when you commit your changes, and the commit is
aborted if errors are found or files are modified so you can review the changes. Since
many problems are fixed automatically, you can run `make lint` twice or retry your
commit and see if everything is fixed automatically already.

If you find additional pre-commit hooks that might be worth to include, simply add them
to `.pre-commit-config.yaml` and submit a pull request.

## Repo structure

In the repository, all code is inside the `primap2/` directory, with the unit tests
all in the `primap2/tests/` directory.
Documentation is mainly in the `docs/` directory, but some documentation which should
be easily accessible directly from the GitHub starting page is also at the top level
(namely, `AUTHORS.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, the ``LICENSE`` and the
`README.md` itself).
Licenses of software included from other projects are in the `licenses/` directory.
Additional configuration files for the Python packaging and assorted tools are
directly at the top level.

In the main `primap2/` directory, the publicly accessible API is defined in the
`__init__.py` and `accessors.py` files.
The main API is provided as an
[`xarray` extension](https://xarray.pydata.org/en/stable/internals.html#extending-xarray).
Using the xarray extension model, we provide "accessor" classes for DataArray and
Dataset and register them with xarray under the `pr` namespace.
For the user, the primap2 functionalities operating on a DataArray or Dataset are then
directly accessible at `ds.pr.name` after importing `primap2`.
These accessor classes are found in `accessors.py`.
In order to separate concerns and keep the code tidy, different
functionality is internally split into different python files using classes.
Therefore, the actual implementation of functions is not done in `accessors.py`, but
in python files with a leading underscore, and the functionality is included into
`accessors.py` using inheritance.

Functions which do not operate on DataArrays or Datasets are also included in the
respective python file which bundles similar functionality, and if they should be part
of the public API are imported in `__init__.py` and included in `__all__` so that
they are available directly at the package level.

Some specialized, optional functionality is bundled together in sub-modules. Currently,
there are two public sub-modules:
- {ref}`primap2.pm2io` provides I/O functions for easily reading data from other formats into
  the primap2 data format.
- {ref}`primap2.csg` contains the Composite Source Generator functionality to combine
  multiple data sources into a single harmonized dataset.


## Adding new functions

To include new functionality, first check if your new function would fit one of the
files that exist already from the intended functionality.
If it does, simply add your function as a method to the corresponding Accessor class
in that file (or as a standalone function if it does not operate on an existing
DataArray or Dataset).
Note that the DataArray or Dataset to be operated on is not passed to the function as
a separate argument, instead it is available as `self._ds` for Datasets or
`self._da` for DataArrays.

If none of the existing "functionality packages" fits your envisioned function, add
a new "functionality package".
To do this, you first need to think of a succinct description of the topic of your
package, a few words only, for example "aggregate", or "data format".
Then, add a new python file `primap2/_my_topic.py` (note the leading underscore)
with the following content:

```python
"""Overall documentation"""
from . import _accessor_base

class DataArrayMyTopicAccessor(_accessor_base.BaseDataArrayAccessor):
  def my_function(self, *, arguments):
      """Does really nice things on a data array."""
      return self._da

class DatasetMyTopicAccessor(_accessor_base.BaseDatasetAccessor):
  def my_function(self, *, arguments):
      """Does really nice things on a data set."""
      return self._ds
```

Replace `MyTopic` in the class names with your chosen topic and
`my_function` with a more descriptive, unique name.
Also provide better overall documentation in the docstring on the first line and
proper function documentation, of course.
If you are only writing functions for either DataArrays or Datasets, you can delete
the other Accessor class.
To include your new package in the public API, import your classes in `accessors.py`
and add them to the definition of the `PRIMAP2DatasetAccessor` and the
`PRIMAP2DataArrayAccessor`.
Afterwards, your functions are accessible after importing `primap2` on any xarray
DataArray or Dataset object as `obj.pr.my_function`.

Ideally, you also add tests for your new functionality, and all tests for the file
`_topic.py` should be included in `tests/test_topic.py`. Also check out the
documentation section below to document your code.

Within methods defined on Accessor classes, you can use any other PRIMAP2 functionality
via `self._ds.pr.other_functio`` just like outside of PRIMAP2.

## Documentation

For documenting RPIMAP2, we use `sphinx` and host the documentation online at
[ReadTheDocs](https://primap2.readthedocs.io/).
Before your changes land in the main PRIMAP2 branch, where ReadTheDocs picks them up,
you can compile the documentation locally using `make docs` in the terminal and
open the `docs/build/html/index.html` file in your web browser.

Static documentation (such as this section) is written directly in
[Markedly Structured Text (MyST)](https://myst-parser.readthedocs.io)
in files in the `docs/source/` directory and included into the documentation by adding the
file to `docs/source/index.md`.
If you have a part of the documentation which is using python examples a lot, it might
be a good idea to write the documentation as a myst notebook instead.
Add the notebook in the `docs/source/` folder or one of its sub-folders as a file with
the `.md` suffix, then add the following at the very top:
```
---
jupytext:
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
```
Now, you can either write it directly in yuor editor, or open it in `jupyter lab`,
right-click it, select "open with -> jupytext notebook" and edit and run it like a
normal jupyter notebook. The notebook will be saved as markdown, which makes it easy to
see changes using simple diffing tools and make quick edits with a simple text editor.
The notebook will be run automatically when compiling the documentation, ensuring that
the output is always up-to-date.

The API documentation, i.e. the documentation of the functions in the `primap2/`
directory, is done automatically using sphinx.
To enable this for your functions, the first step is to document them using docstrings
using the [numpy style](https://numpydoc.readthedocs.io/en/latest/format.html)
in reStructuredText syntax.
PyCharm can help you with that (check below in the pycharm section for how to enable
it), just start typing  three double quotes at the start of a function, and you will get
a template already filled in with all arguments to the function.
Don't hesitate to include a lot of information in your docstring.
Ideally, the function should need no further comments in the main function body to
be understood - simple comments in the function body are not visible in the generated
documentation or the classic `help()` function at the python command line.

## Testing

Adding unit tests for your functions can help uncovering bugs or inconsistencies in the
API.
The more your function is used also by other people and in other downstream functions,
the likelier it is that your function will be used in somewhat unexpected ways and
bugs will be difficult to find. Therefore, tests for these functions are more important
than tests for more ephemeral functions, but every test helps. Consider simply
copy+pasting whatever smoke-testing of your function you are doing during development
to `primap2/tests/test_topic.py` into a function starting with `test_`.
That way, you have a good start for the unit tests of your new function.

Some infrastructure is already provided for tests, in particular you can take a
minimal, opulent, or empty PRIMAP2 Dataset to run your tests on. Check out
`primap2/tests/conftest.py` to see the testing Datasets and look at e.g.
`primap2/tests/test_data_format.py` for some tests using these Datasets.
Each test gets a fresh copy of the example Datasets, so don't worry changing anything
within your test.

## Logging

We use [loguru](https://github.com/Delgan/loguru) for easy and expressive logging.
If you want to report an error to the user, consider to simply `raise` an Exception,
which will interrupt the program flow for the user and thereby certainly alert the
user to the error. If, on the other hand, you just want to warn the user or report
on your progress or emit debugging information, use the logging facilities of
loguru:

```python
from loguru import logger

def my_func(path):
   if not path.exists():
       logger.warning(f'Path {path!r} does not exist, choosing default path')
```

Whenever you feel like introducing some "print" statements, just use `logger.debug`
instead, and save yourself re-introducing print statements whenever you have to start
debugging again.

## Continuous Integration

The linting and testing is automatically performed for all supported Python versions
using github actions for every commit to the main PRIMAP2 branch and for every
pull request.
The exact steps are defined in `.github/workflows/ci.yml`, which basically does
what `make lint` and `make test` do, but for all supported python versions.
You can check out the
[results at github](https://github.com/primap-community/primap2/actions).

Additionally, we also test if everything works using the development versions of central
upstream libraries. This helps us identify problems with updates in libraries we depend
on before they are released. The exact workflow is defined in
`.github/workflows/ci-upstream-dev.yml`.

## Pycharm integration

Developing PRIMAP2 with Pycharm works best if you:

1. Set the development virtual environment as the python
   project interpreter in `File | Settings | Project | Python interpreter` by
   selecting `venv/bin/python` as the Python interpreter.
   This ensures that you use the same python version and packages in Pycharm and e.g.
   when running tests.
2. Generate stub files for xarray which include the PRIMAP2 accessors to get code
   insight including autocompletion for PRIMAP2 functions.
   For this, first run `make stubs` in a terminal, then right click on the stubs
   folder and select `Mark directory as | Sources root`.
   Now restart Pycharm and afterwards you should have helpful tooltips and code
   completion for PRIMAP2 functions.
3. Change the docstring format in
   `File | Settings | Tools | Python integrated tools | Docstrings | Docstring Format`
   to `Numpy`.
4. If you want to run tests in pycharm instead of the terminal using `make test`,
   you can add a configuration at
   `Run | Edit configurations | + | python tests | pytest`.
   Afterwards, you can run the tests by selecting this configuration at the top right
   bar and clicking on the "run" or "run with coverage" icons.
5. If you want to run the `ruff` code formatter from PyCharm, look at the
   [`ruff` plugin](https://plugins.jetbrains.com/plugin/20574-ruff).
6. A couple of plugins can be useful in PyCharm for PRIMAP2 development:

   * [Makefile support](https://plugins.jetbrains.com/plugin/9333-makefile-language)
      to run Makefile targets directly from PyCharm
   * [CSV Plugin](https://plugins.jetbrains.com/plugin/10037-csv-plugin)
      to view and edit CSV files
   * [Matlab support](https://plugins.jetbrains.com/plugin/10941-matlab-support)
      to quickly view .m files without starting matlab
   * [Toml](https://plugins.jetbrains.com/plugin/8195-toml)
      for editing pyproject.toml
   * [.ignore](https://plugins.jetbrains.com/plugin/7495--ignore)
      for better support of `.gitignore` files

## Deploying

A reminder for the maintainers on how to deploy.

1.  Commit all your changes.
2.  Run `tbump X.Y.Z`.
3.  Wait a bit that the release on github and zenodo is created.
4.  Run `make README.md` to update the citation information in the README from the
    zenodo API. Check if the version is actually correct, otherwise grab a tea and
    wait a little more for zenodo to mint the new version. Once it worked, commit the
    change.
5.  Upload the release to pyPI: `make release`
