Literature Context (lit_contextizer) Package README
===================================================
A package for representing contextual features (e.g. cell type, species) from literature and detecting
context-relationship associations

Installation
------------
To download this code and install in development mode, do the following:

.. code-block::

    $ git clone https://github.com/BenevolentAI/Stanford-Collab
    $ pip install -e .

..
    Testing |build| |coverage|
    --------------------------

Testing
-------

To test this code, please use ``tox``:

.. code-block::

    $ pip install tox
    $ tox

Note that ``tox`` is configured to automate running tests and checking test coverage, checking ``pyroma`` compliance,
checking ``flake8`` compliance, checking ``doc8`` compliance (for ``.rst`` files), enforcing README style guides, and
building ``sphinx`` documentation.

..
    Documentation |documentation|
    -----------------------------

Documentation
_____________

Running ``tox`` above should automatically build the ``readTheDocs``-style ``sphinx`` documentation, however this can
also be accomplished by running the following:

.. code-block::

    $ cd docs
    $ make html
    $ open build/html/index.html

Usage
-----
This package is currently set up so that the training of the BERT model can be easily run as a package using a
command-line interface as follows:

.. code-block::

    $ # Make sure that installation was successful as described above
    $
    $ python -m lit_contextizer

..
    .. |build| image:: https://travis-ci.com/BenevolentAI/Stanford-Collab.svg?branch=master
        :target: https://travis-ci.com/BenevolentAI/Stanford-Collab
        :alt: Build Status

..
    .. |coverage| image:: https://codecov.io/gh/CoronaWhy/drug-lit-contradictory-claims/branch/master/graph/badge.svg
          :target: https://codecov.io/gh/CoronaWhy/drug-lit-contradictory-claims

..
    .. |documentation| image:: https://readthedocs.org/projects/drug-lit-contradictory-claims/badge/?version=latest
      :target: https://drug-lit-contradictory-claims.readthedocs.io/en/latest/?badge=latest
      :alt: Documentation Status
