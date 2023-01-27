Literature Context (lit_contextizer) Package README
===================================================
A package for extracting protein-protein relations and biological contexts (e.g. cell type,
tissues) from full scientific biomedical text then detecting instances where the context is qualifying the relationship.

Installation
------------
First navigate to the directory where you would like this package to live and clone this repo. Then install this as a local package with pip.

.. code-block::

    $ cd {HOME}
    $ git clone https://github.com/dnsosa/lit_contextizer.git
    $ cd lit_contextizer
    $ pip install -e .


Usage
-----
In this section, we will walk you through all the steps for data pre-processing and analysis.


Data Processing
_______________

Option 1: Loading pre-processed data from SimTK
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you'd like to go straight to analysis/experiments, you can download the pre-processed datasets from the affiliated `lit_con SimTK project page <https://simtk.org/projects/lit_con>`_ .

.. code-block::

    $ # Make sure you're in the lit_contextizer dir you cloned
    $ mkdir output
    $
    $ # Download preprocessed data from SimTK
    $ wget https://simtk.org/docman/view.php/2474/14118/simtk_context_ppi_output_files.zip -P output
    $ unzip output/simtk_context_ppi_output_files.zip
    $ mv output/simtk_context_ppi_output_files/* output/

Option 2: Pre-process data from scratch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you'd like to process the data from scratch all the way through extracting features, we will describe the data processing pipeline as a series of steps (processes) taking inputs and yielding outputs.


**Step 1: Extracting all relationships from PubMed full text articles using dependency parsing**

To build the relation data from scratch, you will need to use the `Biotext project <https://github.com/jakelever/biotext>`_ to download PubMed/PMC, convert to BioC files and match entity mentions against data from PubTator. Follow the instructions on the `Biotext page <https://github.com/jakelever/biotext>`_ to create the PubTator aligned files. You should then complete the BIOTEXT and CORES variable in the shell commands below.

Input: PubMed and PMC files converted to BioC XML files and aligned with PubTator entity extractions

Process:

.. code-block::

    $ export BIOTEXT=/path/to/pubtator/bioc/files
    $ export CORES=1
    $ cd metadata && sh gather_metadata.sh
    $ cd relation_extraction
    $ MODE=full BIOTEXT=$BIOTEXT snakemake --cores $CORES

Output: ``relation_extraction/working/with_tidy_citation_distances/*.bioc.xml`` (BioC XML files with PPI relations extracted and context words extracted.)

In subsequent steps, processes are executed via the command-line interface (CLI) provided in the ``lit_contextizer`` package.

**Step 2: Identifying contexts mentioned in papers containing one of the extracted relations**

Input: Directory ``relation_extraction/working/with_tidy_citation_distances`` containing the annotated full texts

Process:

.. code-block::

    $ python -m lit_contextizer --output_folder output --full_text_dir relation_extraction/working/with_tidy_citation_distances --parse_files --dump_annots_context

Output: ``output/pmid2contexts.csv``, ``output/contexts2pmid.csv``


**Step 3: Create the Insider corpora and extract features from Insider documents**
Repeat this process for each of ``{CTs, tissues, combined}``.

Input: ``output/pmid2contexts.csv``, ``output/contexts2pmid.csv``, ``output/all_pubmed_relations_df``, ``relation_extraction/working/with_tidy_citation_distances``

Process:

.. code-block::

    $ python -m lit_contextizer --no_parse_files --insider_context_type {CTs, tissues, combined} --output_folder output --full_text_dir relation_extraction/working/with_tidy_citation_distances

Output: ``output/{CTs, tissues, combined}_insider_papers_features_df.tsv``


**Step 4: Download GIANT PPIs**

Input: None

Process:

.. code-block::

    $ wget -P input/GIANT_PPIs "https://s3-us-west-2.amazonaws.com/humanbase/networks/adipose_tissue_top.gz"
    $ wget -P input/GIANT_PPIs "https://s3-us-west-2.amazonaws.com/humanbase/networks/liver_top.gz"
    $ wget -P input/GIANT_PPIs "https://s3-us-west-2.amazonaws.com/humanbase/networks/lung_top.gz"
    $ cd input/GIANT_PPIs
    $ gunzip *

Output: ``input/GIANT_PPIs``


**Step 5: Extract features from papers containing one of our extracted relations that describes a PPI found in one of the retrieved GIANT networks.**

Input: ``input/GIANT_PPIs``

Process:

.. code-block::

    $ python -m lit_contextizer --no_parse_files --networks_folder input/GIANT_PPIs --paper_subset giant --output_folder output

Output: ``adipose_tissue_ppi_pmc_df_conf0.0.csv``, ``liver_ppi_pmc_df_conf0.0.csv``, ``lung_ppi_pmc_df_conf0.0.csv``, ``ppi_pmc_tissues_features_df_conf0.0.csv``


**Step 6: Extract features from papers containing one of our extracted relations that describes one of our input PPI pairs.**

Input: ``dengue_ppi_paras.csv``

.. code-block::

    $ python -m lit_contextizer --no_parse_files --networks_folder input/GIANT_PPIs --paper_subset giant --output_folder output

Output: ``dengue_papers_features_df.tsv``



Analysis
________

Step-by-step runthrough of analyses can be found in the provdied notebook, ``notebooks/Literature Contextizer Analyses Notebook.ipynb``


Testing
-------

To test this code, use ``tox``:

.. code-block::

    $ pip install tox
    $ tox

We have configured ``tox`` to check a) test coverage, b) ``pyroma`` compliance for package metadata, c) ``flake8`` compliance for PEP-8 code style, d) ``doc8`` compliance for ``.rst`` files, e) README style guidelines, and f) ``sphinx`` documentation builds.


Documentation
_____________

Running ``tox`` above should automatically build the ``readTheDocs``-style ``sphinx`` documentation, however this can
also be accomplished by running the following:

.. code-block::

    $ cd docs
    $ make html
    $ open build/html/index.html


