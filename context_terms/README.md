# Extracting Context Terms from Text

This part of the project focuses on extracting context terms from text. Context terms include cell lines, cell types and tissues. Below are some examples of contexts (highlighted in red) alongside gene regulation relations.

![Examples of context terms](https://github.com/BenevolentAI/Stanford-Collab/raw/annotation/context_terms/examplecontexts.png)

The [generateCellContextsList.py](https://github.com/BenevolentAI/Stanford-Collab/blob/annotation/context_terms/generateCellContextsList.py) script makes various SPARQL queries to WikiData to get lists of cell lines, cell types and tissue types along with synonyms.

The [extractContextTerms.py](https://github.com/BenevolentAI/Stanford-Collab/blob/annotation/context_terms/extractContextTerms.py) script uses the lists of contexts generated by the other script and finds them in text. It takes a BioC file as input, finds mentions of the context words and outputs another BioC file with the annotations added.