# Extraction of Regulation Relations

This part of the project focuses on extracting sentences that mention gene regulation events.

This uses the BioText project as input. It downloads and converts the biomedical literature (PubMed & accessible PubMed Central) to the BioC format. We use the BioText project's ability to get PubTator Central annotations too.

The pipeline does the following steps. It is managed using Snakemake.
- parseForDepPaths.py: Find sentences that contain at least two gene/proteins and extracts the augmented dependency path between each pair.
- splitData.py: Aggregates the dependency paths for lots of files and groups by entity type. We actually only use the Gene-Gene set here.
- makeMatrix.py: Create a sparse matrix of dependency path & gene ID pairs.
- calcEmbeddings.py: Created dense vector embedding representations for all dependency paths using a low-rank SVD approximation the above matrix
- countDepPaths.py: Count the number of sentences that contain each dependency path

With all this information, the annotateDepPaths.py script can be used to annotate popular dependency paths. And then with these annotations, the filterForWantedDepPaths.py script is used to filter BioC documents to find relations with the annotated dependency paths.

The annotations file (annotations.Gene\_Gene.tsv) contains the set of hundreds of augmented dependency paths with their annotations. The first column contains the annotations. For speedy purposes, 'a' denotes upregulation or 'r' denotes downregulation. Other annotations mean no association or unclear.

Fifty documents were annotated for context and can be found in fifty_docs.bioc.xml.gz.