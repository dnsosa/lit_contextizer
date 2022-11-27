import kindred

corpus = kindred.Corpus()

#txt1 = '<gene conceptid="1">EGFR</gene>-mediated downregulation of <gene>P53</gene>.'
#txt2 = 'It completely prevented the inhibition of <gene>hoxd4</gene> expression induced by <gene>miR-10a</gene>.'
#txt2 = 'Inhibition of <gene>P53</gene> downregulates <gene>BRCA1</gene>.'

txt1 = 'We did not find that <gene>EGFR</gene> inhibited <gene>P53</gene>.'
#txt2 = "<gene>EGFR</gene> doesn't downregulate <gene>P53</gene>."

#txt1 = 'Decreased <gene>EGFR</gene> did not reduce <gene>P53</gene> expression.'
#txt1 = "<gene>HBx</gene> downregulates PDCD4 expression at least partially through <gene>miR-21</gene>."
txt1 = 'Interestingly, as <Gene id="1">JWA</Gene> upregulated <Gene id="2">XRCC1</Gene> expression in normal cells, <Gene id="3">JWA</Gene> downregulated <Gene id="4">XRCC1</Gene> expression through promoting the degradation of XRCC1 in cisplatin-resistant <Disease id="5">gastric cancer</Disease> cells. <relation type="upregulate" gene1="1" gene2="2" /><relation type="downregulate" gene1="3" gene2="4" />'

corpus.addDocument(kindred.Document(txt1,metadata={'pmid':25476899},loadFromSimpleTag=True))

mapping = {'JWA':'10550','XRCC1':'7515','gastric cancer':'MESH:D013274'}
for e in corpus.documents[0].entities:
	if e.text in mapping:
		e.metadata['conceptid'] = mapping[e.text]
#corpus.addDocument(kindred.Document(txt2,metadata={'pmid':67890},loadFromSimpleTag=True))

kindred.save(corpus, 'biocxml', 'demodocs.bioc.xml')

print("Done.")

