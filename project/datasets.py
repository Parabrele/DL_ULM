"""
file for loading datasets, and testing generalization
"""

"""
CCS datasets :

sentiment classification :
    IMDB (Maas et al., 2011),
    Amazon (McAuley & Leskovec, 2013)

topic classification :
    AG-News (Zhang et al., 2015),
    DBpedia-14 (Lehmann et al., 2015)

NLI :
    RTE (Wang et al., 2018),
    QNLI (Rajpurkar et al., 2016))

story completion :
    COPA (Roemmele et al., 2011)
    Story-Cloze (Mostafazadeh et al., 2017))

question answering :
    BoolQ (Clark et al., 2019)

common sense reasoning :
    PIQA (Bisk et al., 2020).
"""

"""
Geometry of truth datasets :

Name Topic Rows
cities Locations of world cities 1496
sp en trans Spanish-English translation 354
neg cities Negations of statments in cities 1496
neg sp en trans Negations of statements in sp en trans 354
larger than Numerical comparisons: larger than 1980
smaller than Numerical comparisons: smaller than 1980
cities cities conj Conjunctions of two statements in cities 1500
cities cities disj Disjunctions of two statements in cities 1500
companies true false Claims about companies; from Azaria & Mitchell (2023) 1200
common claim true false Various claims; from Casper et al. (2023b) 4450
counterfact true false Various factual recall claims; from Meng et al. (2022) 31960
likely Nonfactual text with likely or unlikely final tokens 10000

"""

def test_generalisation(activation_dataset_list, probe):
    """
    test generalisation of probe to a collection of datasets
    """
    res = []
    for dataset in activation_dataset_list:
        x, y = dataset.x, dataset.y
        pred = (probe(x) > 0).float()
        acc = (pred == y).float().mean()
        res.append(acc)
    return res