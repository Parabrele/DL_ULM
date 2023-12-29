from datasets import load_dataset

# TODO : check if "test" is the right split for all datasets

"""
CCS datasets :
"""

"""
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

CC_DATASETS = [
    "imdb test",
    "amazon_polarity test",
    "ag_news test",
    "dbpedia_14 test",
    "yangwang825/rte test",
    "SetFit/qnli test",
    "story_cloze test",
    "boolq validation",
    "piqa test",
]

"""
Geometry of truth datasets :

cities  Locations of world cities
sp en trans     Spanish-English translations
neg cities      Negations of statments in cities
neg sp en trans Negations of statements in sp en tran
larger than     Numerical comparisons: larger than
smaller than    Numerical comparisons: smaller than
cities cities conj  Conjunctions of two statements in cities
cities cities disj  Disjunctions of two statements in cities
companies true false    Claims about companies; from Azaria & Mitchell (2023)
common claim true false Various claims; from Casper et al. (2023b)
counterfact true false  Various factual recall claims; from Meng et al. (2022)
likely Nonfactual text with likely or unlikely final tokens
"""

# TODO : clément passe ton code

"""
SAE general training datasets :
"""

SAE_DATASETS = [
    "bookcorpus train[:1%]", # 74M * 0.01 = 740k
    "wikimedia/wikipedia train[:10%]", # 6.41M * 0.1 = 641k
]

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