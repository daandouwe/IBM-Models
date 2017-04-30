"""
This module helps you compute AER for validation set after each iteration of EM.
Check the test function below for an example of how to use these helper functions.

For test results, please use the official AER perl script.
"""

def read_naacl_alignments(path):
    """
    Read NAACL-formatted alignment files.

    :param path: path to file
    :return: a list of pairs [sure set, possible set]
        each entry in the set maps an input position to an output position
        sentences start from 1 and a NULL token is indicated with position 0
    """
    with open(path) as fi:
        ainfo = {}
        for i, line in enumerate(fi.readlines()):
            fields = line.split()
            if not fields:
                continue
            sure = True  # by default we assumed Sure links
            prob = 1.0  # by default we assume prob 1.0
            if len(fields) < 3:
                raise ValueError('Missing required fields in line %d: %s' % (i, line.strip()))
            snt_id, x, y = int(fields[0]), int(fields[1]), int(fields[2])
            if len(fields) == 5:
                sure = fields[3] == 'S'
                prob = float(fields[4])
            if len(fields) == 4:
                if fields[3] in {'S', 'P'}:
                    sure = fields[3] == 'S'
                else:
                    prob = float(fields[3])
            snt_info = ainfo.get(snt_id, None)
            if snt_info is None:
                snt_info = [set(), set()]  # S and P sets
                ainfo[snt_id] = snt_info
            if sure:  # Note that S links are also P links: http://dl.acm.org/citation.cfm?id=992810
                snt_info[0].add((x, y))
                snt_info[1].add((x, y))
            else:
                snt_info[1].add((x, y))
    return tuple(v for k, v in sorted(ainfo.items(), key=lambda pair: pair[0]))


class AERSufficientStatistics:
    """
    Object used to compute AER for a corpus.
    """

    def __init__(self):
        self.a_and_s = 0
        self.a_and_p = 0
        self.a = 0
        self.s = 0

    def __str__(self):
        return '%s/%s/%s/%s %s' % (self.a_and_s, self.a_and_p, self.a, self.s, self.aer())

    def update(self, sure, probable, predicted):
        """
        Update AER sufficient statistics for a set of predicted links given goldstandard information.

        :param sure: set of sure links 
            a links is a tuple of 1-based positions (from, to) where 0 is reserved for NULL
        :param probable: set of probable links (must incude sure links)
        :param predicted: set of predicted links
        """
        self.a_and_s += len(predicted & sure)
        self.a_and_p += len(predicted & probable)
        self.a += len(predicted)
        self.s += len(sure)

    def aer(self):
        """Return alignment error rate: 1 - (|A & S| + |A & P|)/(|A| + |S|)"""
        return 1 - (self.a_and_s + self.a_and_p) / (self.a + self.s)


def test(path, prediction_path):
    # 1. Read in gold alignments
    gold_sets = read_naacl_alignments(path)

    # 2. Read in predictions
    predictions = read_naacl_alignments(prediction_path)

    # 3. Compute AER

    # first we get an object that manages sufficient statistics 
    metric = AERSufficientStatistics()
    # then we iterate over the corpus 
    for gold, pred in zip(gold_sets, predictions):
        metric.update(sure=gold[0], probable=gold[1], predicted=pred[0])
    # AER
    return metric.aer()

