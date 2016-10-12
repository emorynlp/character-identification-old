from const import StringConst

SPEAKER_FEAT_KEY = "speaker"


class SceneReader:
    def __init__(self, form=3, lemma=4, pos=5, nament=10, feats=6, dhead=7, deprel=8, sheads=9, utterance_id=0, sentence_id=1, referant=11):
        self.form = form
        self.lemma = lemma
        self.pos = pos
        self.nament = nament
        self.feats = feats
        self.dhead = dhead
        self.deprel = deprel
        self.sheads = sheads
        self.utterance_id = utterance_id
        self.sentence_id = sentence_id
        self.referant = referant

    def __get_value(self, values, idx, tag):
        if idx < 0 or len(values) <= idx:
            return None

        string = values[idx]
        return None if tag and StringConst.BLANK == string else string

    def __gen_feat_map(self, feat_string):
        m_feat = dict()
        pairs = feat_string.split(StringConst.PIPE)

        for pair in pairs:
            p = pair.split(StringConst.EQUAL)
            m_feat[p[0]] = p[1]

        return m_feat
