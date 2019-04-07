import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import MeCab
from gensim.models import Doc2Vec

# Japanese Font Setting
plt.style.use("ggplot")
font = {"family":"IPAexGothic"}
mpl.rc('font', **font)


def cos_sim(v1, v2) -> float:
    """Cosine similarity"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


class LabelEstimator():
    """Pretrained-Embedding-Model-based Estimation of Document Label"""
    def __init__(self, d2v_model_path: str, mecab_dict_path: str):
        print("Loading Doc2Vec model...")
        self.d2v = Doc2Vec.load(d2v_model_path)
        self.wv = self.d2v.wv
        print("Done!")

        if mecab_dict_path is None:
            self.mecab = MeCab.Tagger('-Owakati')
        else:
            self.mecab = MeCab.Tagger(f'-Owakati -d {mecab_dict_path}')

        self.parse = self.mecab.parse
        self.labels = None
        return

    def set_labels(self, labels: list) -> None:
        self.labels = labels
        # mapping of label -> vector
        self.label_vec = {w:self._word2vec_concat([w]) for w in labels}
        return

    def clear_labels(self) -> None:
        self.labels = None
        self.label_vec = {}
        return

    def vectorize(self, sentence: list, threshold=3) -> "np.ndarray":
        # 単語数が閾値以下の場合は文書ベクトルの推定ではなく単語ベクトルの加算を行う
        if len(sentence) > threshold:
            return self.d2v.infer_vector(sentence)
        else:
            return self._word2vec_concat(sentence)

    def estimate(self, doc: str, show=True):
        """
        `doc` : str
            A document/sentence text to estimate labels.

        `show` : bool : default=True
            show result in the form of pandas dataframe / plot graph
        """
        assert type(doc) == str, f"input should be string not {type(doc)}"
        sentence = self.parse(doc).rstrip().split()

        v = self.vectorize(sentence)

        # `result`=tupleのlist, [(label_n, cossim)...]
        if self.labels is None:
            result = self.d2v.similar_by_vector(v)
        else:
            result = [(k, cos_sim(v, lv)) for k, lv in self.label_vec.items()]
            result = sorted(result, reverse=True, key=lambda x:x[1])
        # show data or not
        if show:
            return self._show_result(result)
        else:
            return result

    def _word2vec_concat(self, word_list: list) -> "np.ndarray":
        # out-of-vocab 対策で Doc2Vec.infer_vector() を用いる
        vectors = np.array([self.d2v.wv.get_vector(w)
                            if w in self.d2v.wv.vocab
                            else self.d2v.infer_vector([w])
                            for w in word_list])
        v = np.sum(vectors, axis=0)
        return v

    def _show_result(self, result: list) -> str:
        df = pd.DataFrame(result)
        df.columns = ["Label", "Score (cos-sim)"]
        df.set_index("Label", inplace=True)
        df.plot(kind="barh")
        plt.show()
        return df
