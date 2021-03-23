import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import sub_sequence
from DeepModel import OneHotDeepModel, ChemicalDeepModel


class SpectrumEncoder:
    def __init__(self):
        pass
    def encode(self, data, seq_len):
        with tqdm(total=len(data)) as pbar:
            pbar.set_description("Extracting Spectrum Feature:")
            def spectrum(seq,seq_len):
                seq = sub_sequence(seq,seq_len)
                nucleotide = ['A', 'G', 'C', 'T']
                npse = {}
                for d in range(4):
                    for start in nucleotide:
                        for end in nucleotide:
                            pattern = start + str(d) + end
                            npse[pattern] = 0

                for d in range(4):
                    for i in range(len(seq) - d - 1):
                        npse[seq[i] + str(d) + seq[i + d + 1]] += 1 / (len(seq) - d - 1)

                pbar.update(1)
                return np.array(list(npse.values()))

            return np.array(list(map(lambda seq:spectrum(seq,seq_len), data)))


class GraphEncoder:
    def __init__(self):
        pass
    def encode(self,data,seq_len):
        with tqdm(total=len(data)) as pbar:
            pbar.set_description("Extracting Graph Feature:")
            def graph(seq,seq_len):
                seq = sub_sequence(seq,seq_len)
                w = int(len(seq) / 2)
                dinuIndex = {'AA': 0, 'AT': 1, 'AG': 2, 'AC': 3,
                             'TA': 4, 'TT': 5, 'TG': 6, 'TC': 7,
                             'GA': 8, 'GT': 9, 'GG': 10, 'GC': 11,
                             'CA': 12, 'CT': 13, 'CG': 14, 'CC': 15}
                leftGraph = np.zeros([16 * 16], dtype=float)
                rightGraph = np.zeros([16 * 16], dtype=float)
                for i in range(w - 3):
                    frontNode = seq[i:i + 2]
                    backNode = seq[i + 2:i + 4]
                    row = dinuIndex[frontNode]
                    col = dinuIndex[backNode]
                    leftGraph[16 * row + col] += 1

                    frontNode = seq[i + w + 1:i + w + 3]
                    backNode = seq[i + w + 3:i + w + 5]
                    row = dinuIndex[frontNode]
                    col = dinuIndex[backNode]
                    rightGraph[16 * row + col] += 1

                leftIntense = np.where(leftGraph != 0)[0].size / len(leftGraph)
                rightIntense = np.where(rightGraph != 0)[0].size / len(rightGraph)

                dge = [leftIntense]
                dge.extend(list(leftGraph))
                dge.append(rightIntense)
                dge.extend(list(rightGraph))

                pbar.update(1)
                return np.array(dge)

            return np.array(list(map(lambda seq:graph(seq,seq_len), data)))


class DecimalEncoder:
    def __init__(self):
        pass
    def encode(self,data,seq_len):
        with tqdm(total=len((data))) as pbar:
            pbar.set_description("Extracting Decimal Feature:")
            def decimal(seq,seq_len):
                seq = sub_sequence(seq,seq_len)
                dde = []
                decimalEncoding = {'AA': 0, 'AG': 1, 'AC': 2, 'AT':3,
                                   'GA': 4, 'GG': 5, 'GC': 6, 'GT': 7,
                                   'CA': 8, 'CG': 9, 'CC': 10, 'CT': 11,
                                   'TA': 12, 'TG': 13, 'TC': 14, 'TT': 15}
                for i in range(len(seq)-1):
                    dde.append(decimalEncoding[seq[i]+seq[i+1]])

                pbar.update(1)
                return np.array(dde)

            return np.array(list(map(lambda seq:decimal(seq,seq_len), data)))


class OneHotEncoder:
    def __init__(self):
        pass
    def encode(self, data, seq_len):
        with tqdm(total=len(data)) as pbar:
            pbar.set_description("One-hot Encoding:")

            def onehot(seq, seq_len):
                seq = sub_sequence(seq, seq_len)
                encoder = {'A': [1, 0, 0, 0], 'G': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
                encoding = []
                for nu in seq:
                    encoding.append(encoder[nu])

                pbar.update(1)
                return encoding

            return np.array(list(map(lambda seq: onehot(seq, seq_len), data))).reshape((-1, seq_len * 4)).astype(float)


class ChemicalEncoder:
    def __init__(self):
        pass
    def encode(self,data,seq_len):
        with tqdm(total=len(data)) as pbar:
            pbar.set_description("Chemical Encoding:")

            def chemical(seq, seq_len):
                seq = sub_sequence(seq,seq_len)
                encoder = {'A': [1, 1, 1], 'G': [1, 0, 0], 'C': [0, 1, 0], 'T': [0, 0, 1]}
                encoding = []
                for nu in seq:
                    encoding.append(encoder[nu])
                pbar.update(1)
                return encoding

            return np.array(list(map(lambda seq: chemical(seq, seq_len), data))).reshape((-1, seq_len * 3)).astype(
                float)


class DeepOneHotEncoder:
    def __init__(self, model_dir, n_models=5):
        self.model_dir = model_dir
        self.n_models = n_models

    def encode(self, data, seq_len):
        onehot = OneHotEncoder()
        data = onehot.encode(data,seq_len).reshape((-1,101,4,1)).astype(float)
        encoding = np.array([])
        n_models = self.n_models
        batch_num = 20
        batch_size = int(len(data)/batch_num)

        with tqdm(total=n_models*batch_num) as pbar:
            pbar.set_description("Extracting OneHot Deep Feature:")

            for i in range(n_models):
                model = OneHotDeepModel()
                model.load_weights(f"{self.model_dir}/model{i+1}").expect_partial()
                each_model_encoding = model.encode(data[:batch_size]).numpy()
                pbar.update(1)

                for j in range(1,batch_num-1):
                    each_model_encoding = np.vstack((each_model_encoding,model.encode(data[j*batch_size:(j+1)*batch_size]).numpy()))
                    pbar.update(1)

                each_model_encoding = np.vstack((each_model_encoding,model.encode(data[(batch_num-1)*batch_size:]).numpy()))
                pbar.update(1)

                if i == 0:
                    encoding = each_model_encoding
                else:
                    encoding = np.hstack((encoding,each_model_encoding))

        return encoding


class DeepChemicalEncoder:
    def __init__(self, model_dir, n_models=5):
        self.model_dir = model_dir
        self.n_models = n_models

    def encode(self,data,seq_len):
        chemical = ChemicalEncoder()
        data = chemical.encode(data,seq_len).reshape((-1,101,3,1)).astype(float)
        encoding = np.array([])
        n_models = self.n_models
        batch_num = 20
        batch_size = int(len(data)/batch_num)

        with tqdm(total=n_models*batch_num) as pbar:
            pbar.set_description("Extracting Chemical Deep Feature:")

            for i in range(n_models):
                model = ChemicalDeepModel()
                model.load_weights(f"{self.model_dir}/model{i+1}").expect_partial()
                each_model_encoding = model.encode(data[:batch_size]).numpy()
                pbar.update(1)

                for j in range(1,batch_num-1):
                    each_model_encoding = np.vstack((each_model_encoding,model.encode(data[j*batch_size:(j+1)*batch_size]).numpy()))
                    pbar.update(1)

                each_model_encoding = np.vstack((each_model_encoding,model.encode(data[(batch_num-1)*batch_size:]).numpy()))
                pbar.update(1)

                if i == 0:
                    encoding = each_model_encoding
                else:
                    encoding = np.hstack((encoding,each_model_encoding))

        return encoding


def extract_features_from_raw_data(dataset_file, params):
    df = pd.read_excel(dataset_file)
    data = df.seq.to_numpy()
    labels = df.label.to_numpy()
    features = []

    # spectrum feature
    spectrum = SpectrumEncoder()
    features.append(spectrum.encode(data, params.spectrum_seq_len))

    # decimal feature
    decimal = DecimalEncoder()
    features.append(decimal.encode(data, params.decimal_seq_len))

    # graph feature
    graph = GraphEncoder()
    features.append(graph.encode(data, params.graph_seq_len))

    # deep one-hot feature
    deep_one_hot = DeepOneHotEncoder(params.deep_onehot_model_dir)
    features.append(deep_one_hot.encode(data, params.onehot_seq_len))

    # deep chemical feature
    deep_chemical = DeepChemicalEncoder(params.deep_chemical_model_dir)
    features.append(deep_chemical.encode(data, params.chemical_seq_len))

    return features, labels