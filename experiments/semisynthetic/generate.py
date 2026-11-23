import json
from pathlib import Path

from scipy.special import softmax
import numpy as np
import pickle


def _split_labels(n_patients, seed, train_fraction=0.70, validation_fraction=0.10):
    train_count = int(n_patients * train_fraction)
    validation_count = int(n_patients * validation_fraction)
    if min(train_count, validation_count, n_patients - train_count - validation_count) <= 0:
        raise ValueError('Shared splitting requires non-empty train, validation, and test sets')
    labels = np.empty(n_patients, dtype=object)
    permutation = np.random.default_rng(seed).permutation(n_patients)
    validation_end = train_count + validation_count
    labels[permutation[:train_count]] = 'train'
    labels[permutation[train_count:validation_end]] = 'validation'
    labels[permutation[validation_end:]] = 'test'
    return [str(label) for label in labels]


def simulate_mimic(
    data_dir,
    dataset_name,
    output,
    seed=None,
    header_path=None,
    truth_output=None,
    split_output=None,
    split_seed=62042,
):
    data_path = Path(data_dir)
    output_path = Path(output)
    if seed is not None:
        np.random.seed(seed)

    with (data_path / f'{dataset_name}.pkl').open('rb') as handle:
        data = pickle.load(handle)
    resolved_header = Path(header_path) if header_path else data_path / 'header.pkl'
    with resolved_header.open('rb') as handle:
        features = pickle.load(handle)
    demos = ['agegroup', 'heightgroup', 'weightgroup', 'gender']

    n_features = len(features) + len(demos)
    s = np.random.multivariate_normal(np.zeros(shape=(n_features)), 0.1 * np.eye(n_features))
    m = np.random.normal(0, 0.1)

    cov = np.random.uniform(low=-1, high=1, size=(n_features + 1, n_features + 1))
    w_cov = 0.1 * (np.dot(cov, cov.T))
    mu1 = np.zeros(shape=(n_features + 1))
    mu0 = np.ones(shape=(n_features + 1)) * 0.1
    w1 = np.random.multivariate_normal(mean=mu1, cov=w_cov, size=(1,)).T
    w0 = np.random.multivariate_normal(mean=mu0, cov=w_cov, size=(1,)).T

    data_sim = {patient_id: dict(datum) for patient_id, datum in data.items()}
    truth_patients = []

    for id, datum in data.items():
        x = []
        for val in features:
            x.append(datum[val])
        x = np.array(x).T

        x_demo = []
        for demo in demos:
            x_demo.append(datum[demo])
        x_demo = np.array(x_demo)

        x_all, t, y_f, y_cf, y0, y1 = [], [], [], [], [], []
        timestamps = len(x)
        for ts in range(timestamps):
            x_cur = np.concatenate((x[ts], x_demo))

            if ts > 0:
                p = min(ts + 1, 6)
                weights = np.arange(1, p)
                weights = softmax(weights)
                x_avg = np.average(np.array(x_all)[-(p-1):], weights=weights, axis=0)
            else:
                x_avg = x_cur

            x_all.append(x_cur)

            t_cur = np.matmul(x_avg, s) + m
            t_cur = np.random.binomial(1, 1 / (1 + np.exp(-t_cur)))
            t.append(t_cur)

            tmp = np.concatenate((x_avg, [t_cur]))
            y_t_f = t_cur * np.matmul(tmp, w1) + (1 - t_cur) * np.matmul(tmp,w0)
            y_t_cf = t_cur * np.matmul(tmp, w0) + (1 - t_cur) * np.matmul(tmp,w1)

            y_f.append(y_t_f)
            y_cf.append(y_t_cf)

            if t_cur == 0:
                y0.append(float(y_t_f[0]))
                y1.append(float(y_t_cf[0]))
            else:
                y0.append(float(y_t_cf[0]))
                y1.append(float(y_t_f[0]))

        data_sim[id]['treatment'] = t
        factual = np.concatenate(y_f)
        counterfactual = np.concatenate(y_cf)
        data_sim[id]['outcome'] = factual.tolist()
        truth_patients.append(
            {
                'source_patient_id': str(id),
                'y0': y0,
                'y1': y1,
                'ite': (np.asarray(y1) - np.asarray(y0)).tolist(),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('wb') as handle:
        pickle.dump(data_sim, handle, protocol=pickle.HIGHEST_PROTOCOL)
    labels = _split_labels(len(data_sim), split_seed)
    split_patients = [
            {'source_patient_id': str(patient_id), 'split': label}
            for patient_id, label in zip(data_sim, labels)
        ]
    truth_path = Path(truth_output)
    split_path = Path(split_output)
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    truth_path.write_text(
            json.dumps(
                {
                    'metadata': {
                        'n_patients': len(data_sim),
                        'sequence_length': len(next(iter(data_sim.values()))['treatment']),
                        'dgp_seed': seed,
                        'training_dataset_contains_counterfactual_truth': False,
                    },
                    'patients': truth_patients,
                },
                separators=(',', ':'),
                allow_nan=False,
            )
        )
    split_path.write_text(
            json.dumps(
                {
                    'metadata': {
                        'n_patients': len(data_sim),
                        'split_seed': split_seed,
                        'fractions': {'train': 0.70, 'validation': 0.10, 'test': 0.20},
                        'split_counts': {
                            name: labels.count(name)
                            for name in ('train', 'validation', 'test')
                        },
                    },
                    'patients': split_patients,
                },
                separators=(',', ':'),
                allow_nan=False,
            )
        )
    print(f'Save synthetic-MIMIC data in {output_path}')


