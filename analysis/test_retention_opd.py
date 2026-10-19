"""CPU checks for diagnostic noninterference and paired ER19 statistics.
Run with the qwen Python environment; uses a tiny CPU model, no real weights/cache.
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RetentionTests(unittest.TestCase):
    def test_diagnostic_does_not_change_weights_or_selection(self):
        saved = {}

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Dropout(0.25), torch.nn.Linear(8, 2))

            def save_pretrained(self, p):
                saved[str(p)] = {k: v.clone() for k, v in self.state_dict().items()}

        def encode(k, *_):
            i = int(k)
            return torch.tensor([[i / 10, 0.3, 1.0]]), torch.tensor([i % 2])

        def seq_logp(model, batch):
            x, y = batch
            return -torch.nn.functional.cross_entropy(model.net(x), y), torch.tensor(1)

        with tempfile.TemporaryDirectory() as tmp:
            common = types.ModuleType('common')
            common.RUNS = Path(tmp); common.CACHE = Path(tmp); common.ZMODE = 'mr'; common.QWEN = 'tiny'; common.NFRAMES = 16
            common.json = json; common.Path = Path; common.resolve = lambda x: x
            rows = {str(i): {'id': str(i), 'target': 'x'} for i in range(20)}
            common.manifest = lambda: {'rows': rows, 'train': list(rows)[:16], 'val': list(rows)[16:], 'split_sha256': 'fixture'}
            common.task_io = lambda r, t: ('prompt', r['target'])
            common.wjson = lambda p, r: p.write_text(json.dumps(r))
            io = types.ModuleType('vt_io'); io.load_model = lambda **kw: Tiny(); io.encode = encode; io.seq_logp = seq_logp
            with patch.dict(sys.modules, {'common': common, 'vt_io': io}), \
                 patch.object(torch.cuda, 'get_rng_state_all', return_value=[]), \
                 patch.object(torch.cuda, 'set_rng_state_all'), \
                 patch.object(torch.cuda, 'max_memory_allocated', return_value=0):
                module = load(REPO / 'pipeline/sft.py', 'sft_cpu_fixture')
                for name, extra in [('plain', []), ('diagnostic', ['--checkpoint-steps', '1'])]:
                    args = ['sft.py', '--mode', 'student', '--out', name, '--epochs', '3', '--stop-after', '1', '--lr', '2e-5'] + extra
                    with patch.object(sys, 'argv', args):
                        module.main()
                for k, v in saved[str(Path(tmp) / 'plain/epoch1')].items():
                    self.assertTrue(torch.equal(v, saved[str(Path(tmp) / 'diagnostic/epoch1')][k]), k)
                s = json.loads((Path(tmp) / 'diagnostic/selection.json').read_text())
                self.assertEqual(s['selected'], 'epoch1')
                self.assertEqual(set(s['val_nll']), {'1'})
                self.assertIn(str(Path(tmp) / 'diagnostic/step1'), saved)

    def test_bootstrap_samples_clusters_with_sample_weighting(self):
        mod = load(REPO / 'analysis/analyze_retention_opd.py', 'er19_analysis_fixture')
        d = np.array([1., -1., -1., 0.]); groups = np.array(['a', 'b', 'b', 'c'])
        actual = mod.bootstrap(d, groups, n=200)
        rng = np.random.default_rng(0); us = np.unique(groups)
        bs = [d[np.concatenate([np.flatnonzero(groups == u) for u in rng.choice(us, len(us))])].mean() for _ in range(200)]
        np.testing.assert_allclose(actual['ci95'], np.percentile(bs, [2.5, 97.5]))
        self.assertEqual(actual['diff'], -0.25)
        zero = mod.bootstrap((d - d) - (d - d), groups)
        self.assertEqual(zero['ci95'], [0., 0.])


if __name__ == '__main__':
    unittest.main()
