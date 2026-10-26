"""Synthetic/mock only. Run on the designated test host, never fetch a model."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (check_groups, loss_values, select_threshold, sha256,
                    verify_freeze, write_json)
import train


class TensorMock:
    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return [1.0, 2.0, 3.0]


class HelpersTest(unittest.TestCase):
    def test_tensor_and_dict_loss(self):
        self.assertEqual(loss_values(TensorMock()), [1, 2, 3])
        self.assertEqual(loss_values({'otm': TensorMock(), 'oto': [4, 5]}),
                         {'otm': [1, 2, 3], 'oto': [4, 5]})

    def test_nonfinite_loss_rejected(self):
        for value in (float('nan'), float('inf'), {'x': [float('-inf')]}):
            with self.assertRaises(ValueError):
                loss_values(value)

    def test_groups_never_reassigned(self):
        rows = [{'id': 'synthetic-a', 'group': 'event-a', 'split': 'train'},
                {'id': 'synthetic-b', 'group': 'event-b', 'split': 'val'}]
        before = json.dumps(rows)
        self.assertEqual(check_groups(rows), {'train': 1, 'val': 1})
        self.assertEqual(before, json.dumps(rows))

    def test_leaking_group_rejected(self):
        with self.assertRaisesRegex(ValueError, 'group_crosses'):
            check_groups([{'id': 'a', 'group': 'event', 'split': 'train'},
                          {'id': 'b', 'group': 'event', 'split': 'test'}])

    def test_identical_content_cross_split_rejected(self):
        with self.assertRaisesRegex(ValueError, 'identical_content'):
            check_groups([{'id': 'a', 'group': 'a', 'split': 'train', 'sha256': 'mock'},
                          {'id': 'b', 'group': 'b', 'split': 'val', 'sha256': 'mock'}])

    def test_missing_group_and_duplicate_id_rejected(self):
        for rows in ([], [{'id': 'a', 'split': 'train'}],
                     [{'id': 'a', 'group': 'a', 'split': 'train'}] * 2):
            with self.assertRaises(ValueError):
                check_groups(rows)

    def test_threshold_tie_uses_higher_confidence(self):
        rows = [dict(split='val', threshold=t, tp=3, fp_boxes=0, n=4)
                for t in (.2, .5)]
        self.assertEqual(select_threshold(rows), .5)

    def test_threshold_rejects_test_and_invalid_counts(self):
        for row in (dict(split='test', threshold=.5, tp=1, fp_boxes=0, n=1),
                    dict(split='val', threshold=.5, tp=2, fp_boxes=0, n=1)):
            with self.assertRaises(ValueError):
                select_threshold([row])

    def test_freeze_detects_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            item = root / 'synthetic.txt'
            item.write_text('synthetic fixture')
            frozen = root / 'freeze.json'
            write_json(frozen, {'schema': 1, 'files': {str(item): sha256(item)}})
            verify_freeze(frozen)
            item.write_text('changed fixture')
            with self.assertRaisesRegex(ValueError, 'frozen_input_changed'):
                verify_freeze(frozen)

    def test_wrong_head_rejected(self):
        for validator in (types.SimpleNamespace(end2end=True), types.SimpleNamespace()):
            with self.assertRaisesRegex(RuntimeError, 'end2end'):
                train.check_head(validator)

    def test_git_paths_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / '.git').write_text('mock worktree marker')
            with self.assertRaisesRegex(ValueError, 'outside_git'):
                train.outside_checkout(root / 'private.pt')

    def test_network_guard_denies_before_connect(self):
        # Restore patched socket members after the test.
        with patch.multiple(train.socket, getaddrinfo=Mock(), create_connection=Mock()), \
                patch.object(train.socket.socket, 'connect'), \
                patch.object(train.socket.socket, 'connect_ex'), \
                patch.object(train.socket.socket, 'sendto'):
            train.disable_network()
            with self.assertRaisesRegex(RuntimeError, 'network_disabled'):
                train.socket.getaddrinfo('example.invalid', 443)


class DatasetContractTest(unittest.TestCase):
    def test_local_directories_and_no_image_processing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ('train', 'val'):
                (root / split).mkdir()
                (root / split / 'synthetic.record').write_text('no image contents')
            config = root / 'data.yaml'
            config.write_text('mock input')
            data = dict(path=str(root), train='train', val='val', names=['object'])
            yaml = types.SimpleNamespace(safe_load=lambda text: data)
            with patch.dict(sys.modules, {'yaml': yaml}):
                normalized, files = train.local_dataset(config)
            self.assertEqual(normalized['train'], str(root / 'train'))
            self.assertEqual(len(files), 3)

    def test_download_script_rejected(self):
        yaml = types.SimpleNamespace(safe_load=lambda text: {'download': 'mock command'})
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / 'data.yaml'
            config.write_text('mock')
            with patch.dict(sys.modules, {'yaml': yaml}):
                with self.assertRaisesRegex(ValueError, 'only_local'):
                    train.local_dataset(config)

    def test_remote_split_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / 'data.yaml'
            config.write_text('mock')
            data = dict(path=str(root), train='https://example.invalid/data',
                        val='val', names=['object'])
            yaml = types.SimpleNamespace(safe_load=lambda text: data)
            with patch.dict(sys.modules, {'yaml': yaml}):
                with self.assertRaisesRegex(ValueError, 'local_split_path'):
                    train.local_dataset(config)


class EntryPointMockTest(unittest.TestCase):
    def exercise(self, mode):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / 'dataset'
            dataset.mkdir()
            files = [root / 'data.yaml', root / 'weights.pt', root / 'hparams.json',
                     dataset / 'synthetic.txt']
            for path in files:
                path.write_text('synthetic mock only')
            files[2].write_text('{"epochs": 2}')
            frozen = root / 'freeze.json'
            write_json(frozen, {'schema': 1, 'files': {str(p): sha256(p) for p in files}})
            output = root / 'output'
            args = ['train.py', mode, '--family', 'yolo26n', '--data', str(files[0]),
                    '--weights', str(files[1]), '--hparams', str(files[2]),
                    '--freeze', str(frozen), '--output', str(output)]
            model = Mock()
            model.task = 'detect'
            registered = {}
            model.add_callback.side_effect = lambda event, callback: registered.update({event: callback})
            def fit(**kwargs):
                device = types.SimpleNamespace(type='cpu')
                registered['on_train_start'](types.SimpleNamespace(device=device))
                registered['on_val_start'](types.SimpleNamespace(device=device))
                registered['on_fit_epoch_end'](types.SimpleNamespace(
                    validator=types.SimpleNamespace(end2end=False), epoch=2, epochs=2,
                    device=device, tloss={'otm': TensorMock(), 'oto': [1, 2, 3]}))
            model.train.side_effect = fit
            model.val.return_value.results_dict = {'metrics/mock': .5}
            yolo = Mock(return_value=model)
            modules = {'ultralytics': types.SimpleNamespace(YOLO=yolo),
                       'yaml': types.SimpleNamespace(safe_dump=lambda *a, **k: 'mock yaml')}
            backend = (types.SimpleNamespace(__version__=train.VERSION),
                       types.SimpleNamespace(__version__='mock-cpu',
                                             set_num_threads=Mock(), get_num_threads=lambda: 2))
            with patch.object(sys, 'argv', args), patch.dict(sys.modules, modules), \
                    patch.object(train, 'local_dataset', return_value=(
                        {'path': str(dataset), 'val': str(dataset)}, [files[3]])), \
                    patch.object(train, 'configure_backend', return_value=backend):
                train.main()
            call = model.train.call_args if mode == 'train' else model.val.call_args
            self.assertTrue(call.kwargs['nms'])
            self.assertEqual(call.kwargs['device'], 'cpu')
            self.assertEqual(call.kwargs['workers'], 0)
            self.assertFalse(call.kwargs['plots'])
            self.assertTrue((output / 'completion.json').is_file())
            if mode == 'train':
                self.assertTrue(call.kwargs['val'])
                self.assertFalse(call.kwargs['resume'])
                progress = json.loads((output / 'progress.json').read_text())
                self.assertEqual(progress['epoch_completed'], 2)
                self.assertEqual(progress['loss']['otm'], [1, 2, 3])
                self.assertEqual(progress['torch_threads'], 2)
                backend[1].set_num_threads.assert_called_with(2)

    def test_train_mock(self):
        self.exercise('train')

    def test_validation_mock(self):
        self.exercise('validate')


if __name__ == '__main__':
    unittest.main()
