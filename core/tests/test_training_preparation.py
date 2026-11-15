"""The CPU pipeline consumes exactly one native batch and preserves admission."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rsi_vla.adapters.pi05_training import prepare_cpu_demonstrations

MODULE = 'rsi_vla.adapters.pi05_training.'


class PreparationTests(unittest.TestCase):
    def test_cpu_pipeline_routes_fixed_inputs_and_preserves_not_admitted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / 'evidence.json'
            evidence.write_text('{}')
            norm = root / 'norm.json'
            norm.write_text('{"norm_stats": {}}')
            permit = {key: str(evidence) for key in
                      ('license_evidence_ref', 'action_semantics_ref', 'evaluation_isolation_ref')}
            permit.update(max_subset_frames=239, max_subset_bytes=268435456)
            def subset(metadata, source, output, **kwargs):
                output.mkdir()
                return {'dataset_root': str(output), 'source_kind': 'public_demonstration'}
            with patch(MODULE+'verify_installed_sources', return_value={}), \
                 patch(MODULE+'acquire_demonstration_files', return_value={'revision': 'pinned'}) as acquire, \
                 patch(MODULE+'materialize_demonstration_subset', side_effect=subset) as convert, \
                 patch(MODULE+'inspect_native_config', return_value={}), \
                 patch(MODULE+'read_native_batch', return_value={'evidence_level': 'fixture'}) as batch:
                result = prepare_cpu_demonstrations(metadata_root=root/'metadata', output=root/'job',
                    permit=permit, model_path=root/'model', norm_stats_path=norm)
            acquire.assert_called_once_with(root/'metadata', root/'job/download', permit)
            self.assertEqual(convert.call_args.kwargs, {'max_frames':239, 'max_output_bytes':268435456})
            batch.assert_called_once()
            self.assertEqual(result['real_native_batches'], 1)
            self.assertEqual(result['training_steps'], 0)
            self.assertFalse(result['production_eligible'])
            self.assertEqual(result['manifest']['training_admission'], 'not_admitted_interface_batch_only')
            self.assertEqual(json.loads((root/'job/cpu-preparation.json').read_text())['actual_model_load'], 'not_run')

    def test_missing_isolation_blocks_before_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(MODULE+'acquire_demonstration_files') as acquire:
                with self.assertRaisesRegex(ValueError, 'local evidence'):
                    prepare_cpu_demonstrations(metadata_root=root, output=root/'job', permit={},
                        model_path=root/'model', norm_stats_path=root/'norm')
            acquire.assert_not_called()
            self.assertFalse((root/'job').exists())
