"""Demonstration bounds are checked before the actual external download boundary."""

import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


class DemonstrationLimitTests(unittest.TestCase):
    def test_real_cli_passes_explicit_subset_bounds_and_rejects_overpermit_before_fetch(self):
        repo = Path(__file__).resolve().parents[2]
        module = runpy.run_path(str(repo / 'scripts/prepare_pi05_training.py'))
        main = module['main']
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ):
            root = Path(temp)
            permit = root / 'permit.json'
            permit.write_text(json.dumps(dict(authorized=True, max_subset_frames=239,
                                             max_subset_bytes=268435456)))
            acquire = Mock(return_value={})
            materialize = Mock(return_value={'status': 'isolated subset'})
            args = ['prepare_pi05_training.py', '--acquire', 'demonstrations', '--permit', str(permit),
                    '--metadata-root', str(root / 'metadata'), '--output', str(root / 'raw'),
                    '--subset-output', str(root / 'subset'), '--max-subset-frames', '239',
                    '--max-subset-bytes', '268435456']
            with patch.dict(main.__globals__, dict(acquire_demonstration_files=acquire,
                    materialize_demonstration_subset=materialize, dynamic_lock=contextlib.nullcontext)):
                with patch.object(sys, 'argv', args), contextlib.redirect_stdout(io.StringIO()):
                    main()
                self.assertEqual(materialize.call_args.kwargs,
                                 dict(max_frames=239, max_output_bytes=268435456))
                acquire.reset_mock()
                with patch.object(sys, 'argv', args[:-1] + ['268435457']):
                    with self.assertRaisesRegex(ValueError, 'exceeds permit'):
                        main()
                acquire.assert_not_called()


if __name__ == '__main__':
    unittest.main()
