"""Independent development view references stay readable after reconciliation."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rsi_vla.storage import ResearchStore, atomic_json
from rsi_vla.task_completion import TaskCompletionBackend


class DevelopmentViewMaterializationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.backend = TaskCompletionBackend.__new__(TaskCompletionBackend)
        self.backend.store = ResearchStore(self.root/'research')
        self.backend.evaluations = SimpleNamespace(root=self.root/'evaluation')
        self.ref = self.root/'evaluation/development/experiment/imports/evidence_views/view.json'
        self.view = dict(view_id='view', bundle_id='bundle', l0=[dict(active_bundle='bundle', run_id='run', episode_id='ep')], selected_media=[])
        atomic_json(self.ref, self.view)
        self.result = dict(role='development', bundle_id='bundle', data_refs=[str(self.ref)],
                           intervention=dict(evidence_view_id='view', information_scope='development',
                                             identity=dict(run_id='run', episode_id='ep')))

    def test_exact_view_is_copied_without_relabeling_or_changing_source(self):
        before = self.ref.read_bytes()
        self.backend.preserve_development_views([self.result])
        self.assertEqual(self.backend.store.get('evidence_views', 'view'), self.view)
        self.assertEqual(self.ref.read_bytes(), before)
        self.backend.preserve_development_views([self.result])

    def test_formal_namespace_or_conflicting_identity_cannot_be_imported(self):
        foreign = self.root/'evaluation/results/view.json'
        atomic_json(foreign, self.view)
        with self.assertRaisesRegex(ValueError, 'independent import domain'):
            self.backend.preserve_development_views([{**self.result, 'data_refs': [str(foreign)]}])
        atomic_json(self.ref, {**self.view, 'bundle_id': 'other'})
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.backend.preserve_development_views([self.result])
