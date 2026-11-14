"""Actual development costs settle before later policy-consumption evidence."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from rsi_vla.storage import atomic_json
from rsi_vla.task_completion import TaskCompletionBackend


class DevelopmentCostStageTests(unittest.TestCase):
    def test_cost_stage_uses_existing_settlement_without_collecting_or_reexecuting(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'closed-run'
            context = dict(run_id='run', episode_id='ep', bundle_id='candidate', protocol_id='protocol')
            atomic_json(output/'runtime_context.json', context)
            op = dict(operation_id='settle-run', kind='development_reconcile', payload=dict(
                request_id='request', experiment_id='experiment', output_ref=str(output),
                permit_ref='permit.json', costs_only=True))
            backend = TaskCompletionBackend.__new__(TaskCompletionBackend)
            backend.config = dict(operations=[op])
            tools = Mock()
            tools.job.state = {"charges": []}
            tools._hard_costs.return_value = {"model_requests": 4}
            backend.tools = Mock(return_value=(tools, {}))
            with patch('rsi_vla.research_cycle._settle_trial_output', return_value={'model_requests': 4}) as settle, \
                 patch('rsi_vla.research_cycle.reconcile_trial') as reconcile:
                value = backend.execute(op, {})
            settle.assert_called_once_with(tools, 'experiment', str(output), 'permit.json', context)
            reconcile.assert_not_called()
            tools.job.consume.assert_called_once_with({"model_requests": 4}, "experiment experiment")
            self.assertTrue(value['costs_settled'])
            self.assertFalse(value['result_collected'])


if __name__ == '__main__':
    unittest.main()
