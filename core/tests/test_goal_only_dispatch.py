"""A registered goal-only request is not blocked by absent runtime-rule trials."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rsi_vla.bounded_runner import execute
from rsi_vla.research_transport import pack


class StopAtDispatchBoundary(Exception):
    pass


class GoalOnlyDispatchTests(unittest.TestCase):
    def call(self, request):
        with tempfile.TemporaryDirectory() as temp:
            tools = SimpleNamespace(domain_policy=dict(opportunity_only=True), job=SimpleNamespace(state={'job_id': 'goal-job'}), base=SimpleNamespace(bundle_id='candidate'), protocol={'protocol_id': 'protocol'})
            lock = Mock(side_effect=StopAtDispatchBoundary)
            with patch('rsi_vla.bounded_runner.validate_permit', return_value=request), \
                 patch('rsi_vla.bounded_runner.read', return_value={'limits': {'model_requests': 1}}):
                return execute(tools, Mock(), pack(request), root=Path(temp), output=Path(temp)/'not-started',
                               permit_path=Path(temp)/'permit.json', preflight=[], cost_limit={},
                               gate_factory=Mock(), supervisor=Mock(), lock_factory=lock)

    def request(self):
        return dict(request_id='goal', opportunity_only=True, allowed_operators=[],
                    task_policy=dict(opportunity_discovery=True))

    def test_registered_goal_reaches_dispatch_without_a_rule_operator(self):
        with self.assertRaises(StopAtDispatchBoundary):
            self.call(self.request())

    def test_goal_flag_cannot_smuggle_an_experiment(self):
        with self.assertRaisesRegex(PermissionError, 'no trial operators'):
            self.call({**self.request(), 'allowed_operators': ['relation_feedback_update']})

    def test_normal_research_still_requires_feasible_operator(self):
        request = self.request()
        request.pop('opportunity_only')
        with self.assertRaisesRegex(PermissionError, 'no executable evidenced'):
            self.call(request)


if __name__ == '__main__':
    unittest.main()
