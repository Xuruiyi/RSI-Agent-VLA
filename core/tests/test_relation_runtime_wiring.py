"""Registered relation controls reach the real launch and request consumers."""

import copy
import unittest
from unittest.mock import Mock

import test_runtime_plan
import test_task_completion
import test_task_policy
from rsi_vla.storage import atomic_json
from rsi_vla.task_completion import TaskResearchAdapter


class RelationRuntimeProjectionTests(unittest.TestCase):
    setUp = test_runtime_plan.RuntimePlanTests.setUp
    plan = test_runtime_plan.RuntimePlanTests.plan

    def route(self):
        return dict(route_id='pick', task_key='position_swap_salad', state='pick-salad',
                    requested_worker='pi0_pick', trigger_workers=['pi0_pick', 'vla_execute'],
                    instruction='pick up the salad dressing bottle',
                    allowed_workers=['pi0_pick', 'vla_execute', 'handoff', 'defer'],
                    criterion=dict(schema='rsi-skill-lift-criterion-v1', criterion_ref='fixture-criterion',
                                   object_position_key='salad_dressing_1_pos', min_lift_m=.02,
                                   max_eef_object_distance_m=.15, min_gripper_opening=0.,
                                   max_gripper_opening=.08))

    def configure(self):
        path = self.artifacts / 'research/profiles/initial-medium.json'
        profile = test_runtime_plan.json.loads(path.read_text())
        route = self.route()
        profile['relation_routes'] = [route]
        profile['relation_skill_starts'] = {'position_swap_salad': dict(
            route_id='pick', tool='vla_execute', instruction=route['instruction'], max_chunks=4)}
        atomic_json(path, profile)
        return path, profile

    def test_real_runtime_projection_binds_route_and_cannot_start_services(self):
        self.configure()
        plan = self.plan()
        context = plan['context']
        self.assertEqual(context['relation_control']['routes'][0]['selected_worker'], 'pi0_pick')
        self.assertEqual(context['relation_control']['version_signature']['phi_ref'], context['phi_ref'])
        self.assertEqual(context['relation_control']['rho_ref'], context['rho_ref'])
        self.assertEqual(context['relation_skill_start']['max_chunks'], 4)
        self.assertFalse((self.artifacts / 'run').exists())
        self.assertTrue(plan['blockers'])

    def test_foreign_goal_or_action_allowance_is_rejected_before_dispatch(self):
        path, profile = self.configure()
        for change in [dict(instruction='place the bottle'), dict(max_chunks=25), dict(tool='pi0_doubled')]:
            with self.subTest(change=change):
                current = copy.deepcopy(profile)
                current['relation_skill_starts']['position_swap_salad'].update(change)
                atomic_json(path, current)
                with self.assertRaisesRegex(ValueError, 'skill start'):
                    self.plan()


class ScopeCandidateAccountingTests(unittest.TestCase):
    setUp = test_task_completion.ConsumerAdmissionTests.setUp

    def test_new_protocol_keeps_source_candidates_and_excludes_deployment_copies(self):
        self.scope['candidate_protocol_ids'] = ['old-source', self.backend.protocol['protocol_id']]
        atomic_json(self.config['scope_authorization_ref'], self.scope)
        for identity, protocol in [('source-old', 'old-source'), ('source-current', self.backend.protocol['protocol_id']),
                                   ('identity-copy', 'deployment-import')]:
            self.backend.store.put('candidates', identity, dict(candidate_id=identity, protocol_id=protocol))
        op = dict(operation_id='submit-third', kind='candidate_submit', payload=dict(
            request_id='r', candidate_id='third', experiment_ids=['trial'], cost_snapshot_ref='cost.json'))
        self.config['operations'] = [op]
        self.backend.tools = Mock()
        with self.assertRaisesRegex(PermissionError, 'Candidate cap'):
            self.backend.execute(op, {})
        self.backend.tools.assert_not_called()


class GoalOnlyTaskSchemaTests(unittest.TestCase):
    setUp = test_task_policy.SC3OpportunityTests.setUp
    result = test_task_policy.SC3OpportunityTests.result

    def test_task_goal_only_request_has_a_valid_empty_development_array(self):
        self.tools.domain_policy.update(opportunity_only=True, rule_operators=[])
        adapter = TaskResearchAdapter(dict(request_id='goal', reserved_bundle_id='unused', development_slots=[]))
        adapter.prepare_request(self.tools, request_id='goal', result_ids=['base-result'],
                                max_input_tokens=100, max_output_tokens=10)
        schema = adapter.response_schema('goal', tools=self.tools)
        slots = schema['properties']['development_slots']
        self.assertEqual(slots['maxItems'], 0)
        self.assertEqual(slots['items']['type'], 'object')
        self.assertNotIn('anyOf', slots['items'])
        self.assertEqual(schema['properties']['decision']['enum'], ['propose_goal', 'defer'])


if __name__ == '__main__':
    unittest.main()
