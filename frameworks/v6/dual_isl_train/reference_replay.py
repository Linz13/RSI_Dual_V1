"""Conservative reference reuse scoped to one read-only TTS rollout call."""
from __future__ import annotations

from .adapters import audit_adapter_pair


class RolloutReferenceReplay:
    def __init__(self, worker):
        self.worker = worker
        self.verified = False
        self.metrics = {'requested': worker.cfg.get('generation', {}).get('reference_replay_reuse', False),
                        'reason': 'disabled', 'candidates': 0, 'reused': 0, 'independent': 0,
                        'validation_candidates': 0, 'validation_max_abs_error': None,
                        'adapter_audit': None}
        self.eligible = False
        if not self.metrics['requested']:
            return
        if not worker.has_reference_adapter:
            self.metrics['reason'] = 'base_reference'
            return
        if not self._eval():
            self.metrics['reason'] = 'not_eval'
            return
        # Tensor equality alone does not establish equality of LoRA scaling/config.
        configs = worker.policy.peft_config
        left, right = (dict(configs[name].to_dict()) for name in ('default', 'reference'))
        for cfg in (left, right):
            cfg.pop('inference_mode', None)  # trainability does not affect eval forward
        if left != right:
            self.metrics['reason'] = 'adapter_config_mismatch'
            return
        audit = audit_adapter_pair(worker.policy)
        self.metrics['adapter_audit'] = audit
        if not audit['exact'] or not audit['checked_tensors']:
            self.metrics['reason'] = 'adapter_mismatch'
            return
        self.versions = self._versions()
        self.eligible = True
        self.metrics['reason'] = 'equal_adapters'

    def _eval(self):
        return not any(module.training for module in self.worker.policy.modules())

    def _versions(self):
        # Catch a parameter update/replacement within this session. Never cache
        # equivalence across optimizer updates, worker actions or rollout calls.
        return tuple((name, id(p), p._version, str(p.dtype), str(p.device))
                     for name, p in self.worker.policy.named_parameters())

    def reference(self, request, codes, policy_values):
        self.metrics['candidates'] += 1
        if self.eligible and (not self._eval() or self.versions != self._versions()):
            self.eligible = False
            self.metrics['reason'] = 'model_state_changed'
        if self.eligible and self.verified:
            self.metrics['reused'] += 1
            return tuple(v.detach() for v in policy_values), 'reused'
        actual = self.worker.incremental_trajectory_logprobs(request, codes, reference=True, grad=False)
        self.metrics['independent'] += 1
        if self.eligible:
            torch = self.worker.torch
            errors = []
            for a, b in zip(policy_values, actual):
                if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
                    self.eligible = False
                    self.metrics['reason'] = 'validation_nonfinite_or_shape'
                    return actual, 'independent_validation_failed'
                errors.append(float((a.float() - b.float()).abs().max().item()))
            self.metrics['validation_candidates'] += 1
            self.metrics['validation_max_abs_error'] = max(errors)
            # Production admission is stricter than the independent probe's 5e-4:
            # require exact equality on the first candidate on every rank.
            self.verified = max(errors) == 0.0
            if not self.verified:
                self.eligible = False
                self.metrics['reason'] = 'validation_difference'
                return actual, 'independent_validation_failed'
            return actual, 'independent_validation'
        return actual, 'independent_' + self.metrics['reason']
