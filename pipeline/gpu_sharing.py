"""Explicit, run-local permission for cooperative PlugGuard GPU sharing."""

import json
import os

from pipeline.common import inside, sha


def load_shared_policy():
    name = os.environ.get('MSRC_GPU_SHARE_POLICY')
    if not name:
        return None
    if os.environ.get('MSRC_SAFETY_BACKEND') != 'plugguard':
        raise ValueError('GPU sharing is only supported by the bounded PlugGuard backend')
    path = inside(name)
    if hasattr(os, 'getuid') and path.stat().st_uid != os.getuid():
        raise ValueError('GPU sharing policy must belong to the current user')
    policy = json.loads(path.read_text(encoding='utf-8'))
    expected = {'version', 'gpu_index', 'gpu_uuid', 'allowed_other_users',
                'allocator_cap_gib', 'min_start_free_mib', 'min_run_free_mib',
                'target_duty_when_shared', 'check_interval_seconds', 'authorization'}
    if set(policy) != expected or policy['version'] != 1:
        raise ValueError('Unknown or incomplete GPU sharing policy')
    if (type(policy['gpu_index']) is not int or policy['gpu_index'] < 0
            or not isinstance(policy['gpu_uuid'], str)
            or not policy['gpu_uuid'].startswith('GPU-')):
        raise ValueError('GPU sharing policy needs one exact index and UUID')
    users = policy['allowed_other_users']
    if (not isinstance(users, list) or not users
            or not all(isinstance(user, str) and user.strip() == user and user for user in users)
            or len(set(users)) != len(users)):
        raise ValueError('GPU sharing requires explicit other-process owners')
    if (policy['allocator_cap_gib'] != 24
            or not isinstance(policy['min_start_free_mib'], int)
            or not isinstance(policy['min_run_free_mib'], int)
            or policy['min_start_free_mib'] < 32768
            or policy['min_run_free_mib'] < 8192
            or policy['min_start_free_mib'] <= policy['min_run_free_mib']
            or not 0 < policy['target_duty_when_shared'] <= 0.5
            or not 0 < policy['check_interval_seconds'] <= 10):
        raise ValueError('GPU sharing memory reserve or cooperative duty is unsafe')
    if not isinstance(policy['authorization'], str) or not policy['authorization'].strip():
        raise ValueError('GPU sharing needs a recorded user authorization')
    return dict(policy, source_path=str(path), source_sha256=sha(path))
