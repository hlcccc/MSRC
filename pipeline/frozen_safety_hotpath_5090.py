"""Opt-in safety hot path, gated by paired GPU development validation."""
from dataclasses import asdict
import math
import time

import numpy as np

from extensions.multilayer.runner import ARMS, validate_commits
from pipeline.frozen_hotpath_5090 import prepare_directions
from safety_lab.multilayer_capture import LayerCapture
from safety_lab.multilayer_signals import project, selected_layers


def signal_readback(vector, logits, entropy):
    """Copy unchanged float32 features and scalar results in one transfer."""
    import torch

    if vector.ndim != 2 or vector.shape[0] != 3 or vector.shape[1] < 1:
        raise ValueError('expected exactly three single-request layer vectors')
    if logits.shape != (1, 2) or entropy.ndim != 0:
        raise ValueError('expected one binary risk and one scalar entropy')
    if (logits.dtype != torch.float32 or entropy.dtype != torch.float32 or
            logits.device != vector.device or entropy.device != vector.device):
        raise ValueError('expected original float32 signals on the same device')
    if vector.requires_grad or logits.requires_grad or entropy.requires_grad:
        raise ValueError('inference-only readback required')
    risk = logits.softmax(-1)[0, 1]
    shape = tuple(vector.shape)
    packed = torch.cat((vector.float().reshape(-1), risk.reshape(1), entropy.reshape(1)))
    values = packed.cpu().numpy()
    return values[:-2].reshape(shape), float(values[-2]), float(values[-1])


def can_skip_scoring(decisions, fitted, live_arm, retain_features):
    """The full shadow must continue even when every delivery gate is blocked."""
    return (fitted is not None and live_arm is None and not retain_features and
            set(decisions) == set(ARMS) and
            all(decisions[name].alarm is True for name in ARMS))


def decode_request_candidate(experiment, model, proc, detector, row, fitted=None,
                             limit=128, live_arm=None, *, retain_features=True,
                             diagnostics=None):
    """Preserve generation and gate math; optionally discard unused shadow scores.

    retain_features=False is a separate contract: features are explicitly None.
    It is not valid for development, calibration, or any feature-consuming caller.
    Only the guarded safety continuation may enable this implementation.
    """
    th, pg = experiment.torch, experiment.pg
    controls, decisions = experiment.controls(fitted), {}
    prepared = prepare_directions(fitted) if fitted is not None else None
    events = {arm: [] for arm in ARMS}
    tokens, layers, risks, entropies = [], [], [], []
    decoder = experiment.decoder(model)
    stats = dict(scored_tokens=0, shadow_only_tokens=0, joint_signal_readbacks=0,
                 retain_features=retain_features, candidate_decode=True)
    started = time.time_ns()
    th.cuda.synchronize()
    clock = time.perf_counter()
    with th.inference_mode(), LayerCapture(decoder, selected_layers(len(decoder.layers))) as capture:
        inputs = pg.inputs_for(proc, row)
        ids = inputs['input_ids']
        kwargs = {k: v for k, v in inputs.items() if k != 'input_ids'}
        kwargs.update(use_cache=True, logits_to_keep=1)
        kwargs['position_ids'] = model._prepare_position_ids_for_generation(ids, dict(kwargs, input_ids=ids))
        processors = pg.native_greedy_processors(model, ids, kwargs, limit)
        capture.start(prefill=True)
        out = model(**model.prepare_inputs_for_generation(ids, is_first_iteration=True, **kwargs),
                    output_hidden_states=False, return_dict=True)
        _, prefix = capture.result()
        state = pg.prefix_init(detector, prefix.float())
        del prefix
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        for position in range(1, limit + 1):
            skip = can_skip_scoring(decisions, fitted, live_arm, retain_features)
            raw = out.logits[:, -1].float()
            if not skip:
                probs = raw.softmax(-1)
                entropy_tensor = -(probs * probs.clamp_min(1e-30).log()).sum(-1)[0] / math.log(probs.shape[-1])
            token = processors(ids, raw).argmax(-1, keepdim=True)
            tid = int(token.item())
            if tid in eos:
                break
            kwargs = model._update_model_kwargs_for_generation(out, kwargs)
            ids = th.cat([ids, token], -1)
            if skip:
                capture.__exit__(None, None, None)
            else:
                capture.start(prefill=False)
            out = model(**pg.prepare_decode_inputs(model, ids, kwargs),
                        output_hidden_states=False, return_dict=True)
            tokens.append(tid)
            if skip:
                stats['shadow_only_tokens'] += 1
                continue
            vector, hidden = capture.result()
            logits, state = pg.online_step(detector, hidden.float(), state)
            vector, risk, entropy = signal_readback(vector, logits, entropy_tensor)
            z = project(vector, prepared) if fitted is not None else np.zeros(3)
            stats['scored_tokens'] += 1
            stats['joint_signal_readbacks'] += 1
            if retain_features:
                risks.append(risk)
                entropies.append(entropy)
                if fitted is None:
                    layers.append(vector.astype(np.float16))
            for arm, control in controls.items():
                if arm in decisions:
                    continue
                d = (control.step(risk, entropy, z) if arm in ARMS[2:] else
                     control.step(risk, entropy if arm == 'controller' else None))
                scored = time.time_ns()
                events[arm].append(dict(position=position, risk=risk, entropy=entropy,
                    decision=asdict(d), score_ns=scored,
                    release_ns=time.time_ns() if d.released else None))
                if d.alarm:
                    decisions[arm] = d
            if live_arm is not None and live_arm in decisions:
                break
        for arm, control in controls.items():
            if arm not in decisions:
                decisions[arm] = control.finish()
    th.cuda.synchronize()
    elapsed = time.perf_counter() - clock
    if not tokens:
        raise ValueError('empty generated response; preserve as failure')
    arms = []
    for arm in ARMS:
        d = decisions[arm]
        answer = proc.tokenizer.decode(tokens[:d.released], skip_special_tokens=True)
        if d.alarm:
            answer += '\nI cannot help with that request.'
        arms.append(dict(**{k: row[k] for k in ('id', 'group', 'suite', 'split', 'label')},
            arm=arm, warn=d.alarm, reason=d.reason,
            trigger_token=d.generated if d.alarm else None,
            generated_tokens=d.generated, released_tokens=d.released, answer=answer,
            events=events[arm], start_ns=started, shadow_tokens=len(tokens),
            shared_shadow_generation=True))
    if live_arm is None:
        validate_commits([dict(id=row['id'], arms=arms)], {row['id']: row})
    features = (dict(layers=np.asarray(layers, dtype=np.float16),
                     risks=np.asarray(risks), entropies=np.asarray(entropies))
                if retain_features else None)
    if diagnostics is not None:
        diagnostics.update(stats)
    return arms, tokens, features, elapsed
