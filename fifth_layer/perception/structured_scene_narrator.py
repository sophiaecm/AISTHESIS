"""Evidence-only English narration. No model calls and no perception mutations."""
from time import perf_counter, time, monotonic
from queue import Queue, Full
from threading import Thread
import traceback
from fifth_layer.perception.spatial import horizontal_position, vertical_position, distance_relation, overlap_relation

MIN_NARRATION_CONFIDENCE = 0.5
MOTION_DEADBAND = 0.01
FAST_SCENE_MAX_AGE_SECONDS = 3.0
FAST_SCENE_DISPLAY_TTL_SECONDS = 3.0
FAST_SCENE_DEBOUNCE_SECONDS = 0.3
FAST_SCENE_MIN_DISPLAY_SECONDS = 3.0
FAST_SCENE_NORMAL_UPDATE_INTERVAL_SECONDS = 1.5
FAST_SCENE_CONFIRMATION_OBSERVATIONS = 2
ROAD_ACTORS = {'person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'traffic light', 'stop sign'}
RISK_ORDER = {'UNKNOWN': 0, 'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'IMMEDIATE': 4}
MOTIONS = {'stationary': 'is stationary in the image', 'moving_left': 'is moving left',
           'moving_right': 'is moving right', 'moving_up': 'is moving upward',
           'moving_down': 'is moving downward', 'moving_upward': 'is moving upward',
           'moving_downward': 'is moving downward'}


def location(d, width, height):
    if not width or not height:
        return 'in the image'
    try:
        h = horizontal_position(d, width)
        v = {'top': 'upper', 'bottom': 'lower', 'middle': 'middle'}[vertical_position(d, height)]
    except (ValueError, TypeError, KeyError):
        return 'in the image'
    region = h if v == 'middle' else v if h == 'center' else v+'-'+h
    if region in {'upper', 'lower'}:
        region += ' region'
    return 'near the '+region+' of the scene'


class StructuredSceneNarrator:
    def __init__(self, mode='access', detail_level=None, max_characters=1600, max_sentences=None):
        self.mode = mode
        self.detail_level = detail_level or ('accessibility_detailed' if mode == 'access' else 'detailed')
        if mode not in {'access', 'adas'} or self.detail_level not in {'concise', 'detailed', 'accessibility_detailed'}:
            raise ValueError('Unsupported narration mode or detail level')
        self.max_characters = max_characters
        self.max_sentences = max_sentences or {'concise': 3, 'detailed': 8, 'accessibility_detailed': 10}[self.detail_level]

    def narrate(self, world_state):
        started = perf_counter()
        data = world_state.data if hasattr(world_state, 'data') else world_state
        stamp = getattr(world_state, 'timestamp', data.get('observation_timestamp'))
        accepted = [d for d in data.get('accepted_detections', [])
                    if d.get('observation_state', 'observed') == 'observed'
                    and not any(d.get(k) for k in ('is_predicted', 'predicted', 'uncertain', 'rejected'))
                    and d.get('accepted') is not False
                    and d.get('status') not in {'uncertain', 'rejected', 'predicted'}
                    and float(d.get('confidence', 1)) >= MIN_NARRATION_CONFIDENCE]
        width, height = data.get('image_width', 0), data.get('image_height', 0)
        stable = data.get('stable_motion_evidence', [m for m in data.get('motion_evidence', []) if m.get('stable') is True])
        motion_by_id = {m['track_id']: m for m in stable if m.get('track_id') is not None}
        accepted.sort(key=lambda d: (self.mode == 'adas' and d.get('class_name') not in ROAD_ACTORS,
            motion_by_id.get(d.get('track_id'), {}).get('motion_state', 'stationary') == 'stationary'))
        groups = {}
        for d in accepted:
            groups.setdefault(d.get('class_name', 'object'), []).append(d)
        objects, object_sentences = [], []
        numbers = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five'}
        for label, items in groups.items():
            count = len(items)
            plural = {'person': 'people', 'bus': 'buses', 'mouse': 'mice'}.get(label,
                label+'es' if label.endswith(('s', 'ch', 'sh', 'x')) else label+'s')
            locs = list(dict.fromkeys(location(d, width, height) for d in items))
            where = locs[0] if len(locs) == 1 else 'across different parts of the image'
            objects.append(dict(class_name=label, count=count, locations=locs,
                                track_ids=[d.get('track_id') for d in items]))
            if len(object_sentences) < 3:
                object_sentences.append(f"{numbers.get(count, str(count))} {label if count == 1 else plural} {'is' if count == 1 else 'are'} visible {where}.")
        if len(groups) > 3:
            object_sentences.append('Other accepted objects are also visible in the image.')
        if not accepted:
            object_sentences = ['No objects are currently confirmed by accepted detections.']
        motion = []
        for d in accepted:
            m = motion_by_id.get(d.get('track_id'))
            if not m or m.get('reliable') is False:
                continue
            state = m.get('motion_state')
            if state != 'stationary' and m.get('normalized_motion', 1) < MOTION_DEADBAND:
                continue
            if state in MOTIONS:
                sentence = f"The tracked {d.get('class_name', 'object')} {MOTIONS[state]}."
                if sentence not in motion:
                    motion.append(sentence)
        if not motion:
            motion = ['No reliable motion estimate is currently available.']
        # Existing relation records refer to object IDs/indices; never infer depth.
        spatial = []
        labels = {d.get('object_id'): d.get('class_name', 'object') for d in accepted if d.get('object_id') is not None}
        relations = {'left_of': 'to the left of', 'right_of': 'to the right of',
                     'above': 'above', 'below': 'below', 'near': 'near',
                     'overlapping': 'overlapping with'}
        for r in data.get('scene_relations', []):
            a, b = labels.get(r.get('first_object_id')), labels.get(r.get('second_object_id'))
            rel = relations.get(r.get('relation'))
            if a and b and a != b and rel:
                spatial.append(f'The {a} is {rel} the {b} in the image.')
        if not spatial and len(accepted) >= 2 and width and height:
            a = accepted[0]
            b = next((d for d in accepted[1:] if d['class_name'] != a['class_name']), None)
            if b:
                try:
                    relation = ('overlapping with' if overlap_relation(a,b) == 'overlapping' else
                                'near' if distance_relation(a,b,width,height) == 'near' else None)
                    if relation:
                        spatial.append(f"The {a['class_name']} is {relation} the {b['class_name']} in the image.")
                except (ValueError, KeyError, TypeError):
                    pass
        occlusion = []
        predicted = [p for p in data.get('predicted_tracks', [])
                     if p.get('observation_state') == 'predicted' and p.get('is_predicted') is True
                     and (stamp is None or stamp <= p.get('prediction_valid_until', float('inf')))]
        for p in predicted[:2]:
            label = p.get('class_name', 'object')
            box = {'box_xyxy': p.get('predicted_bbox', [0,0,0,0])}
            occlusion.append(f"A previously observed {label} may be temporarily out of view {location(box, width, height)}; its position is an estimate.")
        if any(e.get('possible_occlusion_evidence') or e.get('has_overlap_evidence') for e in data.get('occlusion_evidence', [])):
            occlusion.append('Possible occlusion is indicated by the image geometry.')
        reasoning = data.get('narration_reasoning', data)
        latent = reasoning.get('latent', reasoning.get('latent_temporal_state', 'insufficient_evidence'))
        event = reasoning.get('prediction', reasoning.get('predicted_event', 'indeterminate'))
        latent_text = {'trajectory_toward_occlusion': 'The movement pattern suggests a possible loss of visibility.',
            'visibility_loss_possible': 'The image evidence suggests possible loss of visibility.',
            'trajectory_toward_view_exit': 'The movement pattern suggests an approach to the image boundary.'}.get(latent, '')
        prediction = ''
        if isinstance(event, str) and 'actor_may_emerge' in event:
            prediction = 'An actor may emerge from the occluded region.'
        elif latent == 'motion_continuation_possible' and any('is moving' in s for s in motion):
            prediction = 'The observed movement may continue over the next short interval.'
        elif latent in {'insufficient_evidence', 'stationary_state_observed'} or event in {'indeterminate', None}:
            prediction = 'There is insufficient evidence for a reliable near-future prediction.'
        risk = str(reasoning.get('risk', reasoning.get('risk_level', 'UNKNOWN'))).upper()
        risk_text = ('Immediate risk is indicated by the current evidence.' if risk == 'IMMEDIATE' else
                     'High risk is indicated by the current evidence.' if risk == 'HIGH' else
                     'Current risk is assessed as medium.' if risk == 'MEDIUM' else
                     'Current risk is assessed as low.' if risk == 'LOW' else
                     'Current risk cannot yet be determined reliably.')
        uncertainty = reasoning.get('uncertainty', 1.0)
        calibrated = reasoning.get('calibrated_confidence')
        uncertainty_text = ('Evidence is limited, so the near-future state remains uncertain.'
                            if float(uncertainty) >= .7 or (calibrated is not None and calibrated < .4) else '')
        if self.detail_level == 'concise':
            sentences = ([risk_text] if risk in {'HIGH', 'IMMEDIATE'} else []) + object_sentences[:1] + motion[:1] + [risk_text]
        else:
            sentences = ([risk_text] if risk in {'HIGH', 'IMMEDIATE'} else []) + object_sentences[:1] + motion[:1] + occlusion[:1] + [latent_text or prediction] + object_sentences[1:] + spatial[:1] + [risk_text, uncertainty_text]
        chosen = []
        for sentence in sentences:
            if sentence and sentence not in chosen and len(chosen) < self.max_sentences:
                if len(' '.join(chosen+[sentence])) <= self.max_characters:
                    chosen.append(sentence)
        result = dict(description=' '.join(chosen), detail_level=self.detail_level, mode=self.mode,
            objects_summary=objects, spatial_summary=list(dict.fromkeys(spatial)), motion_summary=motion,
            occlusion_summary=occlusion, latent_summary=latent_text, prediction_summary=prediction,
            risk_summary=risk_text, uncertainty_summary=uncertainty_text,
            evidence_types=(['observed'] if accepted else [])+(['inferred'] if latent_text else [])+(['predicted'] if predicted or (prediction and not prediction.startswith('There is insufficient')) else []),
            generated_timestamp=time(), source='structured_local', risk=risk,
            accepted_object_count=len(accepted), predicted_track_count=len(predicted), sentence_count=len(chosen))
        result['latency_ms'] = (perf_counter()-started)*1000
        return result


class FastSceneState:
    """Single retained result; semantic debounce and freshness independent of deep."""
    def __init__(self, narrator=None, logger=None, clock=monotonic, wall_clock=time):
        self.narrator = narrator or StructuredSceneNarrator(detail_level='detailed')
        self.logger, self.clock, self.wall_clock = logger, clock, wall_clock
        self.reset()

    def reset(self):
        self.result = None
        self.last_observation = None
        self.refreshed = self.changed = float('-inf')
        self.initial_age = 0
        self._last_display = None
        self.signature = self.candidate_signature = None
        self.candidate_count = self.empty_count = 0
        self.observed_ids = set()
        self.moving_actors = set()
        self.has_support = False
        self.cleared = False

    def _semantics(self, state, result):
        data = state.data
        reasoning = data.get('narration_reasoning', data)
        objects = tuple(sorted((o['class_name'], o['count'], tuple(sorted(o['locations'])))
                               for o in result['objects_summary']))
        # Text summaries are categorical; no confidence, pixels, IDs or counters.
        signature = (objects, tuple(sorted(result['motion_summary'])),
            tuple(sorted(result['occlusion_summary'])),
            str(reasoning.get('latent', reasoning.get('latent_temporal_state', ''))),
            str(reasoning.get('prediction', reasoning.get('predicted_event', ''))),
            result['risk'], result['uncertainty_summary'])
        ids = {i for o in result['objects_summary'] for i in o['track_ids'] if i is not None}
        motion = data.get('stable_motion_evidence', [m for m in data.get('motion_evidence', []) if m.get('stable') is True])
        road_ids = {i for o in result['objects_summary'] if o['class_name'] in ROAD_ACTORS
                    for i in o['track_ids'] if i is not None}
        moving = {m.get('track_id') for m in motion if m.get('track_id') in road_ids
                  and m.get('motion_state', '').startswith('moving_')
                  and m.get('reliable') is not False and m.get('normalized_motion', 1) >= MOTION_DEADBAND}
        predicted_ids = {p.get('track_id') for p in data.get('predicted_tracks', [])
                         if p.get('observation_state') == 'predicted' and p.get('is_predicted') is True
                         and state.timestamp <= p.get('prediction_valid_until', float('inf'))}
        event = signature[4].lower()
        critical_event = any(word in event for word in ('actor_may_emerge', 'collision', 'immediate_hazard'))
        critical = ((result['risk'] in {'HIGH', 'CRITICAL', 'IMMEDIATE'} and
                     result['risk'] != (self.result or {}).get('risk')) or
                    (critical_event and (self.signature is None or event != self.signature[4].lower())) or
                    bool(moving-self.moving_actors) or bool(self.observed_ids & predicted_ids))
        possible = any(e.get('possible_occlusion_evidence') or e.get('has_overlap_evidence')
                       for e in data.get('occlusion_evidence', []))
        signature += (possible,)
        if possible and self.signature is not None and not self.signature[-1]:
            critical = True
        return signature, ids, moving, critical

    def update(self, state, completed_observation=False):
        original_timestamp = state.timestamp
        if state.timestamp is None and completed_observation:
            from fifth_layer.world_state import WorldState
            state = WorldState(self.wall_clock(), state.data)
        count = len(state.data.get('accepted_detections', []))
        age = max(0, self.wall_clock()-state.timestamp) if state.timestamp is not None else None
        def diagnostic(reason, result=None, **extra):
            if self.logger:
                self.logger('FAST_SCENE_DEBUG', current_state_type=type(state).__name__,
                    fast_state_identity=id(self),
                    accepted_detections_count=count, narrator_detection_count=count,
                    tracked_ids=[d.get('track_id') for d in state.data.get('accepted_detections', [])],
                    observation_timestamp=state.timestamp, original_timestamp=original_timestamp,
                    observation_age_seconds=age, update_reason=reason,
                    generated_description=(result or {}).get('description'),
                    displayed_description=self.description(), **extra)
        if state.timestamp is None:
            diagnostic('missing_timestamp_without_completion')
            return self.result
        try:
            result = self.narrator.narrate(state)
            now = self.clock()
            if age > FAST_SCENE_MAX_AGE_SECONDS:
                diagnostic('observation_age_exceeded', result)
                return self.result
            if self.result is None and result['accepted_object_count'] == 0:
                diagnostic('waiting_for_first_accepted_detection', result)
                return None
            new_observation = state.timestamp != self.last_observation
            self.last_observation = state.timestamp
            signature, ids, moving, urgent = self._semantics(state, result)
            memory = state.data.get('fast_scene_track_memory', [])
            self.has_support = bool(result['accepted_object_count'] or result['predicted_track_count'] or
                                    any(t.get('status') in {'active', 'temporarily_missing'} for t in memory))
            if new_observation:
                self.empty_count = 0 if self.has_support else self.empty_count+1
            self.refreshed, self.initial_age = now, age
            same = signature == self.signature
            if signature == self.candidate_signature:
                if new_observation:
                    self.candidate_count += 1
            else:
                self.candidate_signature, self.candidate_count = signature, 1
            ready = (self.candidate_count >= FAST_SCENE_CONFIRMATION_OBSERVATIONS and
                     now-self.changed >= FAST_SCENE_MIN_DISPLAY_SECONDS and
                     now-self.changed >= FAST_SCENE_NORMAL_UPDATE_INTERVAL_SECONDS)
            publish = (self.result is None or self.cleared or urgent or ready) and (not same or self.cleared)
            reason = 'semantic_unchanged' if same else 'critical_change' if urgent else 'stable_change' if publish else 'debounced'
            # One missing observation never replaces the last useful narration.
            if not self.has_support:
                publish = False
                reason = 'awaiting_track_evidence' if self.empty_count < 2 else 'no_remaining_track_evidence'
            text_changed = False
            if publish:
                text_changed = self.result is None or result['description'] != self.result['description'] or self.cleared
                self.result = result
                self.signature, self.observed_ids, self.moving_actors = signature, ids, moving
                if text_changed:
                    self.changed = now
                self.cleared = False
            if self.logger:
                self.logger('FAST_SCENE_GENERATED',
                    **{k: result[k] for k in ('mode','detail_level','latency_ms','accepted_object_count','predicted_track_count','sentence_count')},
                    timestamp=state.timestamp, update_reason='completed_observation')
                self.logger('FAST_SCENE_UPDATED' if text_changed else 'FAST_SCENE_UNCHANGED',
                    **{k: result[k] for k in ('mode','detail_level','latency_ms','accepted_object_count','predicted_track_count','sentence_count')},
                    timestamp=state.timestamp, update_reason=reason)
            diagnostic(reason, result)
            return self.result
        except Exception as exc:
            if self.logger:
                self.logger('FAST_SCENE_ERROR', timestamp=state.timestamp, mode=self.narrator.mode,
                    detail_level=self.narrator.detail_level, latency_ms=0, accepted_object_count=0,
                    predicted_track_count=0, sentence_count=0, update_reason=str(exc),
                    traceback=traceback.format_exc())
            diagnostic('exception', traceback=traceback.format_exc())
            return self.result

    def description(self):
        elapsed = self.clock()-self.refreshed
        # No new inference is not evidence of disappearance. Clear only after
        # repeated real observations report neither objects nor track memory.
        if (self.result and not self.has_support and self.empty_count >= 2
                and self.clock()-self.changed >= FAST_SCENE_MIN_DISPLAY_SECONDS):
            self.cleared = True
        if self.result and not self.cleared:
            text, reason = self.result['description'], 'fresh_result'
        else:
            text = 'Waiting for a fresh structured observation...'
            reason = 'no_result' if self.result is None else 'no_remaining_track_evidence'
        if self.logger and text != self._last_display:
            self.logger('FAST_SCENE_DISPLAY_DEBUG', fast_state_identity=id(self),
                displayed_description=text, update_reason=reason,
                elapsed_since_refresh_seconds=elapsed if self.result else None,
                initial_observation_age_seconds=self.initial_age)
        self._last_display = text
        return text


class NarrationEventLogger:
    """Bounded nonblocking diagnostic delivery; overflow drops logs, never frames."""
    def __init__(self, sink):
        self.queue = Queue(maxsize=64)
        def consume():
            while True:
                event, fields = self.queue.get()
                try:
                    sink(event, **fields)
                except Exception:
                    pass
                finally:
                    self.queue.task_done()
        Thread(target=consume, daemon=True).start()

    def __call__(self, event, **fields):
        try:
            self.queue.put_nowait((event, fields))
        except Full:
            pass
