"""Single-flight deep result admission and bounded, monotonic display lifetime."""
from copy import deepcopy
from threading import RLock
from time import monotonic, time
from uuid import uuid4

DEEP_ANALYSIS_TIMEOUT_SECONDS = 12.0
TEMPORAL_DEEP_MAX_AGE_SECONDS = 15.0
TEMPORAL_DEEP_DISPLAY_TTL_SECONDS = 5.0
DEEP_SHUTDOWN_TIMEOUT_SECONDS = 0.10
SCENE_DESCRIPTION_MAX_AGE_SECONDS = 45.0
SCENE_DESCRIPTION_DISPLAY_TTL_SECONDS = 25.0


class DeepAnalysisState:
    def __init__(self, logger=None, clock=monotonic, wall_clock=time):
        self.logger, self.clock, self.wall_clock = logger, clock, wall_clock
        self.lock = RLock()
        self.sequence = self.applied_sequence = self.generation = 0
        self.busy = None
        self.active = None
        self.worker = None
        self.scene = None
        self.scene_sequence = 0
        self.scene_enabled = False

    def trace(self, event, job=None, **fields):
        if self.logger:
            self.logger(event, **{**(self._metadata(job) if job else {}), **fields})

    def apply_scene(self, job, description):
        with self.lock:
            age = self._metadata(job)['result_age_seconds']
            reason = ('session_superseded' if job['generation'] != self.generation else
                      'sequence_superseded' if job['sequence'] <= self.scene_sequence else
                      'snapshot_age_exceeded' if age > SCENE_DESCRIPTION_MAX_AGE_SECONDS else
                      'empty_description' if not description.strip() else None)
            if reason:
                self.trace('SCENE_DESCRIPTION_STALE', job, status='stale', rejection_reason=reason)
                return False
            self.scene = (description.strip(), job, self.clock())
            self.scene_sequence = job['sequence']
            self.trace('SCENE_DESCRIPTION_APPLIED', job, status='fresh', temporal_status=job['status'], rejection_reason=None)
            return True

    def scene_description(self):
        with self.lock:
            if not self.scene_enabled:
                return None
            if self.scene:
                description, job, applied = self.scene
                if (self.clock()-applied < SCENE_DESCRIPTION_DISPLAY_TTL_SECONDS and
                        self._metadata(job)['result_age_seconds'] <= SCENE_DESCRIPTION_MAX_AGE_SECONDS):
                    return description
                self.trace('SCENE_DESCRIPTION_STALE', job, rejection_reason='display_ttl_or_snapshot_age')
                self.scene = None
            return 'Analyzing scene...'

    def _event(self, name, job):
        if self.logger:
            self.logger(name, **self._metadata(job))

    def _metadata(self, job):
        now = self.clock()
        return dict(analysis_id=job['analysis_id'], source='temporal_deep',
            snapshot_sequence_id=job['sequence'], snapshot_timestamp=job['snapshot_timestamp'],
            analysis_started_timestamp=job['started_wall'],
            analysis_completed_timestamp=job.get('completed_wall'),
            result_age_seconds=job['initial_age']+max(0, now-job['started']),
            analysis_duration_seconds=max(0, job.get('completed_mono', now)-job['started']), status=job['status'],
            track_id=job.get('track_id'), prediction_id=job.get('prediction_id'))

    def begin(self, snapshot):
        with self.lock:
            if self.busy is not None:
                return None
            self.sequence += 1
            wall = self.wall_clock()
            started = self.clock()
            snapshot_mono = getattr(snapshot, 'snapshot_monotonic', None)
            job = dict(analysis_id=uuid4().hex, sequence=self.sequence,
                generation=self.generation, snapshot_timestamp=snapshot.timestamp,
                started=started, started_wall=wall,
                initial_age=(max(0, started-snapshot_mono) if snapshot_mono is not None else
                             max(0, wall-snapshot.timestamp) if snapshot.timestamp is not None else 0),
                status='fresh')
            self.busy = job
            self.scene_enabled = True
            self._event('TEMPORAL_DEEP_STARTED', job)
            self.trace('SCENE_ANALYSIS_STARTED', job)
            return job

    def _status(self, job):
        if job['status'] != 'fresh':
            return job['status']
        if job['generation'] != self.generation or job['sequence'] <= self.applied_sequence:
            return 'superseded'
        elapsed = self.clock()-job['started']
        if elapsed > DEEP_ANALYSIS_TIMEOUT_SECONDS:
            return 'timed_out'
        if job['initial_age']+elapsed > TEMPORAL_DEEP_MAX_AGE_SECONDS:
            return 'stale'
        return 'fresh'

    def complete(self, job, publish):
        """Publish all side effects atomically only after admission checks."""
        with self.lock:
            job['completed_wall'] = self.wall_clock()
            job['completed_mono'] = self.clock()
            status = self._status(job)
            if status != 'fresh':
                self._reject(job, status)
                return False
            result = publish()
            self.applied_sequence = job['sequence']
            if result and result.get('source') == 'temporal_deep':
                job.update(track_id=result.get('track_id'), prediction_id=result.get('prediction_id'))
                result.update(self._metadata(job))
                self.active = (deepcopy(result), job, self.clock())
                self._event('TEMPORAL_DEEP_APPLIED', job)
            return True

    def _reject(self, job, status):
        if job.get('reported_status') != status:
            job['status'] = status
            job['reported_status'] = status
            self.trace('TEMPORAL_DEEP_'+status.upper(), job, rejection_reason=status)

    def finish(self, job):
        with self.lock:
            if self.busy is job:
                self.busy = None

    def current(self):
        with self.lock:
            if self.busy and self.clock()-self.busy['started'] > DEEP_ANALYSIS_TIMEOUT_SECONDS:
                self._reject(self.busy, 'timed_out')
            if self.active:
                result, job, applied = self.active
                if (self.clock()-applied >= TEMPORAL_DEEP_DISPLAY_TTL_SECONDS or
                        self._metadata(job)['result_age_seconds'] > TEMPORAL_DEEP_MAX_AGE_SECONDS):
                    self._event('TEMPORAL_DEEP_CLEARED', job)
                    self.active = None
                else:
                    return dict(deepcopy(result), result_age_seconds=self._metadata(job)['result_age_seconds'])
            return None

    def invalidate(self):
        with self.lock:
            self.generation += 1
            if self.busy:
                self._reject(self.busy, 'superseded')
            if self.active:
                self._event('TEMPORAL_DEEP_CLEARED', self.active[1])
            self.active = None
            self.scene = None
            self.scene_enabled = False

    def shutdown(self):
        self.invalidate()
        worker = self.worker
        if worker is not None:
            worker.join(timeout=DEEP_SHUTDOWN_TIMEOUT_SECONDS)
