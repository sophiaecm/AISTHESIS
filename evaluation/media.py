"""One temporary, hashed frame selection shared by every condition."""
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import math


def file_hash(path):
    digest = sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def prepare_media(path, *, kind='image', start=0, stop=None, stride=1):
    """Video indices are [start, stop), timestamps use frame_index / source FPS.

    Images are temporarily snapshotted byte-for-byte to prevent between-mode
    edits. Video frames decode sequentially once and are saved as lossless PNG.
    Timestamp convention is nominal FPS, not variable-frame-rate presentation time.
    """
    source = Path(path).resolve(strict=True)
    if kind not in ('image', 'video'):
        raise ValueError('kind must be image or video')
    if type(start) is not int or start < 0 or type(stride) is not int or stride < 1:
        raise ValueError('start must be nonnegative and stride must be positive integers')
    if stop is not None and (type(stop) is not int or stop <= start):
        raise ValueError('stop must be greater than start')
    identity = file_hash(source)
    with TemporaryDirectory(prefix='aisthesis-evaluation-') as directory:
        root = Path(directory)
        frames = []
        base = dict(source_path=str(source), source_sha256=identity, kind=kind)
        if kind == 'image':
            if start != 0 or stop is not None or stride != 1:
                raise ValueError('frame selection applies only to video')
            from shutil import copyfile
            frame_path = root / ('input' + source.suffix)
            copyfile(source, frame_path)
            if file_hash(frame_path) != identity:
                raise ValueError('source changed while preparing image')
            frames.append(dict(base, sequence=0, frame_number=0, timestamp=0.,
                               timestamp_basis='single_image_origin', frame_sha256=identity,
                               evaluation_path=str(frame_path)))
        else:
            import cv2
            capture = cv2.VideoCapture(str(source))
            try:
                if not capture.isOpened():
                    raise ValueError('video could not be opened')
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                if not math.isfinite(fps) or fps <= 0:
                    raise ValueError('video requires a finite positive FPS')
                index = 0
                while stop is None or index < stop:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if index >= start and (index - start) % stride == 0:
                        frame_path = root / f'{index:09d}.png'
                        if not cv2.imwrite(str(frame_path), frame):
                            raise ValueError('could not write temporary frame')
                        frames.append(dict(base, sequence=len(frames), frame_number=index,
                            timestamp=index / fps, timestamp_basis='frame_number/source_fps', fps=fps,
                            frame_sha256=file_hash(frame_path), evaluation_path=str(frame_path)))
                    index += 1
            finally:
                capture.release()
            if file_hash(source) != identity:
                raise ValueError('source changed while preparing video')
        if not frames:
            raise ValueError('selection contains no frames')
        yield frames
