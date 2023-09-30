import cv2
import imageio
import numpy as np
from tqdm import tqdm

from ..types.image_types import Image


class VideoProcessor:
    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self._stream.close()


class VideoReader(VideoProcessor):
    fps_key = "fps"
    size_key = "size"
    nframe_key = "nframes"
    duration_key = "duration"

    def __init__(self, fpath: str, verbose: bool = True):
        self._stream = imageio.get_reader(fpath, "ffmpeg")
        self._meta_data = self._stream.get_meta_data()
        self._verbose = verbose

    @property
    def fps(self) -> float:
        return self._meta_data[self.fps_key]

    @property
    def size(self) -> tuple[int, int]:
        return self._meta_data[self.size_key]

    @property
    def frame_number(self) -> int:
        # TODO: fix not accurate frame number
        n_frames = self._meta_data[self.nframe_key]

        if np.isinf(n_frames):
            n_frames = self.fps * self._meta_data[self.duration_key]

        return int(n_frames)

    @property
    def duration_s(self) -> float:
        frame_n = int(self._stream.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_count = self._stream.get(cv2.CAP_PROP_FPS)
        return frame_n / fps_count

    def get_frames(self):
        stream = self._stream.iter_data()
        if self._verbose:
            stream = tqdm(stream, total=self.frame_number)

        for frame in stream:
            frame = np.asarray(frame)
            yield frame


class VideoWriter(VideoProcessor):
    """Class implementation for video conversion from frames."""

    def __init__(self, fpath: str, fps: float = 30.0, verbose: bool = False):
        self._stream = imageio.get_writer(fpath, fps=fps)
        self._verbose = verbose

    def write_frame(self, frame: Image):
        """Write single frame."""

        self._stream.append_data(frame)

    def write(self, frames: list[Image]):
        """Write list of frames."""

        stream = frames
        if self._verbose:
            stream = tqdm(frames, desc="Frames processing")

        for frame in stream:
            self.write_frame(frame)
