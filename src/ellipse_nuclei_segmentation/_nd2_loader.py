import numpy as np
import nd2
from einops import rearrange


class ND2Loader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.movie = None
        self.metadata = None
        self.n_frames = 0
        self.spacing = None
        self.min_spacing = None
        self._load()

    def _load(self):
        self.movie = nd2.imread(self.data_path, dask=True)

        with nd2.ND2File(self.data_path) as ndfile:
            self.metadata = ndfile.metadata

        self.n_frames = self.movie.shape[0]

        self.spacing = self.metadata.channels[0].volume.axesCalibration[::-1]
        self.min_spacing = min(self.spacing)
        self.scale = tuple(s / self.min_spacing for s in self.spacing)

    def read_frame(self, frame_num: int) -> np.ndarray:
        frame_num = int(np.clip(frame_num, 0, self.n_frames - 1))
        frame = np.asarray(self.movie[frame_num])
        return rearrange(frame, "z c y x -> c z y x")

    def get_channels(self, frame_num: int):
        volume = self.read_frame(frame_num)
        gfp = volume[0].astype(np.float32)
        bf = volume[1].astype(np.float32)
        return gfp, bf

    @property
    def channel_names(self):
        return ["GFP", "BF"]

    @property
    def n_z_slices(self):
        if self.movie is not None:
            return self.movie.shape[1]  # (frames, z, channels, y, x)
        return 0

    def close(self):
        self.movie = None
