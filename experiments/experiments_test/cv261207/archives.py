"""Read either original VisDrone ZIP layout with the shared record parser."""
from pathlib import PurePosixPath
from zipfile import ZipFile

from main.dataset import VisDroneZIP


def members(archive, directory, suffixes):
    return [name for name in archive.namelist()
            if PurePosixPath(name).parent.name == directory
            and name.lower().endswith(suffixes)
            and '__MACOSX' not in name
            and not PurePosixPath(name).name.startswith('._')]


class OriginalZIP(VisDroneZIP):
    def __init__(self, path, expected):
        self.archive = ZipFile(path)
        try:
            self.images = sorted(members(self.archive, 'images', ('.jpg', '.jpeg', '.png')))
            label_paths = members(self.archive, 'annotations', ('.txt',))
            self.labels = {PurePosixPath(n).stem: n for n in label_paths}
            ids = {PurePosixPath(n).stem for n in self.images}
            if (len(self.images) != expected or len(ids) != expected
                    or len(label_paths) != expected or ids != self.labels.keys()
                    or len(self.archive.namelist()) != len(set(self.archive.namelist()))):
                raise ValueError(f'Split membership mismatch: {path.name}')
        except Exception:
            self.archive.close()
            raise
