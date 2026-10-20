"""
Single Evidence Folder Manager for Gate-In and Gate-Out.

Enforces:
1. Exactly ONE evidence folder per physical container event.
2. Structure:
     evidence/
     ├── GATE_IN/
     │   └── <event_id>_<container_number>/
     │       ├── <camera_id>_frame_<seq:03d>.jpg
     │       ├── ...
     │       ├── best_<camera_id>.jpg
     │       └── ...
     └── GATE_OUT/
         └── <event_id>_<container_number>/
             ├── <camera_id>_frame_<seq:03d>.jpg
             ├── ...
             ├── best_<camera_id>.jpg
             └── ...
3. If container number is not yet confirmed, creates folder as <event_id> and updates to
   <event_id>_<container_number> once confirmed, preserving all files.
4. All cameras assigned to the gate write directly into this single folder.
5. No separate folders per camera, angle, timestamp, frame batch, or OCR attempt.
6. Maintains backwards-compatibility links for legacy snapshot lookups.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

log = logging.getLogger('gate.evidence_manager')


def _create_junction_or_symlink(target_dir: Path, link_dir: Path) -> bool:
    """Create a Windows NTFS Directory Junction or POSIX symlink without requiring admin privileges."""
    try:
        if link_dir.exists() or link_dir.is_symlink():
            try:
                if link_dir.resolve() == target_dir.resolve():
                    return True
            except Exception:
                pass
            try:
                if sys.platform == 'win32':
                    os.unlink(str(link_dir))
                else:
                    link_dir.unlink(missing_ok=True)
            except Exception:
                pass
        else:
            # Might be a broken junction (exists() returns False for broken junctions on Windows)
            try:
                os.unlink(str(link_dir))
            except Exception:
                pass

        link_dir.parent.mkdir(parents=True, exist_ok=True)
        if sys.platform == 'win32':
            try:
                import _winapi
                _winapi.CreateJunction(str(target_dir.resolve()), str(link_dir.resolve()))
                return True
            except Exception:
                pass
        os.symlink(str(target_dir.resolve()), str(link_dir.resolve()), target_is_directory=True)
        return True
    except Exception:
        return False


class EvidenceFolderManager:
    """
    Central event-level evidence folder manager.
    Coordinates all cameras, frame storage, best-frame selection, and manifest generation.
    """

    def __init__(self, root: Path | str = 'evidence', legacy_root: Path | str = 'snapshots'):
        self.evidence_root = Path(root).resolve()
        self.legacy_root = Path(legacy_root).resolve() if legacy_root else (self.evidence_root.parent / 'snapshots')

        # Ensure base directories exist
        (self.evidence_root / 'GATE_IN').mkdir(parents=True, exist_ok=True)
        (self.evidence_root / 'GATE_OUT').mkdir(parents=True, exist_ok=True)
        self.legacy_root.mkdir(parents=True, exist_ok=True)

        # In-memory mapping of event_id -> active folder path
        self._folder_cache: Dict[str, Path] = {}

    @staticmethod
    def get_gate_type(gate_id: str = '', direction: str = '') -> str:
        """Categorize into 'GATE_IN' or 'GATE_OUT'."""
        dir_u = str(direction or '').upper()
        gate_l = str(gate_id or '').lower()
        if dir_u in ('ENTRY', 'GATE_IN', 'IN') or 'in' in gate_l:
            return 'GATE_IN'
        if dir_u in ('EXIT', 'GATE_OUT', 'OUT') or 'out' in gate_l:
            return 'GATE_OUT'
        return 'GATE_IN'

    def get_existing_folder(self, event_id: str) -> Optional[Path]:
        """Find an existing evidence folder for this event_id across GATE_IN and GATE_OUT."""
        cached = self._folder_cache.get(event_id)
        if cached and cached.is_dir():
            return cached

        # Check GATE_IN and GATE_OUT
        for gate_type in ('GATE_IN', 'GATE_OUT'):
            gate_dir = self.evidence_root / gate_type
            if not gate_dir.is_dir():
                continue
            for entry in gate_dir.iterdir():
                if entry.is_dir() and (entry.name == event_id or entry.name.startswith(f"{event_id}_")):
                    self._folder_cache[event_id] = entry
                    return entry

        # Check legacy snapshots root
        legacy_dir = self.legacy_root / event_id
        if legacy_dir.is_dir():
            real_dir = legacy_dir.resolve()
            if real_dir.is_dir():
                self._folder_cache[event_id] = real_dir
                return real_dir
            return legacy_dir

        return None

    def _relpath(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.evidence_root)).replace('\\', '/')
        except ValueError:
            try:
                return str(path.relative_to(self.legacy_root)).replace('\\', '/')
            except ValueError:
                return path.name

    def get_or_create_folder(self, event_id: str, gate_id: str = '', direction: str = '',
                             container_number: Optional[str] = None) -> Path:
        """
        Get or initialize the single evidence folder for this container event.
        Guarantees that exactly one folder is created for the container session.
        """
        existing = self.get_existing_folder(event_id)
        clean_num = str(container_number or '').strip().upper() if container_number else None

        if existing and existing.is_dir():
            # If folder currently lives inside legacy_root directly, migrate it into evidence_root
            if not existing.is_relative_to(self.evidence_root):
                gate_type = self.get_gate_type(gate_id, direction)
                target_name = f"{event_id}_{clean_num}" if clean_num else str(event_id)
                target_path = self.evidence_root / gate_type / target_name
                target_path.mkdir(parents=True, exist_ok=True)
                for item in list(existing.iterdir()):
                    shutil.move(str(item), str(target_path / item.name))
                try:
                    existing.rmdir()
                except OSError:
                    pass
                self._folder_cache[event_id] = target_path
                _create_junction_or_symlink(target_path, self.legacy_root / event_id)
                return target_path

            # If we now have a confirmed container number and the folder name doesn't have it yet, rename it!
            if clean_num and not existing.name.endswith(f"_{clean_num}"):
                new_name = f"{event_id}_{clean_num}"
                new_path = existing.parent / new_name
                if not new_path.exists():
                    try:
                        existing.rename(new_path)
                        self._folder_cache[event_id] = new_path
                        # Update legacy link
                        _create_junction_or_symlink(new_path, self.legacy_root / event_id)
                        return new_path
                    except Exception as exc:
                        log.debug('folder_rename_failed', extra={'error': str(exc)})
            return existing

        # Create new folder
        gate_type = self.get_gate_type(gate_id, direction)
        folder_name = f"{event_id}_{clean_num}" if clean_num else str(event_id)
        folder_path = self.evidence_root / gate_type / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        self._folder_cache[event_id] = folder_path

        # Link to legacy root so legacy tests and tools find it seamlessly
        _create_junction_or_symlink(folder_path, self.legacy_root / event_id)
        return folder_path

    def update_container_number(self, event_id: str, container_number: str) -> Optional[Path]:
        """Update folder name from <event_id> to <event_id>_<container_number> upon verification."""
        clean_num = str(container_number or '').strip().upper()
        if not clean_num:
            return None
        folder = self.get_existing_folder(event_id)
        if not folder or not folder.is_dir():
            return None
        if folder.name.endswith(f"_{clean_num}"):
            return folder

        # Ensure folder is under evidence_root
        if not folder.is_relative_to(self.evidence_root):
            gate_type = "GATE_IN"
            target_path = self.evidence_root / gate_type / f"{event_id}_{clean_num}"
            target_path.mkdir(parents=True, exist_ok=True)
            for item in list(folder.iterdir()):
                shutil.move(str(item), str(target_path / item.name))
            try:
                folder.rmdir()
            except OSError:
                pass
            self._folder_cache[event_id] = target_path
            _create_junction_or_symlink(target_path, self.legacy_root / event_id)
            return target_path

        new_name = f"{event_id}_{clean_num}"
        new_path = folder.parent / new_name
        if not new_path.exists():
            try:
                folder.rename(new_path)
                self._folder_cache[event_id] = new_path
                _create_junction_or_symlink(new_path, self.legacy_root / event_id)
                return new_path
            except Exception as exc:
                log.debug('folder_update_rename_failed', extra={'error': str(exc)})
                return folder
        return new_path

    def merge_event_folders(self, source_event_id: str, target_event_id: str,
                            container_number: Optional[str] = None) -> Optional[Path]:
        """
        Merge all files from source_event_id into target_event_id folder.
        Guarantees that when draft events are absorbed into a finalized passage session,
        all captured frames, best frames, and proof images are preserved in the single folder.
        """
        if not source_event_id or not target_event_id or source_event_id == target_event_id:
            return None
        source_folder = self.get_existing_folder(source_event_id)
        target_folder = self.get_or_create_folder(target_event_id, container_number=container_number)
        if not source_folder or not source_folder.is_dir() or source_folder == target_folder:
            return target_folder

        try:
            for item in list(source_folder.iterdir()):
                target_item = target_folder / item.name
                if not target_item.exists():
                    shutil.move(str(item), str(target_item))
                elif item.is_file():
                    item.unlink(missing_ok=True)
            try:
                source_folder.rmdir()
            except OSError:
                pass
            self._folder_cache[source_event_id] = target_folder
            _create_junction_or_symlink(target_folder, self.legacy_root / source_event_id)
            return target_folder
        except Exception as exc:
            log.debug('merge_event_folders_error', extra={'error': str(exc)})
            return target_folder

    def save_captured_frame(self, event_id: str, camera_id: str, image: np.ndarray,
                            frame_sequence: int, gate_id: str = '', direction: str = '',
                            container_number: Optional[str] = None) -> str:
        """
        Store an eligible captured frame from any of the gate cameras directly into the single evidence folder.
        Filename format: <camera_id>_frame_<seq:03d>.jpg
        """
        folder = self.get_or_create_folder(event_id, gate_id=gate_id, direction=direction, container_number=container_number)
        filename = f"{camera_id}_frame_{int(frame_sequence):03d}.jpg"
        final_path = folder / filename

        ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise OSError(f'Failed to encode captured frame for camera {camera_id}')

        tmp_path = final_path.with_suffix('.tmp')
        try:
            tmp_path.write_bytes(encoded.tobytes())
            os.replace(tmp_path, final_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        # Also write into legacy link if not junction
        legacy_dir = self.legacy_root / event_id
        if legacy_dir.is_dir() and not (legacy_dir / filename).exists():
            try:
                (legacy_dir / filename).write_bytes(encoded.tobytes())
            except Exception:
                pass

        # Return relative path
        return self._relpath(final_path)

    def save_best_frame(self, event_id: str, camera_id: str, image: np.ndarray,
                        gate_id: str = '', direction: str = '',
                        container_number: Optional[str] = None,
                        quality_score: float = 0.0) -> Tuple[str, str]:
        """
        Save the independently evaluated best frame for a camera inside the single evidence folder.
        Saves both best_<camera_id>.jpg and <camera_id>.jpg (for backward compatibility).
        Returns (relative_best_path, relative_legacy_path).
        """
        folder = self.get_or_create_folder(event_id, gate_id=gate_id, direction=direction, container_number=container_number)

        best_filename = f"best_{camera_id}.jpg"
        legacy_filename = f"{camera_id}.jpg"

        best_path = folder / best_filename
        legacy_path = folder / legacy_filename

        ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise OSError(f'Failed to encode best frame for camera {camera_id}')

        data_bytes = encoded.tobytes()

        # Write best_<camera_id>.jpg
        tmp_best = best_path.with_suffix('.tmp')
        try:
            tmp_best.write_bytes(data_bytes)
            os.replace(tmp_best, best_path)
        finally:
            tmp_best.unlink(missing_ok=True)

        # Write <camera_id>.jpg
        tmp_legacy = legacy_path.with_suffix('.tmp')
        try:
            tmp_legacy.write_bytes(data_bytes)
            os.replace(tmp_legacy, legacy_path)
        finally:
            tmp_legacy.unlink(missing_ok=True)

        # Write quality metadata
        best_path.with_suffix('.quality').write_text(f"{quality_score:.2f}", encoding='utf-8')
        legacy_path.with_suffix('.quality').write_text(f"{quality_score:.2f}", encoding='utf-8')

        # Mirror in legacy folder
        legacy_dir = self.legacy_root / event_id
        if legacy_dir.is_dir():
            for fname in (best_filename, legacy_filename):
                try:
                    target = legacy_dir / fname
                    if not target.exists():
                        target.write_bytes(data_bytes)
                except Exception:
                    pass

        rel_best = self._relpath(best_path)
        rel_legacy = self._relpath(legacy_path)
        return rel_best, rel_legacy

    def save_crop(self, event_id: str, camera_id: str, crop: np.ndarray,
                  seq: int = 1, gate_id: str = '', direction: str = '',
                  container_number: Optional[str] = None) -> str:
        """Store OCR proof crop inside the single evidence folder."""
        folder = self.get_or_create_folder(event_id, gate_id=gate_id, direction=direction, container_number=container_number)
        filename = f"{camera_id}_crop_{int(seq):03d}.png"
        final_path = folder / filename

        ok, encoded = cv2.imencode('.png', crop, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not ok:
            raise OSError('Failed to encode crop')

        tmp_path = final_path.with_suffix('.tmp')
        try:
            tmp_path.write_bytes(encoded.tobytes())
            os.replace(tmp_path, final_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        return self._relpath(final_path)

    def save_event_json(self, event_id: str, event_data: dict, manifest_data: Optional[dict] = None) -> Tuple[Path, Path]:
        """Save event.json and evidence.json directly inside the single evidence folder."""
        gate_id = str(event_data.get('gate_id') or '')
        direction = str(event_data.get('event_type') or '')
        container_num = event_data.get('container_number')
        folder = self.get_or_create_folder(event_id, gate_id=gate_id, direction=direction, container_number=container_num)

        # 1. event.json
        event_path = folder / 'event.json'
        tmp_event = folder / 'event.json.tmp'
        try:
            with tmp_event.open('w', encoding='utf-8') as f:
                json.dump(event_data, f, indent=2, default=str)
            os.replace(tmp_event, event_path)
        finally:
            tmp_event.unlink(missing_ok=True)

        # 2. evidence.json
        manifest_path = folder / 'evidence.json'
        if manifest_data is not None:
            tmp_man = folder / 'evidence.json.tmp'
            try:
                with tmp_man.open('w', encoding='utf-8') as f:
                    json.dump(manifest_data, f, indent=2, default=str)
                os.replace(tmp_man, manifest_path)
            finally:
                tmp_man.unlink(missing_ok=True)

        # Mirror in legacy root for any tools reading snapshots/<event_id>/event.json
        legacy_dir = self.legacy_root / event_id
        if legacy_dir.is_dir():
            for fname, fpath in (('event.json', event_path), ('evidence.json', manifest_path)):
                try:
                    if fpath.exists():
                        (legacy_dir / fname).write_bytes(fpath.read_bytes())
                except Exception:
                    pass

        return event_path, manifest_path

    def resolve_path(self, relative: str | Path) -> Path:
        """
        Locate any file whether referenced as:
        - GATE_IN/<folder>/<file>
        - GATE_OUT/<folder>/<file>
        - <event_id>/<file>
        - <file>
        """
        rel_str = str(relative).replace('\\', '/')
        if rel_str.startswith('/'):
            rel_str = rel_str.lstrip('/')

        # 1. Direct match in evidence_root
        cand = (self.evidence_root / rel_str).resolve()
        if cand.is_file():
            return cand

        # 2. Direct match in legacy_root
        cand_legacy = (self.legacy_root / rel_str).resolve()
        if cand_legacy.is_file():
            return cand_legacy

        # 3. If path is <event_id>/<filename>:
        parts = rel_str.split('/')
        if len(parts) == 2:
            event_id, fname = parts[0], parts[1]
            folder = self.get_existing_folder(event_id)
            if folder and (folder / fname).is_file():
                return folder / fname
            # Look for best_<fname> or vice versa
            if folder:
                alt = f"best_{fname}" if not fname.startswith('best_') else fname[5:]
                if (folder / alt).is_file():
                    return folder / alt

        # 4. If path is GATE_IN/<folder>/<fname> or GATE_OUT/<folder>/<fname>:
        if len(parts) >= 3 and parts[0] in ('GATE_IN', 'GATE_OUT'):
            folder_name = parts[1]
            fname = parts[-1]
            event_id = folder_name.split('_')[0]
            # Search in both gates
            for g in ('GATE_IN', 'GATE_OUT'):
                for sub in (self.evidence_root / g).glob(f"{event_id}*"):
                    if sub.is_dir() and (sub / fname).is_file():
                        return sub / fname

        # 5. Default resolution against evidence_root or legacy_root
        p = (self.evidence_root / rel_str).resolve()
        if p.is_relative_to(self.evidence_root):
            return p
        return (self.legacy_root / rel_str).resolve()
