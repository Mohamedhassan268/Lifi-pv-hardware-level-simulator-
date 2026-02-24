# cosim/session.py
"""
Session directory manager.

Each simulation run gets its own session directory under workspace/,
containing config, netlists, PWL files, raw outputs, and plots.

Usage:
    from cosim.session import SessionManager

    sm = SessionManager()
    session_dir = sm.create_session()
    sm.save_config(session_dir, config)
    cfg = sm.load_config(session_dir)
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from .system_config import SystemConfig


# Default workspace directory (relative to project root)
_PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_WORKSPACE = _PROJECT_ROOT / 'workspace'


class SessionManager:
    """Manages simulation session directories."""

    def __init__(self, workspace: Optional[Path] = None):
        """
        Initialize session manager.

        Args:
            workspace: Root workspace directory.
                       Defaults to <project>/workspace/
        """
        self.workspace = Path(workspace) if workspace else DEFAULT_WORKSPACE

    def create_session(self, label: str = '') -> Path:
        """
        Create a new session directory.

        Directory name: session_YYYYMMDD_HHMMSS[_label]

        Args:
            label: Optional descriptive label appended to directory name

        Returns:
            Path to the new session directory
        """
        self.workspace.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name = f'session_{timestamp}'
        if label:
            safe_label = ''.join(c if c.isalnum() or c in '-_' else '_'
                                  for c in label)
            name = f'{name}_{safe_label}'

        session_dir = self.workspace / name
        session_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (session_dir / 'netlists').mkdir(exist_ok=True)
        (session_dir / 'pwl').mkdir(exist_ok=True)
        (session_dir / 'raw').mkdir(exist_ok=True)
        (session_dir / 'plots').mkdir(exist_ok=True)

        return session_dir

    def save_config(self, session_dir: Path, config: SystemConfig) -> Path:
        """
        Save config to session directory.

        Args:
            session_dir: Session directory path
            config: SystemConfig to save

        Returns:
            Path to saved config file
        """
        config_path = Path(session_dir) / 'config.json'
        config.save(config_path)
        return config_path

    def load_config(self, session_dir: Path) -> SystemConfig:
        """
        Load config from session directory.

        Args:
            session_dir: Session directory path

        Returns:
            SystemConfig loaded from session
        """
        config_path = Path(session_dir) / 'config.json'
        return SystemConfig.load(config_path)

    def list_sessions(self) -> List[Path]:
        """
        List all session directories, newest first.

        Returns:
            List of session directory paths
        """
        if not self.workspace.exists():
            return []
        sessions = sorted(
            [d for d in self.workspace.iterdir()
             if d.is_dir() and d.name.startswith('session_')],
            reverse=True
        )
        return sessions

    def get_latest_session(self) -> Optional[Path]:
        """Return most recent session directory, or None."""
        sessions = self.list_sessions()
        return sessions[0] if sessions else None

    def delete_session(self, session_dir: Path) -> None:
        """
        Delete a session directory and all contents.

        Args:
            session_dir: Session directory to delete
        """
        session_dir = Path(session_dir)
        if session_dir.exists() and session_dir.is_dir():
            shutil.rmtree(session_dir)

    def session_summary(self, session_dir: Path) -> dict:
        """
        Get a summary of session contents.

        Returns:
            Dict with counts: n_netlists, n_pwl, n_raw, n_plots, has_config
        """
        session_dir = Path(session_dir)
        return {
            'name': session_dir.name,
            'has_config': (session_dir / 'config.json').exists(),
            'n_netlists': len(list((session_dir / 'netlists').glob('*')))
                          if (session_dir / 'netlists').exists() else 0,
            'n_pwl': len(list((session_dir / 'pwl').glob('*')))
                     if (session_dir / 'pwl').exists() else 0,
            'n_raw': len(list((session_dir / 'raw').glob('*.raw')))
                     if (session_dir / 'raw').exists() else 0,
            'n_plots': len(list((session_dir / 'plots').glob('*')))
                       if (session_dir / 'plots').exists() else 0,
        }
