"""
Persistence Layer Module for Llmcode

This module provides a comprehensive persistence layer for storing and retrieving
analysis results, supporting multiple storage backends and efficient caching.
"""

import os
import sys
import json
import time
import pickle
import sqlite3
import hashlib
import threading
from typing import Dict, List, Set, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import contextmanager
import logging
from abc import ABC, abstractmethod

from .static_analysis import StaticAnalyzer
from .code_structure import CodeStructureAnalyzer
from .relationship_mapper import RelationshipMapper
from .dependency_graph import DependencyGraphAnalyzer


@dataclass
class AnalysisMetadata:
    """Metadata for analysis results"""
    file_path: str
    analysis_type: str
    timestamp: float
    file_hash: str
    file_size: int
    analysis_version: str = "1.0"
    language: str = "unknown"
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisMetadata':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class AnalysisResult:
    """Complete analysis result with metadata"""
    metadata: AnalysisMetadata
    result: Dict[str, Any]
    cache_key: str = ""
    expiration_time: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if result is expired"""
        if self.expiration_time is None:
            return False
        return time.time() > self.expiration_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'metadata': self.metadata.to_dict(),
            'result': self.result,
            'cache_key': self.cache_key,
            'expiration_time': self.expiration_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisResult':
        """Create from dictionary"""
        metadata = AnalysisMetadata.from_dict(data['metadata'])
        return cls(
            metadata=metadata,
            result=data['result'],
            cache_key=data.get('cache_key', ''),
            expiration_time=data.get('expiration_time')
        )


class StorageBackend(ABC):
    """Abstract base class for storage backends"""
    
    @abstractmethod
    def store_result(self, result: AnalysisResult) -> bool:
        """Store analysis result"""
        pass
    
    @abstractmethod
    def retrieve_result(self, cache_key: str) -> Optional[AnalysisResult]:
        """Retrieve analysis result by cache key"""
        pass
    
    @abstractmethod
    def delete_result(self, cache_key: str) -> bool:
        """Delete analysis result by cache key"""
        pass
    
    @abstractmethod
    def list_results(self, analysis_type: Optional[str] = None) -> List[str]:
        """List all result cache keys, optionally filtered by analysis type"""
        pass
    
    @abstractmethod
    def clear_all(self) -> bool:
        """Clear all stored results"""
        pass
    
    @abstractmethod
    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage backend information"""
        pass


class FileStorageBackend(StorageBackend):
    """File-based storage backend"""
    
    def __init__(self, storage_dir: str = ".llmcode_cache"):
        """
        Initialize file storage backend
        
        Args:
            storage_dir: Directory to store cache files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.lock = threading.Lock()
        
        # Create subdirectories
        (self.storage_dir / "metadata").mkdir(exist_ok=True)
        (self.storage_dir / "results").mkdir(exist_ok=True)
        (self.storage_dir / "indexes").mkdir(exist_ok=True)
    
    def _get_cache_file_path(self, cache_key: str, file_type: str) -> Path:
        """Get file path for cache key and file type"""
        # Use first 2 characters of hash for subdirectory structure
        hash_part = hashlib.md5(cache_key.encode()).hexdigest()
        subdir = hash_part[:2]
        
        if file_type == "metadata":
            return self.storage_dir / "metadata" / subdir / f"{cache_key}.meta"
        elif file_type == "result":
            return self.storage_dir / "results" / subdir / f"{cache_key}.result"
        elif file_type == "index":
            return self.storage_dir / "indexes" / f"{cache_key}.idx"
        else:
            raise ValueError(f"Unknown file type: {file_type}")
    
    def _ensure_directory_exists(self, file_path: Path):
        """Ensure directory exists for file path"""
        file_path.parent.mkdir(parents=True, exist_ok=True)
    
    def store_result(self, result: AnalysisResult) -> bool:
        """Store analysis result"""
        try:
            with self.lock:
                # Store metadata
                meta_path = self._get_cache_file_path(result.cache_key, "metadata")
                self._ensure_directory_exists(meta_path)
                
                with open(meta_path, 'w') as f:
                    json.dump(result.metadata.to_dict(), f)
                
                # Store result
                result_path = self._get_cache_file_path(result.cache_key, "result")
                self._ensure_directory_exists(result_path)
                
                with open(result_path, 'wb') as f:
                    pickle.dump(result.result, f)
                
                # Update index
                index_path = self._get_cache_file_path(result.cache_key, "index")
                self._ensure_directory_exists(index_path)
                
                index_data = {
                    'cache_key': result.cache_key,
                    'file_path': result.metadata.file_path,
                    'analysis_type': result.metadata.analysis_type,
                    'timestamp': result.metadata.timestamp,
                    'file_hash': result.metadata.file_hash
                }
                
                with open(index_path, 'w') as f:
                    json.dump(index_data, f)
                
                return True
                
        except Exception:
            return False
    
    def retrieve_result(self, cache_key: str) -> Optional[AnalysisResult]:
        """Retrieve analysis result by cache key"""
        try:
            with self.lock:
                # Load metadata
                meta_path = self._get_cache_file_path(cache_key, "metadata")
                if not meta_path.exists():
                    return None
                
                with open(meta_path, 'r') as f:
                    metadata = AnalysisMetadata.from_dict(json.load(f))
                
                # Load result
                result_path = self._get_cache_file_path(cache_key, "result")
                if not result_path.exists():
                    return None
                
                with open(result_path, 'rb') as f:
                    result = pickle.load(f)
                
                return AnalysisResult(
                    metadata=metadata,
                    result=result,
                    cache_key=cache_key
                )
                
        except Exception:
            return None
    
    def delete_result(self, cache_key: str) -> bool:
        """Delete analysis result by cache key"""
        try:
            with self.lock:
                files_to_delete = [
                    self._get_cache_file_path(cache_key, "metadata"),
                    self._get_cache_file_path(cache_key, "result"),
                    self._get_cache_file_path(cache_key, "index")
                ]
                
                deleted = False
                for file_path in files_to_delete:
                    if file_path.exists():
                        file_path.unlink()
                        deleted = True
                
                return deleted
                
        except Exception:
            return False
    
    def list_results(self, analysis_type: Optional[str] = None) -> List[str]:
        """List all result cache keys, optionally filtered by analysis type"""
        try:
            cache_keys = []
            
            # Scan index files
            for index_file in self.storage_dir.glob("indexes/*/*.idx"):
                try:
                    with open(index_file, 'r') as f:
                        index_data = json.load(f)
                    
                    if analysis_type is None or index_data.get('analysis_type') == analysis_type:
                        cache_keys.append(index_data['cache_key'])
                        
                except Exception:
                    continue
            
            return cache_keys
            
        except Exception:
            return []
    
    def clear_all(self) -> bool:
        """Clear all stored results"""
        try:
            with self.lock:
                # Remove all files in subdirectories
                for subdir in ["metadata", "results", "indexes"]:
                    subdir_path = self.storage_dir / subdir
                    if subdir_path.exists():
                        for file_path in subdir_path.rglob("*"):
                            if file_path.is_file():
                                file_path.unlink()
                
                return True
                
        except Exception:
            return False
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage backend information"""
        try:
            total_files = 0
            total_size = 0
            
            for subdir in ["metadata", "results", "indexes"]:
                subdir_path = self.storage_dir / subdir
                if subdir_path.exists():
                    for file_path in subdir_path.rglob("*"):
                        if file_path.is_file():
                            total_files += 1
                            total_size += file_path.stat().st_size
            
            return {
                'backend_type': 'file',
                'storage_dir': str(self.storage_dir),
                'total_files': total_files,
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024)
            }
            
        except Exception:
            return {'backend_type': 'file', 'error': 'Failed to get storage info'}


class SQLiteStorageBackend(StorageBackend):
    """SQLite-based storage backend"""
    
    def __init__(self, db_path: str = ".llmcode_cache.db"):
        """
        Initialize SQLite storage backend
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                
                # Create tables
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS analysis_metadata (
                        cache_key TEXT PRIMARY KEY,
                        file_path TEXT NOT NULL,
                        analysis_type TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        file_hash TEXT NOT NULL,
                        file_size INTEGER NOT NULL,
                        analysis_version TEXT,
                        language TEXT,
                        dependencies TEXT,
                        tags TEXT
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS analysis_results (
                        cache_key TEXT PRIMARY KEY,
                        result_data BLOB NOT NULL,
                        expiration_time REAL,
                        FOREIGN KEY (cache_key) REFERENCES analysis_metadata(cache_key)
                    )
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_analysis_type ON analysis_metadata(analysis_type)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_file_path ON analysis_metadata(file_path)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_timestamp ON analysis_metadata(timestamp)
                ''')
                
                conn.commit()
                
            finally:
                conn.close()
    
    def store_result(self, result: AnalysisResult) -> bool:
        """Store analysis result"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                try:
                    cursor = conn.cursor()
                    
                    # Store metadata
                    cursor.execute('''
                        INSERT OR REPLACE INTO analysis_metadata
                        (cache_key, file_path, analysis_type, timestamp, file_hash, file_size, 
                         analysis_version, language, dependencies, tags)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        result.cache_key,
                        result.metadata.file_path,
                        result.metadata.analysis_type,
                        result.metadata.timestamp,
                        result.metadata.file_hash,
                        result.metadata.file_size,
                        result.metadata.analysis_version,
                        result.metadata.language,
                        json.dumps(result.metadata.dependencies),
                        json.dumps(result.metadata.tags)
                    ))
                    
                    # Store result
                    cursor.execute('''
                        INSERT OR REPLACE INTO analysis_results
                        (cache_key, result_data, expiration_time)
                        VALUES (?, ?, ?)
                    ''', (
                        result.cache_key,
                        pickle.dumps(result.result),
                        result.expiration_time
                    ))
                    
                    conn.commit()
                    return True
                    
                finally:
                    conn.close()
                    
        except Exception:
            return False
    
    def retrieve_result(self, cache_key: str) -> Optional[AnalysisResult]:
        """Retrieve analysis result by cache key"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                try:
                    cursor = conn.cursor()
                    
                    # Get metadata
                    cursor.execute('''
                        SELECT file_path, analysis_type, timestamp, file_hash, file_size,
                               analysis_version, language, dependencies, tags
                        FROM analysis_metadata WHERE cache_key = ?
                    ''', (cache_key,))
                    
                    row = cursor.fetchone()
                    if not row:
                        return None
                    
                    metadata = AnalysisMetadata(
                        file_path=row[0],
                        analysis_type=row[1],
                        timestamp=row[2],
                        file_hash=row[3],
                        file_size=row[4],
                        analysis_version=row[5],
                        language=row[6],
                        dependencies=json.loads(row[7] or '[]'),
                        tags=json.loads(row[8] or '[]')
                    )
                    
                    # Get result
                    cursor.execute('''
                        SELECT result_data, expiration_time
                        FROM analysis_results WHERE cache_key = ?
                    ''', (cache_key,))
                    
                    row = cursor.fetchone()
                    if not row:
                        return None
                    
                    result = pickle.loads(row[0])
                    expiration_time = row[1]
                    
                    return AnalysisResult(
                        metadata=metadata,
                        result=result,
                        cache_key=cache_key,
                        expiration_time=expiration_time
                    )
                    
                finally:
                    conn.close()
                    
        except Exception:
            return None
    
    def delete_result(self, cache_key: str) -> bool:
        """Delete analysis result by cache key"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                try:
                    cursor = conn.cursor()
                    
                    # Delete from both tables
                    cursor.execute('DELETE FROM analysis_results WHERE cache_key = ?', (cache_key,))
                    cursor.execute('DELETE FROM analysis_metadata WHERE cache_key = ?', (cache_key,))
                    
                    conn.commit()
                    return cursor.rowcount > 0
                    
                finally:
                    conn.close()
                    
        except Exception:
            return False
    
    def list_results(self, analysis_type: Optional[str] = None) -> List[str]:
        """List all result cache keys, optionally filtered by analysis type"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                try:
                    cursor = conn.cursor()
                    
                    if analysis_type:
                        cursor.execute('SELECT cache_key FROM analysis_metadata WHERE analysis_type = ?', (analysis_type,))
                    else:
                        cursor.execute('SELECT cache_key FROM analysis_metadata')
                    
                    return [row[0] for row in cursor.fetchall()]
                    
                finally:
                    conn.close()
                    
        except Exception:
            return []
    
    def clear_all(self) -> bool:
        """Clear all stored results"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                try:
                    cursor = conn.cursor()
                    
                    cursor.execute('DELETE FROM analysis_results')
                    cursor.execute('DELETE FROM analysis_metadata')
                    
                    conn.commit()
                    return True
                    
                finally:
                    conn.close()
                    
        except Exception:
            return False
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage backend information"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                try:
                    cursor = conn.cursor()
                    
                    # Get counts
                    cursor.execute('SELECT COUNT(*) FROM analysis_metadata')
                    metadata_count = cursor.fetchone()[0]
                    
                    cursor.execute('SELECT COUNT(*) FROM analysis_results')
                    results_count = cursor.fetchone()[0]
                    
                    # Get database file size
                    db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                    
                    return {
                        'backend_type': 'sqlite',
                        'db_path': self.db_path,
                        'metadata_count': metadata_count,
                        'results_count': results_count,
                        'db_size_bytes': db_size,
                        'db_size_mb': db_size / (1024 * 1024)
                    }
                    
                finally:
                    conn.close()
                    
        except Exception:
            return {'backend_type': 'sqlite', 'error': 'Failed to get storage info'}


class AnalysisCache:
    """High-level cache for analysis results"""
    
    def __init__(self,
                 storage_backend: StorageBackend,
                 max_memory_entries: int = 1000,
                 default_ttl_hours: int = 24,
                 enable_compression: bool = True):
        """
        Initialize analysis cache
        
        Args:
            storage_backend: Storage backend to use
            max_memory_entries: Maximum number of entries to keep in memory
            default_ttl_hours: Default time-to-live in hours
            enable_compression: Whether to enable compression for large results
        """
        self.storage_backend = storage_backend
        self.max_memory_entries = max_memory_entries
        self.default_ttl_seconds = default_ttl_hours * 3600
        self.enable_compression = enable_compression
        
        # In-memory cache
        self.memory_cache: Dict[str, AnalysisResult] = {}
        self.access_times: Dict[str, float] = {}
        self.lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'stores': 0,
            'evictions': 0,
            'expirations': 0
        }
    
    def _generate_cache_key(self, file_path: str, analysis_type: str, file_hash: str) -> str:
        """Generate cache key for file and analysis type"""
        key_data = f"{file_path}:{analysis_type}:{file_hash}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _evict_lru_entries(self, count: int = 1):
        """Evict least recently used entries"""
        if not self.memory_cache:
            return
        
        # Sort by access time
        sorted_entries = sorted(self.access_times.items(), key=lambda x: x[1])
        
        for i in range(min(count, len(sorted_entries))):
            cache_key, _ = sorted_entries[i]
            if cache_key in self.memory_cache:
                del self.memory_cache[cache_key]
                del self.access_times[cache_key]
                self.stats['evictions'] += 1
    
    def _cleanup_expired_entries(self):
        """Clean up expired entries"""
        current_time = time.time()
        expired_keys = []
        
        for cache_key, result in self.memory_cache.items():
            if result.is_expired():
                expired_keys.append(cache_key)
        
        for cache_key in expired_keys:
            del self.memory_cache[cache_key]
            del self.access_times[cache_key]
            self.stats['expirations'] += 1
    
    def store_result(self, result: AnalysisResult) -> bool:
        """Store analysis result"""
        try:
            with self.lock:
                # Set expiration time if not set
                if result.expiration_time is None:
                    result.expiration_time = time.time() + self.default_ttl_seconds
                
                # Clean up expired entries
                self._cleanup_expired_entries()
                
                # Evict entries if memory cache is full
                if len(self.memory_cache) >= self.max_memory_entries:
                    self._evict_lru_entries(5)  # Evict 5 entries
                
                # Store in memory cache
                self.memory_cache[result.cache_key] = result
                self.access_times[result.cache_key] = time.time()
                
                # Store in persistent storage
                success = self.storage_backend.store_result(result)
                if success:
                    self.stats['stores'] += 1
                
                return success
                
        except Exception:
            return False
    
    def retrieve_result(self, file_path: str, analysis_type: str, file_hash: str) -> Optional[AnalysisResult]:
        """Retrieve analysis result"""
        try:
            cache_key = self._generate_cache_key(file_path, analysis_type, file_hash)
            
            with self.lock:
                # Check memory cache first
                if cache_key in self.memory_cache:
                    result = self.memory_cache[cache_key]
                    
                    # Check if expired
                    if result.is_expired():
                        del self.memory_cache[cache_key]
                        del self.access_times[cache_key]
                        self.stats['expirations'] += 1
                    else:
                        # Update access time
                        self.access_times[cache_key] = time.time()
                        self.stats['hits'] += 1
                        return result
                
                # Check persistent storage
                result = self.storage_backend.retrieve_result(cache_key)
                if result:
                    # Check if expired
                    if result.is_expired():
                        self.storage_backend.delete_result(cache_key)
                        self.stats['expirations'] += 1
                    else:
                        # Add to memory cache
                        self.memory_cache[cache_key] = result
                        self.access_times[cache_key] = time.time()
                        self.stats['hits'] += 1
                        return result
                
                self.stats['misses'] += 1
                return None
                
        except Exception:
            self.stats['misses'] += 1
            return None
    
    def delete_result(self, file_path: str, analysis_type: str, file_hash: str) -> bool:
        """Delete analysis result"""
        try:
            cache_key = self._generate_cache_key(file_path, analysis_type, file_hash)
            
            with self.lock:
                # Remove from memory cache
                if cache_key in self.memory_cache:
                    del self.memory_cache[cache_key]
                    del self.access_times[cache_key]
                
                # Remove from persistent storage
                return self.storage_backend.delete_result(cache_key)
                
        except Exception:
            return False
    
    def clear_all(self) -> bool:
        """Clear all cached results"""
        try:
            with self.lock:
                # Clear memory cache
                self.memory_cache.clear()
                self.access_times.clear()
                
                # Clear persistent storage
                return self.storage_backend.clear_all()
                
        except Exception:
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            hit_rate = self.stats['hits'] / max(1, self.stats['hits'] + self.stats['misses'])
            
            return {
                'memory_cache_size': len(self.memory_cache),
                'max_memory_entries': self.max_memory_entries,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'hit_rate': hit_rate,
                'stores': self.stats['stores'],
                'evictions': self.stats['evictions'],
                'expirations': self.stats['expirations'],
                'storage_info': self.storage_backend.get_storage_info()
            }
    
    def cleanup_expired(self) -> int:
        """Clean up expired entries and return count of cleaned entries"""
        try:
            with self.lock:
                cleaned_count = 0
                
                # Clean memory cache
                current_time = time.time()
                expired_keys = []
                
                for cache_key, result in self.memory_cache.items():
                    if result.is_expired():
                        expired_keys.append(cache_key)
                
                for cache_key in expired_keys:
                    del self.memory_cache[cache_key]
                    del self.access_times[cache_key]
                    cleaned_count += 1
                
                # Clean persistent storage
                all_keys = self.storage_backend.list_results()
                for cache_key in all_keys:
                    result = self.storage_backend.retrieve_result(cache_key)
                    if result and result.is_expired():
                        self.storage_backend.delete_result(cache_key)
                        cleaned_count += 1
                
                self.stats['expirations'] += cleaned_count
                return cleaned_count
                
        except Exception:
            return 0


class PersistenceManager:
    """High-level manager for analysis result persistence"""
    
    def __init__(self,
                 storage_type: str = "sqlite",
                 storage_path: str = None,
                 max_memory_entries: int = 1000,
                 default_ttl_hours: int = 24,
                 auto_cleanup: bool = True,
                 cleanup_interval_hours: int = 6):
        """
        Initialize persistence manager
        
        Args:
            storage_type: Type of storage backend ('file' or 'sqlite')
            storage_path: Path for storage (optional)
            max_memory_entries: Maximum memory cache entries
            default_ttl_hours: Default TTL in hours
            auto_cleanup: Whether to enable automatic cleanup
            cleanup_interval_hours: Interval for automatic cleanup
        """
        self.storage_type = storage_type
        self.auto_cleanup = auto_cleanup
        self.cleanup_interval_seconds = cleanup_interval_hours * 3600
        
        # Initialize storage backend
        if storage_type == "file":
            self.storage_backend = FileStorageBackend(storage_path or ".llmcode_cache")
        elif storage_type == "sqlite":
            self.storage_backend = SQLiteStorageBackend(storage_path or ".llmcode_cache.db")
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")
        
        # Initialize cache
        self.cache = AnalysisCache(
            storage_backend=self.storage_backend,
            max_memory_entries=max_memory_entries,
            default_ttl_hours=default_ttl_hours
        )
        
        # Start auto-cleanup if enabled
        self.cleanup_thread = None
        if auto_cleanup:
            self.start_auto_cleanup()
    
    def start_auto_cleanup(self):
        """Start automatic cleanup thread"""
        if self.cleanup_thread is not None:
            return
        
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def stop_auto_cleanup(self):
        """Stop automatic cleanup thread"""
        if self.cleanup_thread is None:
            return
        
        # Thread is daemon, so it will be stopped automatically
        self.cleanup_thread = None
    
    def _cleanup_loop(self):
        """Automatic cleanup loop"""
        while True:
            try:
                time.sleep(self.cleanup_interval_seconds)
                cleaned_count = self.cache.cleanup_expired()
                if cleaned_count > 0:
                    logging.info(f"Auto-cleanup: cleaned {cleaned_count} expired entries")
            except Exception:
                time.sleep(self.cleanup_interval_seconds)
    
    def store_analysis_result(self,
                            file_path: str,
                            analysis_type: str,
                            result: Dict[str, Any],
                            file_hash: str = None,
                            ttl_hours: int = None) -> bool:
        """
        Store analysis result
        
        Args:
            file_path: Path to the analyzed file
            analysis_type: Type of analysis performed
            result: Analysis result data
            file_hash: Hash of the file content (optional)
            ttl_hours: Time-to-live in hours (optional)
            
        Returns:
            Success status
        """
        try:
            # Generate file hash if not provided
            if file_hash is None:
                try:
                    with open(file_path, 'rb') as f:
                        file_content = f.read()
                        file_hash = hashlib.md5(file_content).hexdigest()
                except Exception:
                    file_hash = "unknown"
            
            # Get file size
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            
            # Create metadata
            metadata = AnalysisMetadata(
                file_path=file_path,
                analysis_type=analysis_type,
                timestamp=time.time(),
                file_hash=file_hash,
                file_size=file_size,
                language=self._detect_language(file_path)
            )
            
            # Create result
            cache_key = self.cache._generate_cache_key(file_path, analysis_type, file_hash)
            analysis_result = AnalysisResult(
                metadata=metadata,
                result=result,
                cache_key=cache_key
            )
            
            # Set TTL if provided
            if ttl_hours is not None:
                analysis_result.expiration_time = time.time() + (ttl_hours * 3600)
            
            # Store result
            return self.cache.store_result(analysis_result)
            
        except Exception:
            return False
    
    def retrieve_analysis_result(self,
                               file_path: str,
                               analysis_type: str,
                               file_hash: str = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve analysis result
        
        Args:
            file_path: Path to the analyzed file
            analysis_type: Type of analysis performed
            file_hash: Hash of the file content (optional)
            
        Returns:
            Analysis result or None if not found
        """
        try:
            # Generate file hash if not provided
            if file_hash is None:
                try:
                    with open(file_path, 'rb') as f:
                        file_content = f.read()
                        file_hash = hashlib.md5(file_content).hexdigest()
                except Exception:
                    file_hash = "unknown"
            
            # Retrieve result
            result = self.cache.retrieve_result(file_path, analysis_type, file_hash)
            return result.result if result else None
            
        except Exception:
            return None
    
    def delete_analysis_result(self,
                             file_path: str,
                             analysis_type: str,
                             file_hash: str = None) -> bool:
        """
        Delete analysis result
        
        Args:
            file_path: Path to the analyzed file
            analysis_type: Type of analysis performed
            file_hash: Hash of the file content (optional)
            
        Returns:
            Success status
        """
        try:
            # Generate file hash if not provided
            if file_hash is None:
                try:
                    with open(file_path, 'rb') as f:
                        file_content = f.read()
                        file_hash = hashlib.md5(file_content).hexdigest()
                except Exception:
                    file_hash = "unknown"
            
            return self.cache.delete_result(file_path, analysis_type, file_hash)
            
        except Exception:
            return False
    
    def list_analysis_results(self, analysis_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List analysis results
        
        Args:
            analysis_type: Filter by analysis type (optional)
            
        Returns:
            List of analysis result metadata
        """
        try:
            cache_keys = self.storage_backend.list_results(analysis_type)
            results = []
            
            for cache_key in cache_keys:
                result = self.storage_backend.retrieve_result(cache_key)
                if result:
                    results.append({
                        'cache_key': cache_key,
                        'file_path': result.metadata.file_path,
                        'analysis_type': result.metadata.analysis_type,
                        'timestamp': result.metadata.timestamp,
                        'file_hash': result.metadata.file_hash,
                        'file_size': result.metadata.file_size,
                        'language': result.metadata.language
                    })
            
            return results
            
        except Exception:
            return []
    
    def clear_all_results(self) -> bool:
        """Clear all analysis results"""
        return self.cache.clear_all()
    
    def get_persistence_stats(self) -> Dict[str, Any]:
        """Get comprehensive persistence statistics"""
        return {
            'cache_stats': self.cache.get_cache_stats(),
            'storage_type': self.storage_type,
            'auto_cleanup': self.auto_cleanup,
            'cleanup_interval_hours': self.cleanup_interval_seconds / 3600
        }
    
    def cleanup_expired_results(self) -> int:
        """Clean up expired results and return count"""
        return self.cache.cleanup_expired()
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        ext = Path(file_path).suffix.lower()
        
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.cxx': 'cpp',
            '.cc': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.m': 'objective-c',
            '.mm': 'objective-c++',
            '.sh': 'shell',
            '.bash': 'shell',
            '.zsh': 'shell',
            '.fish': 'shell',
            '.sql': 'sql',
            '.html': 'html',
            '.css': 'css',
            '.scss': 'scss',
            '.sass': 'sass',
            '.less': 'less',
            '.xml': 'xml',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.toml': 'toml',
            '.ini': 'ini',
            '.cfg': 'ini',
            '.conf': 'ini',
            '.md': 'markdown',
            '.txt': 'text'
        }
        
        return language_map.get(ext, 'unknown')
    
    @contextmanager
    def transaction(self):
        """Context manager for transaction-like operations"""
        # For file-based storage, this is a no-op
        # For SQLite, transactions are handled automatically
        yield
    
    def export_results(self, export_path: str, analysis_type: Optional[str] = None) -> bool:
        """
        Export analysis results to file
        
        Args:
            export_path: Path to export file
            analysis_type: Filter by analysis type (optional)
            
        Returns:
            Success status
        """
        try:
            results = self.list_analysis_results(analysis_type)
            
            with open(export_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            return True
            
        except Exception:
            return False
    
    def import_results(self, import_path: str) -> bool:
        """
        Import analysis results from file
        
        Args:
            import_path: Path to import file
            
        Returns:
            Success status
        """
        try:
            with open(import_path, 'r') as f:
                results = json.load(f)
            
            for result_data in results:
                self.store_analysis_result(
                    file_path=result_data['file_path'],
                    analysis_type=result_data['analysis_type'],
                    result={},  # Empty result since we only have metadata
                    file_hash=result_data['file_hash']
                )
            
            return True
            
        except Exception:
            return False
