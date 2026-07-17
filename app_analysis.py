# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Flask blueprint for launching library analysis and database cleaning.

Thin route layer that enqueues the long-running main tasks onto the high
priority RQ queue and returns their job id for the UI to poll via the generic
status routes in `app.py`.

Main Features:
* Routes: `/cleaning` page, `/api/analysis/start` (enqueues
  `tasks.analysis.run_analysis_task`) and `/api/cleaning/start`.
* Archives previously successful main tasks to REVOKED on a new start and
  guards against a second concurrent main task via `get_active_main_task`.
"""

from flask import Blueprint, jsonify, request, render_template
import uuid
import logging

# Import configuration from the main config.py
from config import NUM_RECENT_ALBUMS, TOP_N_MOODS, TASK_STATUS_PENDING

# RQ import
from rq import Retry

# App helper functions
from app_helper import rq_queue_high, save_task_status
from database import clean_up_previous_main_tasks, get_active_main_task

logger = logging.getLogger(__name__)

# Create a Blueprint for analysis-related routes
analysis_bp = Blueprint('analysis_bp', __name__)


@analysis_bp.route('/cleaning', methods=['GET'])
def cleaning_page():
    """
    Serves the HTML page for the Database Cleaning feature.
    ---
    tags:
      - UI
    responses:
      200:
        description: HTML content of the cleaning page.
        content:
          text/html:
            schema:
              type: string
    """
    return render_template(
        'cleaning.html', title='AudioMuse-AI - Database Cleaning', active='cleaning'
    )


@analysis_bp.route('/api/analysis/start', methods=['POST'])
def start_analysis_endpoint():
    """
    Start the music analysis process for recent albums.
    This endpoint enqueues a main analysis task.
    Note: Starting a new analysis task will archive previously successful tasks by setting their status to REVOKED.
    ---
    tags:
      - Analysis
    requestBody:
      description: Configuration for the analysis task.
      required: false
      content:
        application/json:
          schema:
            type: object
            properties:
              num_recent_albums:
                type: integer
                description: Number of recent albums to process.
                default: "Configured NUM_RECENT_ALBUMS"
              top_n_moods:
                type: integer
                description: Number of top moods to extract per track.
                default: "Configured TOP_N_MOODS"
    responses:
      202:
        description: Analysis task successfully enqueued.
        content:
          application/json:
            schema:
              type: object
              properties:
                task_id:
                  type: string
                  description: The ID of the enqueued main analysis task.
                task_type:
                  type: string
                  description: Type of the task (e.g., main_analysis).
                  example: main_analysis
                status:
                  type: string
                  description: The initial status of the job in the queue (e.g., queued).
      400:
        description: Invalid input.
      500:
        description: Server error during task enqueue.
    """
    # Check for any existing active main task to prevent parallel batch runs.
    active_task = get_active_main_task()
    if active_task:
        return jsonify(
            {
                "error": "An active batch task is already in progress.",
                "task_id": active_task['task_id'],
                "status": active_task['status'],
            }
        ), 409

    data = request.json or {}
    # MODIFIED: Removed jellyfin_url, jellyfin_user_id, and jellyfin_token as they are no longer passed to the task.
    # The task now gets these details from the central config.
    num_recent_albums = int(data.get('num_recent_albums', NUM_RECENT_ALBUMS))
    top_n_moods = int(data.get('top_n_moods', TOP_N_MOODS))
    logger.info(
        f"Starting analysis request: num_recent_albums={num_recent_albums}, top_n_moods={top_n_moods}"
    )

    job_id = str(uuid.uuid4())

    # Clean up details of previously successful or stale tasks before starting a new one
    clean_up_previous_main_tasks()
    save_task_status(
        job_id, "main_analysis", TASK_STATUS_PENDING, details={"message": "Task enqueued."}
    )

    # Enqueue task using a string path to its function.
    # MODIFIED: The arguments passed to the task are updated to match the new function signature.
    job = rq_queue_high.enqueue(
        'tasks.analysis.run_analysis_task',
        args=(num_recent_albums, top_n_moods),
        job_id=job_id,
        description="Main Music Analysis",
        retry=Retry(max=3),
        job_timeout=-1,  # No timeout
    )
    return jsonify(
        {"task_id": job.id, "task_type": "main_analysis", "status": job.get_status()}
    ), 202


@analysis_bp.route('/api/cleaning/start', methods=['POST'])
def start_cleaning_endpoint():
    """
    Identify and automatically clean orphaned albums from the database.
    This endpoint enqueues a cleaning task that both identifies and deletes orphaned albums.
    ---
    tags:
      - Cleaning
    responses:
      202:
        description: Database cleaning task successfully enqueued.
        content:
          application/json:
            schema:
              type: object
              properties:
                task_id:
                  type: string
                  description: The ID of the enqueued database cleaning task.
                task_type:
                  type: string
                  description: Type of the task (cleaning).
                  example: cleaning
                status:
                  type: string
                  description: The initial status of the job in the queue (e.g., queued).
      500:
        description: Server error during task enqueue.
    """
    # Cleaning is the ONE start that must also refuse while a sweep is running: both
    # prune track_server_map, each against a snapshot of the server's catalogue taken
    # minutes earlier, so an overlap lets cleaning delete the mappings the sweep just
    # wrote. Every other task type may run alongside a sweep, so they keep the
    # default exclusion.
    active_task = get_active_main_task(exclude_task_types=())
    if active_task:
        return jsonify(
            {
                "error": "An active batch task is already in progress.",
                "task_id": active_task['task_id'],
                "status": active_task['status'],
            }
        ), 409

    # Clean up any previous cleaning tasks
    clean_up_previous_main_tasks()

    job_id = str(uuid.uuid4())
    save_task_status(
        job_id,
        "cleaning",
        TASK_STATUS_PENDING,
        details={"message": "Database cleaning task enqueued."},
    )

    # Enqueue combined cleaning task
    job = rq_queue_high.enqueue(
        'tasks.cleaning.identify_and_clean_orphaned_albums_task',
        job_id=job_id,
        description="Database Cleaning (Identify and Delete Orphaned Albums)",
        retry=Retry(max=2),
        job_timeout=-1,  # No timeout
    )
    return jsonify({"task_id": job.id, "task_type": "cleaning", "status": job.get_status()}), 202


@analysis_bp.route('/api/cleaning/sonic_state', methods=['GET'])
def sonic_state_endpoint():
    """Report per-backend embedding + mood-centroid state for the Cleaning UI.

    Body shape (see ``tasks.cleaning.inspect_sonic_state``): ``active_backend``,
    ``active_dim``, and a ``backends`` list with one row per backend that has
    stored data (plus the active one). Each row carries ``embedding_row_count``,
    ``sample_stored_dim``, ``mood_centroid_count`` and ``is_active``.
    ---
    tags:
      - Cleaning
    responses:
      200:
        description: Per-backend sonic state snapshot.
    """
    from tasks.cleaning import inspect_sonic_state
    return jsonify(inspect_sonic_state()), 200


@analysis_bp.route('/api/cleaning/sonic_state/clear', methods=['POST'])
def sonic_state_clear_endpoint():
    """Drop one inactive backend's embedding + mood-centroid rows.

    Removes only the ``embedding`` rows whose ``backend`` column equals the
    supplied value and that backend's ``mood_centroids_data`` rows. The active
    ``SONIC_BACKEND`` is protected — switching ``SONIC_BACKEND`` first is
    required to clear it. Untouched: the IVF/CLAP/lyrics/artist data,
    playlists, app_config, task history, score rows.

    Body: ``{"backend": "musicnn", "confirm": true}``. The confirm flag is
    required so a stray fetch can't drop the rows.
    ---
    tags:
      - Cleaning
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              backend:
                type: string
              confirm:
                type: boolean
    responses:
      200:
        description: Cleared. Body returns the summary + refreshed state.
      400:
        description: Missing/invalid backend, missing confirmation, or
                     attempt to clear the active backend.
      500:
        description: Database error during cleanup.
    """
    data = request.json or {}
    backend = (data.get('backend') or '').strip()
    if not backend:
        return jsonify({"error": "Missing 'backend' field."}), 400
    if data.get('confirm') is not True:
        return jsonify({
            "error": "Refusing to clear without explicit confirmation.",
            "hint": 'POST {"backend": "<name>", "confirm": true} to proceed.',
        }), 400

    from tasks.cleaning import clear_inactive_backend_data, inspect_sonic_state
    try:
        summary = clear_inactive_backend_data(backend)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("clear_inactive_backend_data failed: %s", e)
        return jsonify({"error": str(e)}), 500
    return jsonify({"summary": summary, "state": inspect_sonic_state()}), 200
