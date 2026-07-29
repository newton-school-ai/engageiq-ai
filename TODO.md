# EngageIQ AI - GitHub Issue #19: Calibration System

## Plan (implementation checklist)

1. **Database persistence**
   - Add `src/models/calibration.py` with a per-student calibration table/model.
   - Register the model in `src/models/__init__.py` so Alembic can see metadata.
   - Add an Alembic migration to create the calibration table.

2. **Calibration core**
   - Create `src/scoring/calibration.py`:
     - `CalibrationManager`
     - calibration session handling (30s window)
     - threshold calculation:
       - `ear_threshold = resting_ear * 0.8`
       - gaze thresholds adjusted relative to natural head pose baseline
     - serialization/deserialization
     - database persistence and retrieval
     - allow recalibration at any time (upsert)

3. **API routes**
   - Create `src/api/routes/calibration.py`:
     - `POST /api/calibrate/{user_id}` (runs 30s calibration session, persists results)
     - `GET /api/calibrate/{user_id}` (returns calibration data / state)
   - Wire router in `src/api/main.py` under `/api/v1`.

4. **Pipeline integration**
   - Update existing detection/classification code to use calibrated thresholds when available:
     - drowsiness EAR threshold
     - head-pose baseline to compute calibrated gaze thresholds
     - fallback to defaults if calibration is skipped/not present
   - Ensure logging + validation + safe defaults.

5. **Tests**
   - Add comprehensive pytest coverage:
     - EAR threshold calculation
     - calibration state transitions / session handling
     - serialization/deserialization
     - recalibration (overwrite/upsert)
     - API endpoints (valid + invalid requests)
     - skipped calibration behavior + warning logging
     - database persistence/retrieval

## Progress
- [ ] Step 1: Add DB model + Alembic migration
- [ ] Step 2: Implement `CalibrationManager`
- [ ] Step 3: Implement API routes + wire router
- [ ] Step 4: Integrate thresholds into detection/scoring pipeline
- [ ] Step 5: Add tests
