-- 010. Training cancellation: allow CANCELLED for training jobs and GA experiments
--      (workflows already allow CANCELLED in 007)

ALTER TABLE training_jobs DROP CONSTRAINT IF EXISTS training_jobs_status_check;
ALTER TABLE training_jobs ADD CONSTRAINT training_jobs_status_check
    CHECK (status IN ('QUEUED', 'RUNNING', 'SUCCEEDED', 'FAILED', 'CANCELLED'));

ALTER TABLE ga_experiments DROP CONSTRAINT IF EXISTS ga_experiments_status_check;
ALTER TABLE ga_experiments ADD CONSTRAINT ga_experiments_status_check
    CHECK (status IN ('RUNNING', 'SUCCEEDED', 'FAILED', 'CANCELLED'));
