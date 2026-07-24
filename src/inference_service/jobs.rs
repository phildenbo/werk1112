use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use super::{
    helpers::{validate_safe_name, write_json_atomic},
    service::InferenceService,
    types::InferenceResult,
};
use crate::{inference::InferenceRequest, model_store::unix_ts};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Loading,
    Running,
    Encoding,
    Completed,
    Failed,
    Cancelled,
}

impl JobStatus {
    pub fn terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobRecord {
    pub id: String,
    pub status: JobStatus,
    pub request: InferenceRequest,
    pub result: Option<InferenceResult>,
    pub error: Option<String>,
    pub created_unix: u64,
    pub updated_unix: u64,
}

#[derive(Debug, Clone)]
pub struct JobStore {
    root: PathBuf,
    mutation_lock: Arc<Mutex<()>>,
}

impl JobStore {
    pub fn new(home: &Path) -> Self {
        Self {
            root: home.join("jobs"),
            mutation_lock: Arc::new(Mutex::new(())),
        }
    }

    pub fn ensure(&self) -> Result<()> {
        fs::create_dir_all(&self.root)?;
        Ok(())
    }

    pub fn create(&self, request: InferenceRequest) -> Result<JobRecord> {
        self.ensure()?;
        let now = unix_ts();
        let record = JobRecord {
            id: super::helpers::new_id("job")?,
            status: JobStatus::Queued,
            request,
            result: None,
            error: None,
            created_unix: now,
            updated_unix: now,
        };
        self.write(&record)?;
        Ok(record)
    }

    pub fn get(&self, id: &str) -> Result<JobRecord> {
        validate_safe_name(id)?;
        let path = self.root.join(format!("{id}.json"));
        let data = fs::read(&path).with_context(|| format!("job '{id}' was not found"))?;
        serde_json::from_slice(&data)
            .with_context(|| format!("invalid persisted job {}", path.display()))
    }

    pub fn list(&self) -> Result<Vec<JobRecord>> {
        self.ensure()?;
        let mut records = fs::read_dir(&self.root)?
            .filter_map(std::result::Result::ok)
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .and_then(|extension| extension.to_str())
                    == Some("json")
            })
            .filter_map(|entry| fs::read(entry.path()).ok())
            .filter_map(|data| serde_json::from_slice(&data).ok())
            .collect::<Vec<_>>();
        records.sort_by_key(|record: &JobRecord| record.created_unix);
        Ok(records)
    }

    pub fn transition(
        &self,
        id: &str,
        status: JobStatus,
        result: Option<InferenceResult>,
        error: Option<String>,
    ) -> Result<JobRecord> {
        let _guard = self
            .mutation_lock
            .lock()
            .map_err(|_| anyhow!("job store mutation lock is poisoned"))?;
        let mut record = self.get(id)?;
        if record.status.terminal() {
            if record.status == status {
                return Ok(record);
            }
            bail!(
                "job '{}' is already in terminal state {:?}",
                record.id,
                record.status
            );
        }
        if !valid_job_transition(record.status, status) {
            bail!("invalid job transition {:?} -> {:?}", record.status, status);
        }
        record.status = status;
        record.result = result;
        record.error = error;
        record.updated_unix = unix_ts();
        self.write(&record)?;
        Ok(record)
    }

    pub fn cancel(&self, id: &str) -> Result<JobRecord> {
        let _guard = self
            .mutation_lock
            .lock()
            .map_err(|_| anyhow!("job store mutation lock is poisoned"))?;
        let mut record = self.get(id)?;
        if record.status.terminal() {
            return Ok(record);
        }
        if !valid_job_transition(record.status, JobStatus::Cancelled) {
            bail!(
                "invalid job transition {:?} -> {:?}",
                record.status,
                JobStatus::Cancelled
            );
        }
        record.status = JobStatus::Cancelled;
        record.result = None;
        record.error = None;
        record.updated_unix = unix_ts();
        self.write(&record)?;
        Ok(record)
    }

    pub fn recover_interrupted(&self) -> Result<usize> {
        let records = self.list()?;
        let mut recovered = 0;
        for record in records
            .into_iter()
            .filter(|record| !record.status.terminal())
        {
            self.transition(
                &record.id,
                JobStatus::Failed,
                None,
                Some("job was interrupted by a Werk server restart".to_string()),
            )?;
            recovered += 1;
        }
        Ok(recovered)
    }

    fn write(&self, record: &JobRecord) -> Result<()> {
        validate_safe_name(&record.id)?;
        self.ensure()?;
        write_json_atomic(&self.root.join(format!("{}.json", record.id)), record)
    }
}

fn valid_job_transition(from: JobStatus, to: JobStatus) -> bool {
    use JobStatus::*;
    matches!(
        (from, to),
        (Queued, Loading | Running | Failed | Cancelled)
            | (Loading, Running | Failed | Cancelled)
            | (Running, Encoding | Completed | Failed | Cancelled)
            | (Encoding, Completed | Failed | Cancelled)
    )
}

#[derive(Clone)]
pub struct JobManager {
    service: InferenceService,
    store: JobStore,
}

impl JobManager {
    pub fn new(service: InferenceService) -> Self {
        let store = JobStore::new(service.store().home());
        if let Err(error) = store.recover_interrupted() {
            eprintln!("warning: failed to recover persisted media jobs: {error:#}");
        }
        Self { service, store }
    }

    pub fn store(&self) -> &JobStore {
        &self.store
    }

    pub fn submit(&self, request: InferenceRequest) -> Result<JobRecord> {
        let record = self.store.create(request)?;
        let job_id = record.id.clone();
        let manager = self.clone();
        tokio::spawn(async move {
            let current = manager.store.get(&job_id);
            if current
                .as_ref()
                .is_ok_and(|record| record.status == JobStatus::Cancelled)
            {
                return;
            }
            if manager
                .store
                .transition(&job_id, JobStatus::Loading, None, None)
                .is_err()
            {
                return;
            }
            if manager
                .store
                .transition(&job_id, JobStatus::Running, None, None)
                .is_err()
            {
                return;
            }
            let service = manager.service.clone();
            let request = match manager.store.get(&job_id) {
                Ok(record) => record.request,
                Err(_) => return,
            };
            let result = tokio::task::spawn_blocking(move || service.execute(request)).await;
            let cancelled = manager
                .store
                .get(&job_id)
                .is_ok_and(|record| record.status == JobStatus::Cancelled);
            if cancelled {
                if let Ok(Ok(result)) = &result {
                    let _ = manager.service.output_store().remove_result(&result.id);
                }
                return;
            }
            match result {
                Ok(Ok(result)) => {
                    let _ = manager
                        .store
                        .transition(&job_id, JobStatus::Encoding, None, None);
                    let _ =
                        manager
                            .store
                            .transition(&job_id, JobStatus::Completed, Some(result), None);
                }
                Ok(Err(error)) => {
                    let _ = manager.store.transition(
                        &job_id,
                        JobStatus::Failed,
                        None,
                        Some(error.to_string()),
                    );
                }
                Err(error) => {
                    let _ = manager.store.transition(
                        &job_id,
                        JobStatus::Failed,
                        None,
                        Some(format!("job worker failed: {error}")),
                    );
                }
            }
        });
        Ok(record)
    }
}
