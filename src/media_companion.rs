use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::{
    collections::BTreeMap,
    env,
    ffi::OsString,
    fmt,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Command, ExitStatus, Stdio},
    thread,
    time::{Duration, Instant},
};

#[cfg(test)]
use std::fs;

const PROTOCOL_VERSION: u32 = 1;
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const DEFAULT_EXECUTE_TIMEOUT: Duration = Duration::from_secs(6 * 60 * 60);
const POLL_INTERVAL: Duration = Duration::from_millis(20);
const STDERR_TAIL_CHARS: usize = 8_000;
const EMBEDDED_COMPANION: &str = include_str!("../runtime/werk_media_companion.py");
const EMBEDDED_BOOTSTRAP: &str = "import io,sys\n\
n=int(sys.stdin.buffer.readline())\n\
code=sys.stdin.buffer.read(n)\n\
payload=sys.stdin.buffer.read()\n\
sys.stdin=io.TextIOWrapper(io.BytesIO(payload),encoding='utf-8')\n\
exec(compile(code,'<werk_media_companion>','exec'))";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompanionDependency {
    pub available: bool,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompanionHealth {
    pub ok: bool,
    pub status: String,
    pub protocol_version: u32,
    #[serde(default)]
    pub companion_version: Option<String>,
    #[serde(default)]
    pub python_version: Option<String>,
    #[serde(default)]
    pub dependencies: BTreeMap<String, CompanionDependency>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompanionOutput {
    pub path: String,
    #[serde(default)]
    pub mime_type: Option<String>,
    #[serde(default)]
    pub size: Option<u64>,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
    #[serde(default)]
    pub duration: Option<f64>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompanionExecution {
    pub ok: bool,
    pub task: String,
    #[serde(default)]
    pub outputs: Vec<CompanionOutput>,
    #[serde(default)]
    pub metadata: Value,
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompanionDoctorCheck {
    pub name: String,
    pub available: bool,
    pub required: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompanionDoctorReport {
    /// True when the launcher and protocol are usable. Optional ML packages do
    /// not make the companion globally unhealthy.
    pub available: bool,
    pub launcher: Option<String>,
    pub summary: String,
    pub checks: Vec<CompanionDoctorCheck>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompanionProtocolError {
    pub command: String,
    pub code: String,
    pub message: String,
    pub detail: Option<String>,
}

impl fmt::Display for CompanionProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "media companion {} failed [{}]: {}",
            self.command, self.code, self.message
        )?;
        if let Some(detail) = self.detail.as_deref().filter(|value| !value.is_empty()) {
            write!(f, " ({detail})")?;
        }
        Ok(())
    }
}

impl std::error::Error for CompanionProtocolError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LauncherKind {
    Executable,
    Python,
}

#[derive(Debug, Clone)]
struct CompanionLauncher {
    program: PathBuf,
    args: Vec<OsString>,
    source: String,
    kind: LauncherKind,
    embedded_script: bool,
}

impl CompanionLauncher {
    fn command(&self, operation: &str) -> Command {
        let mut command = Command::new(&self.program);
        command.args(&self.args).arg(operation);
        command
    }

    fn display(&self) -> String {
        let mut parts = vec![self.program.display().to_string()];
        parts.extend(
            self.args
                .iter()
                .map(|arg| arg.to_string_lossy().into_owned()),
        );
        format!("{} ({})", parts.join(" "), self.source)
    }
}

#[derive(Debug, Clone)]
pub struct CompanionClient {
    launcher: CompanionLauncher,
    request_timeout: Duration,
    execute_timeout: Duration,
}

impl CompanionClient {
    pub fn new() -> Result<Self> {
        Self::discover()
    }

    pub fn discover() -> Result<Self> {
        let launcher = discover_launcher()?;
        Ok(Self {
            launcher,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            execute_timeout: DEFAULT_EXECUTE_TIMEOUT,
        })
    }

    /// Builds a client for an explicitly resolved process. `program` and
    /// `args` are passed directly to `std::process::Command`; no shell parsing
    /// or interpolation is performed.
    pub fn from_command(
        program: impl Into<PathBuf>,
        args: impl IntoIterator<Item = OsString>,
    ) -> Self {
        Self {
            launcher: CompanionLauncher {
                program: program.into(),
                args: args.into_iter().collect(),
                source: "explicit command".to_string(),
                kind: LauncherKind::Executable,
                embedded_script: false,
            },
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            execute_timeout: DEFAULT_EXECUTE_TIMEOUT,
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self.execute_timeout = timeout;
        self
    }

    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    pub fn with_execute_timeout(mut self, timeout: Duration) -> Self {
        self.execute_timeout = timeout;
        self
    }

    pub fn launcher_description(&self) -> String {
        self.launcher.display()
    }

    pub fn health(&self) -> Result<CompanionHealth> {
        let value = self.request_with_timeout("health", &json!({}), self.request_timeout)?;
        let health: CompanionHealth =
            serde_json::from_value(value).context("invalid media companion health response")?;
        if health.protocol_version != PROTOCOL_VERSION {
            bail!(
                "media companion protocol mismatch: Werk expects {}, companion reports {}",
                PROTOCOL_VERSION,
                health.protocol_version
            );
        }
        Ok(health)
    }

    pub fn capabilities(&self) -> Result<Value> {
        self.request_with_timeout("capabilities", &json!({}), self.request_timeout)
    }

    pub fn probe_model(&self, request: &Value) -> Result<Value> {
        self.request_with_timeout("probe-model", request, self.request_timeout)
    }

    pub fn estimate(&self, request: &Value) -> Result<Value> {
        self.request_with_timeout("estimate", request, self.request_timeout)
    }

    pub fn execute(&self, request: &Value) -> Result<CompanionExecution> {
        let value = self.request_with_timeout("execute", request, self.execute_timeout)?;
        serde_json::from_value(value).context("invalid media companion execute response")
    }

    pub fn request(&self, operation: &str, request: &Value) -> Result<Value> {
        let timeout = if operation == "execute" {
            self.execute_timeout
        } else {
            self.request_timeout
        };
        self.request_with_timeout(operation, request, timeout)
    }

    pub fn doctor(&self) -> CompanionDoctorReport {
        let launcher = self.launcher_description();
        let mut checks = vec![CompanionDoctorCheck {
            name: match self.launcher.kind {
                LauncherKind::Executable => "media companion executable".to_string(),
                LauncherKind::Python => "Python media companion".to_string(),
            },
            available: true,
            required: true,
            detail: launcher.clone(),
        }];

        match self.health() {
            Ok(health) => {
                checks.push(CompanionDoctorCheck {
                    name: "media companion protocol".to_string(),
                    available: health.status == "ok",
                    required: true,
                    detail: format!(
                        "protocol v{}; companion {}; Python {}",
                        health.protocol_version,
                        health.companion_version.as_deref().unwrap_or("unknown"),
                        health.python_version.as_deref().unwrap_or("unknown")
                    ),
                });
                for (name, dependency) in health.dependencies {
                    checks.push(CompanionDoctorCheck {
                        name,
                        available: dependency.available,
                        required: false,
                        detail: dependency.detail.or(dependency.version).unwrap_or_else(|| {
                            if dependency.available {
                                "available".to_string()
                            } else {
                                "not installed (optional)".to_string()
                            }
                        }),
                    });
                }
                let available = checks
                    .iter()
                    .filter(|check| check.required)
                    .all(|check| check.available);
                CompanionDoctorReport {
                    available,
                    launcher: Some(launcher),
                    summary: if available {
                        "media companion is usable; missing optional runtimes only limit matching tasks"
                            .to_string()
                    } else {
                        "media companion launcher exists, but its protocol health check failed"
                            .to_string()
                    },
                    checks,
                }
            }
            Err(err) => {
                checks.push(CompanionDoctorCheck {
                    name: "media companion protocol".to_string(),
                    available: false,
                    required: true,
                    detail: err.to_string(),
                });
                CompanionDoctorReport {
                    available: false,
                    launcher: Some(launcher),
                    summary: "media companion health check failed".to_string(),
                    checks,
                }
            }
        }
    }

    pub fn discover_doctor_report() -> CompanionDoctorReport {
        match Self::discover() {
            Ok(client) => client.doctor(),
            Err(err) => CompanionDoctorReport {
                available: false,
                launcher: None,
                summary: "media companion is not configured".to_string(),
                checks: vec![CompanionDoctorCheck {
                    name: "media companion launcher".to_string(),
                    available: false,
                    required: true,
                    detail: err.to_string(),
                }],
            },
        }
    }

    fn request_with_timeout(
        &self,
        operation: &str,
        request: &Value,
        timeout: Duration,
    ) -> Result<Value> {
        validate_operation(operation)?;
        if !request.is_object() {
            bail!("media companion request for '{operation}' must be a JSON object");
        }
        if timeout.is_zero() {
            bail!("media companion timeout must be greater than zero");
        }

        let request_json = serde_json::to_vec(request)
            .with_context(|| format!("failed to serialize media companion {operation} request"))?;
        let input = if self.launcher.embedded_script {
            let script = EMBEDDED_COMPANION.as_bytes();
            let mut framed = format!("{}\n", script.len()).into_bytes();
            framed.extend_from_slice(script);
            framed.extend_from_slice(&request_json);
            framed
        } else {
            request_json
        };
        let mut command = self.launcher.command(operation);
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command.spawn().with_context(|| {
            format!(
                "failed to start media companion for '{operation}' using {}",
                self.launcher.display()
            )
        })?;

        let stdout = child
            .stdout
            .take()
            .context("failed to capture media companion stdout")?;
        let stderr = child
            .stderr
            .take()
            .context("failed to capture media companion stderr")?;
        let stdout_reader = read_pipe(stdout);
        let stderr_reader = read_pipe(stderr);

        let write_result = child
            .stdin
            .take()
            .context("failed to open media companion stdin")
            .and_then(|mut stdin| {
                stdin
                    .write_all(&input)
                    .context("failed to write media companion request")?;
                stdin
                    .flush()
                    .context("failed to flush media companion request")
            });
        if let Err(err) = write_result {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout_reader.join();
            let _ = stderr_reader.join();
            return Err(err);
        }

        let started = Instant::now();
        let status = loop {
            if let Some(status) = child
                .try_wait()
                .context("failed while waiting for media companion")?
            {
                break status;
            }
            if started.elapsed() >= timeout {
                let _ = child.kill();
                let status = child.wait().ok();
                let stdout = join_pipe(stdout_reader);
                let stderr = join_pipe(stderr_reader);
                bail!(
                    "media companion '{operation}' timed out after {:.3}s{}{}{}",
                    timeout.as_secs_f64(),
                    status
                        .map(|status| format!("; terminated with {status}"))
                        .unwrap_or_default(),
                    output_detail("stderr", &stderr),
                    output_detail("stdout", &stdout),
                );
            }
            thread::sleep(POLL_INTERVAL.min(timeout.saturating_sub(started.elapsed())));
        };

        let stdout = join_pipe(stdout_reader);
        let stderr = join_pipe(stderr_reader);
        if !status.success() {
            bail!(
                "media companion '{operation}' exited with {}{}{}",
                exit_status_detail(status),
                output_detail("stderr", &stderr),
                output_detail("stdout", &stdout),
            );
        }

        let value: Value = serde_json::from_slice(&stdout).map_err(|err| {
            anyhow!(
                "media companion '{operation}' returned invalid JSON: {err}{}{}",
                output_detail("stdout", &stdout),
                output_detail("stderr", &stderr)
            )
        })?;
        let object = value.as_object().ok_or_else(|| {
            anyhow!("media companion '{operation}' response must be a JSON object")
        })?;
        match object.get("ok").and_then(Value::as_bool) {
            Some(true) => Ok(value),
            Some(false) => Err(protocol_error(operation, object).into()),
            None => bail!("media companion '{operation}' response has no boolean 'ok' field"),
        }
    }
}

fn protocol_error(operation: &str, object: &Map<String, Value>) -> CompanionProtocolError {
    let error = object.get("error").and_then(Value::as_object);
    let code = error
        .and_then(|error| error.get("code"))
        .and_then(Value::as_str)
        .unwrap_or("companion_error")
        .to_string();
    let message = error
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .or_else(|| object.get("message").and_then(Value::as_str))
        .unwrap_or("media companion rejected the request")
        .to_string();
    let detail = error
        .and_then(|error| error.get("detail"))
        .and_then(value_detail);
    CompanionProtocolError {
        command: operation.to_string(),
        code,
        message,
        detail,
    }
}

fn value_detail(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(value) => Some(value.clone()),
        value => serde_json::to_string(value).ok(),
    }
}

fn validate_operation(operation: &str) -> Result<()> {
    if operation.is_empty()
        || !operation
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    {
        bail!("invalid media companion operation: {operation:?}");
    }
    Ok(())
}

fn read_pipe<R>(mut reader: R) -> thread::JoinHandle<std::io::Result<Vec<u8>>>
where
    R: Read + Send + 'static,
{
    thread::spawn(move || {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        Ok(bytes)
    })
}

fn join_pipe(handle: thread::JoinHandle<std::io::Result<Vec<u8>>>) -> Vec<u8> {
    handle.join().ok().and_then(Result::ok).unwrap_or_default()
}

fn output_detail(label: &str, bytes: &[u8]) -> String {
    if bytes.is_empty() {
        return String::new();
    }
    let text = String::from_utf8_lossy(bytes);
    let text = tail_chars(text.trim(), STDERR_TAIL_CHARS);
    if text.is_empty() {
        String::new()
    } else {
        format!("; {label}: {text}")
    }
}

fn tail_chars(text: &str, max_chars: usize) -> String {
    let count = text.chars().count();
    if count <= max_chars {
        return text.to_string();
    }
    format!(
        "...{}",
        text.chars()
            .skip(count.saturating_sub(max_chars))
            .collect::<String>()
    )
}

fn exit_status_detail(status: ExitStatus) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            return format!("{status} (signal {signal})");
        }
    }
    status.to_string()
}

fn discover_launcher() -> Result<CompanionLauncher> {
    if let Some(configured) = env::var_os("WERK_MEDIA_COMPANION") {
        let path = resolve_program(&configured).ok_or_else(|| {
            anyhow!(
                "WERK_MEDIA_COMPANION does not resolve to an executable file: {}",
                PathBuf::from(&configured).display()
            )
        })?;
        return Ok(CompanionLauncher {
            program: path,
            args: Vec::new(),
            source: "env WERK_MEDIA_COMPANION".to_string(),
            kind: LauncherKind::Executable,
            embedded_script: false,
        });
    }

    let (python, python_source) = discover_python().ok_or_else(|| {
        anyhow!(
            "no media companion executable or Python found; set WERK_MEDIA_COMPANION or WERK_MEDIA_PYTHON"
        )
    })?;
    if let Some((script, source)) = discover_repo_script() {
        return Ok(CompanionLauncher {
            program: python,
            args: vec![script.into_os_string()],
            source: format!("{python_source}; {source}"),
            kind: LauncherKind::Python,
            embedded_script: false,
        });
    }

    Ok(CompanionLauncher {
        program: python,
        args: vec![OsString::from("-c"), OsString::from(EMBEDDED_BOOTSTRAP)],
        source: format!("{python_source}; embedded companion script"),
        kind: LauncherKind::Python,
        embedded_script: true,
    })
}

fn discover_python() -> Option<(PathBuf, String)> {
    if let Some(configured) = env::var_os("WERK_MEDIA_PYTHON") {
        return resolve_program(&configured)
            .map(|path| (path, "env WERK_MEDIA_PYTHON".to_string()));
    }
    for name in python_program_names() {
        if let Some(path) = find_in_path(name) {
            return Some((path, format!("PATH {name}")));
        }
    }
    None
}

fn discover_repo_script() -> Option<(PathBuf, String)> {
    if let Some(configured) = env::var_os("WERK_MEDIA_COMPANION_SCRIPT") {
        let path = PathBuf::from(configured);
        if path.is_file() {
            return Some((path, "env WERK_MEDIA_COMPANION_SCRIPT".to_string()));
        }
    }

    let mut candidates = Vec::new();
    if let Some(root) = option_env!("CARGO_MANIFEST_DIR") {
        candidates.push((
            PathBuf::from(root)
                .join("runtime")
                .join("werk_media_companion.py"),
            "repository runtime script".to_string(),
        ));
    }
    if let Ok(executable) = env::current_exe()
        && let Some(dir) = executable.parent()
    {
        candidates.push((
            dir.join("runtime").join("werk_media_companion.py"),
            "runtime script next to executable".to_string(),
        ));
        candidates.push((
            dir.join("werk_media_companion.py"),
            "script next to executable".to_string(),
        ));
        if let Some(parent) = dir.parent() {
            candidates.push((
                parent.join("runtime").join("werk_media_companion.py"),
                "runtime script next to installation".to_string(),
            ));
        }
    }
    candidates.into_iter().find(|(path, _)| path.is_file())
}

fn python_program_names() -> &'static [&'static str] {
    if cfg!(windows) {
        &["python.exe", "python3.exe", "python", "python3"]
    } else {
        &["python3", "python"]
    }
}

fn resolve_program(program: &OsString) -> Option<PathBuf> {
    let path = PathBuf::from(program);
    if path.components().count() > 1 || path.is_absolute() {
        return path.is_file().then_some(path);
    }
    find_in_path(path.to_str()?)
}

fn find_in_path(name: &str) -> Option<PathBuf> {
    let path = Path::new(name);
    if (path.components().count() > 1 || path.is_absolute()) && path.is_file() {
        return Some(path.to_path_buf());
    }
    let path_env = env::var_os("PATH")?;
    for directory in env::split_paths(&path_env) {
        let candidate = directory.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
        #[cfg(windows)]
        if candidate.extension().is_none() {
            for extension in ["exe", "cmd", "bat"] {
                let candidate = candidate.with_extension(extension);
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::atomic::{AtomicU64, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };

    const MOCK_COMPANION: &str = r#"
import json
import sys
import time

operation = sys.argv[1]
payload = json.load(sys.stdin)

if payload.get("sleep"):
    time.sleep(float(payload["sleep"]))
if payload.get("exit"):
    print("mock process failure", file=sys.stderr)
    raise SystemExit(17)
if payload.get("malformed"):
    print('{"ok":true}{"ok":true}')
    raise SystemExit(0)
if payload.get("reject"):
    print(json.dumps({
        "ok": False,
        "error": {
            "code": "unsupported_parameter",
            "message": "mock rejection",
            "detail": {"parameter": "steps"},
        },
    }))
    raise SystemExit(0)

if operation == "health":
    response = {
        "ok": True,
        "status": "ok",
        "protocol_version": 1,
        "companion_version": "mock",
        "python_version": sys.version.split()[0],
        "dependencies": {
            "torch": {
                "available": False,
                "version": None,
                "detail": "not installed in mock",
            },
            "PIL": {
                "available": True,
                "version": "mock",
                "detail": None,
            },
        },
    }
elif operation == "probe-model":
    response = {
        "ok": True,
        "supported": True,
        "echo": payload,
    }
elif operation == "estimate":
    response = {
        "ok": True,
        "confidence": "heuristic",
        "accelerator_peak_bytes": 123,
    }
elif operation == "execute":
    response = {
        "ok": True,
        "task": payload.get("task", "image_generation"),
        "outputs": [{
            "path": "/tmp/mock.png",
            "mime_type": "image/png",
            "size": 3,
            "width": 1,
            "height": 1,
            "duration": None,
            "metadata": {},
        }],
        "metadata": {"mock": True},
        "warnings": [],
    }
else:
    response = {
        "ok": False,
        "error": {"code": "unknown_command", "message": operation},
    }

print(json.dumps(response))
"#;

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new(label: &str) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let nonce = NEXT.fetch_add(1, Ordering::Relaxed);
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let path = env::temp_dir().join(format!(
                "werk-media-companion-{label}-{}-{timestamp}-{nonce}",
                std::process::id()
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn mock_client(timeout: Duration) -> Option<(TestDirectory, CompanionClient)> {
        let python = python_program_names()
            .iter()
            .find_map(|name| find_in_path(name))?;
        let directory = TestDirectory::new("mock");
        let script = directory.0.join("mock_companion.py");
        fs::write(&script, MOCK_COMPANION).unwrap();
        let client = CompanionClient::from_command(python, vec![script.into_os_string()])
            .with_timeout(timeout);
        Some((directory, client))
    }

    #[test]
    fn mock_companion_successfully_handles_all_public_operations() {
        let Some((_directory, client)) = mock_client(Duration::from_secs(3)) else {
            return;
        };
        let health = client.health().unwrap();
        assert_eq!(health.status, "ok");
        assert_eq!(health.protocol_version, PROTOCOL_VERSION);
        assert!(!health.dependencies["torch"].available);

        let probe = client
            .probe_model(&json!({"model_path": "/tmp/model"}))
            .unwrap();
        assert_eq!(probe["supported"], true);
        let estimate = client
            .estimate(&json!({"task": "image_generation"}))
            .unwrap();
        assert_eq!(estimate["accelerator_peak_bytes"], 123);
        let execution = client
            .execute(&json!({"task": "image_generation"}))
            .unwrap();
        assert_eq!(execution.task, "image_generation");
        assert_eq!(execution.outputs.len(), 1);
    }

    #[test]
    fn protocol_error_preserves_companion_code_and_detail() {
        let Some((_directory, client)) = mock_client(Duration::from_secs(3)) else {
            return;
        };
        let err = client.probe_model(&json!({"reject": true})).unwrap_err();
        let protocol = err.downcast_ref::<CompanionProtocolError>().unwrap();
        assert_eq!(protocol.code, "unsupported_parameter");
        assert!(protocol.detail.as_deref().unwrap().contains("steps"));
    }

    #[test]
    fn nonzero_exit_reports_status_and_stderr() {
        let Some((_directory, client)) = mock_client(Duration::from_secs(3)) else {
            return;
        };
        let err = client
            .probe_model(&json!({"exit": true}))
            .unwrap_err()
            .to_string();
        assert!(err.contains("17") || err.contains("exit"));
        assert!(err.contains("mock process failure"));
    }

    #[test]
    fn timeout_terminates_mock_process() {
        let Some((_directory, client)) = mock_client(Duration::from_millis(100)) else {
            return;
        };
        let started = Instant::now();
        let err = client
            .probe_model(&json!({"sleep": 2}))
            .unwrap_err()
            .to_string();
        assert!(err.contains("timed out"));
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn malformed_or_multiple_json_objects_are_rejected() {
        let Some((_directory, client)) = mock_client(Duration::from_secs(3)) else {
            return;
        };
        let err = client
            .probe_model(&json!({"malformed": true}))
            .unwrap_err()
            .to_string();
        assert!(err.contains("invalid JSON"));
    }

    #[test]
    fn missing_optional_dependencies_do_not_make_doctor_globally_fatal() {
        let Some((_directory, client)) = mock_client(Duration::from_secs(3)) else {
            return;
        };
        let report = client.doctor();
        assert!(report.available);
        let torch = report
            .checks
            .iter()
            .find(|check| check.name == "torch")
            .unwrap();
        assert!(!torch.required);
        assert!(!torch.available);
    }

    #[test]
    fn embedded_script_transport_keeps_large_script_out_of_command_line() {
        let Some(python) = python_program_names()
            .iter()
            .find_map(|name| find_in_path(name))
        else {
            return;
        };
        let client = CompanionClient {
            launcher: CompanionLauncher {
                program: python,
                args: vec![OsString::from("-c"), OsString::from(EMBEDDED_BOOTSTRAP)],
                source: "embedded transport test".to_string(),
                kind: LauncherKind::Python,
                embedded_script: true,
            },
            request_timeout: Duration::from_secs(5),
            execute_timeout: Duration::from_secs(5),
        };
        let health = client.health().unwrap();
        assert_eq!(health.protocol_version, PROTOCOL_VERSION);
        assert_eq!(health.status, "ok");
    }

    #[test]
    fn real_companion_strict_policy_rejects_unsupported_explicit_parameter() {
        let Some(python) = python_program_names()
            .iter()
            .find_map(|name| find_in_path(name))
        else {
            return;
        };
        let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("runtime")
            .join("werk_media_companion.py");
        let client = CompanionClient::from_command(python, vec![script.into_os_string()])
            .with_timeout(Duration::from_secs(5));
        let model = TestDirectory::new("strict-model");
        let output = TestDirectory::new("strict-output");

        let err = client
            .execute(&json!({
                "model_path": model.0.display().to_string(),
                "output_dir": output.0.display().to_string(),
                "task": "image_generation",
                "prompt": "fixture",
                "effective_parameters": {
                    "image.sampler": "euler"
                },
                "explicit_parameters": ["image.sampler"],
                "parameter_policy": "strict"
            }))
            .unwrap_err();
        let protocol = err.downcast_ref::<CompanionProtocolError>().unwrap();
        assert_eq!(protocol.code, "unsupported_parameter");
        assert!(
            protocol
                .detail
                .as_deref()
                .is_some_and(|detail| detail.contains("image.sampler"))
        );
    }

    #[test]
    fn real_companion_warn_policy_reports_ignored_parameter_during_estimate() {
        let Some(python) = python_program_names()
            .iter()
            .find_map(|name| find_in_path(name))
        else {
            return;
        };
        let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("runtime")
            .join("werk_media_companion.py");
        let client = CompanionClient::from_command(python, vec![script.into_os_string()])
            .with_timeout(Duration::from_secs(5));
        let model = TestDirectory::new("warn-model");
        fs::write(model.0.join("weights.bin"), b"fixture").unwrap();

        let estimate = client
            .estimate(&json!({
                "model_path": model.0.display().to_string(),
                "task": "image_generation",
                "effective_parameters": {
                    "image.sampler": "euler",
                    "image.width": 64,
                    "image.height": 64
                },
                "explicit_parameters": ["image.sampler", "image.width"],
                "parameter_policy": "warn"
            }))
            .unwrap();
        let warnings = estimate["warnings"].as_array().unwrap();
        assert!(
            warnings
                .iter()
                .filter_map(Value::as_str)
                .any(|warning| warning.contains("image.sampler"))
        );
        assert_eq!(
            estimate["parameter_support"]["unsupported_explicit_parameters"],
            json!(["image.sampler"])
        );
    }
}
