import React, { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar
} from 'recharts';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [metrics, setMetrics] = useState([]);
  const [experiments, setExperiments] = useState([]);
  const [benchmarks, setBenchmarks] = useState([]);
  const [checkpoints, setCheckpoints] = useState([]);
  const [artifacts, setArtifacts] = useState([]);
  const [reproducibility, setReproducibility] = useState(null);
  const [mlflowStatus, setMlflowStatus] = useState(null);
  const [tbStatus, setTbStatus] = useState(null);
  const [leaderboard, setLeaderboard] = useState([]);
  const [currentExp, setCurrentExp] = useState(null);
  const [currentBench, setCurrentBench] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('CONNECTING');
  const [wsConnected, setWsConnected] = useState(false);
  const [latestRound, setLatestRound] = useState(0);
  const [latestLoss, setLatestLoss] = useState('N/A');
  const [latestDice, setLatestDice] = useState('N/A');

  useEffect(() => {
    // Initial fetch of metrics
    fetch('/api/v1/metrics/default')
      .then((res) => res.json())
      .then((res) => {
        if (res.data && Array.isArray(res.data)) {
          setMetrics(res.data);
          if (res.data.length > 0) {
            const last = res.data[res.data.length - 1];
            setLatestRound(last.round_number);
            setLatestLoss(last.training_loss ? last.training_loss.toFixed(4) : 'N/A');
            setLatestDice(last.dice_score ? last.dice_score.toFixed(4) : 'N/A');
          }
        }
      })
      .catch((err) => console.log('Metrics fetch error:', err));

    // Initial fetch of experiments
    fetch('/api/v1/experiments')
      .then((res) => res.json())
      .then((res) => {
        if (res.data && Array.isArray(res.data) && res.data.length > 0) {
          setExperiments(res.data);
          setCurrentExp(res.data[0]);
        }
      })
      .catch((err) => console.log('Experiments fetch error:', err));

    // Fetch benchmarks
    fetch('/api/v1/benchmarks')
      .then((res) => res.json())
      .then((res) => {
        if (res.data && Array.isArray(res.data) && res.data.length > 0) {
          setBenchmarks(res.data);
          const activeBench = res.data[0];
          setCurrentBench(activeBench);
          fetch(`/api/v1/benchmarks/${activeBench.benchmark_id}/leaderboard`)
            .then((r) => r.json())
            .then((lRes) => {
              if (lRes.data && Array.isArray(lRes.data)) {
                setLeaderboard(lRes.data);
              }
            })
            .catch(() => {});
        }
      })
      .catch((err) => console.log('Benchmarks fetch error:', err));

    // Milestone J endpoints fetch
    fetch('/api/v1/checkpoints')
      .then((r) => r.json())
      .then((d) => d.data && setCheckpoints(d.data))
      .catch(() => {});

    fetch('/api/v1/artifacts')
      .then((r) => r.json())
      .then((d) => d.data?.artifacts && setArtifacts(d.data.artifacts))
      .catch(() => {});

    fetch('/api/v1/system/reproducibility')
      .then((r) => r.json())
      .then((d) => d.data && setReproducibility(d.data))
      .catch(() => {});

    fetch('/api/v1/mlflow/status')
      .then((r) => r.json())
      .then((d) => d.data && setMlflowStatus(d.data))
      .catch(() => {});

    fetch('/api/v1/tensorboard/status')
      .then((r) => r.json())
      .then((d) => d.data && setTbStatus(d.data))
      .catch(() => {});

    // WebSocket telemetry stream
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/v1/telemetry/ws`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      setConnectionStatus('CONNECTED');
      setWsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.event === 'metrics_updated' && msg.data) {
          const newMetric = msg.data;
          setMetrics((prev) => {
            const filtered = prev.filter((m) => m.round_number !== newMetric.round_number);
            return [...filtered, newMetric].sort((a, b) => a.round_number - b.round_number);
          });
          setLatestRound(newMetric.round_number);
          setLatestLoss(newMetric.training_loss ? newMetric.training_loss.toFixed(4) : 'N/A');
          setLatestDice(newMetric.dice_score ? newMetric.dice_score.toFixed(4) : 'N/A');
        }
      } catch (err) {
        console.error('Error parsing WS telemetry message:', err);
      }
    };

    ws.onclose = () => {
      setConnectionStatus('DISCONNECTED');
      setWsConnected(false);
    };

    return () => ws.close();
  }, []);

  const expStatus = currentExp?.status?.toUpperCase() || 'RUNNING';

  return (
    <div className="dashboard-container">
      {/* Header */}
      <header className="header">
        <div className="header-title-group">
          <div className="logo-badge">FM</div>
          <div>
            <h1>FedMed v2.0 Research Platform</h1>
            <p className="header-subtitle">
              Reproducible Federated Learning Experiment Tracking, Checkpoint Registry & Benchmark Platform
            </p>
            <div className="meta-tags">
              <span className="tag">BENCHMARK: {currentBench?.benchmark_id || 'bm_suite'}</span>
              <span className="tag">EXP: {currentExp?.experiment_id || 'default'}</span>
              <span className="tag">STATUS: {expStatus}</span>
              <span className="tag">GIT: {reproducibility?.git?.git_commit?.slice(0, 7) || 'HEAD'}</span>
            </div>
          </div>
        </div>

        <div className="status-indicator">
          <div className={`dot ${wsConnected ? 'green' : 'red'}`}></div>
          <span>TELEMETRY: {connectionStatus}</span>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.5rem', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '0.75rem' }}>
        <button
          onClick={() => setActiveTab('overview')}
          style={{
            padding: '0.6rem 1.2rem',
            borderRadius: '0.5rem',
            border: 'none',
            fontWeight: '600',
            cursor: 'pointer',
            backgroundColor: activeTab === 'overview' ? '#0284c7' : '#1e293b',
            color: '#ffffff'
          }}
        >
          Overview & Telemetry
        </button>
        <button
          onClick={() => setActiveTab('checkpoints')}
          style={{
            padding: '0.6rem 1.2rem',
            borderRadius: '0.5rem',
            border: 'none',
            fontWeight: '600',
            cursor: 'pointer',
            backgroundColor: activeTab === 'checkpoints' ? '#0284c7' : '#1e293b',
            color: '#ffffff'
          }}
        >
          Checkpoint Registry ({checkpoints.length})
        </button>
        <button
          onClick={() => setActiveTab('benchmarks')}
          style={{
            padding: '0.6rem 1.2rem',
            borderRadius: '0.5rem',
            border: 'none',
            fontWeight: '600',
            cursor: 'pointer',
            backgroundColor: activeTab === 'benchmarks' ? '#0284c7' : '#1e293b',
            color: '#ffffff'
          }}
        >
          Benchmark Explorer & Leaderboard
        </button>
        <button
          onClick={() => setActiveTab('reproducibility')}
          style={{
            padding: '0.6rem 1.2rem',
            borderRadius: '0.5rem',
            border: 'none',
            fontWeight: '600',
            cursor: 'pointer',
            backgroundColor: activeTab === 'reproducibility' ? '#0284c7' : '#1e293b',
            color: '#ffffff'
          }}
        >
          MLflow & Reproducibility
        </button>
        <button
          onClick={() => setActiveTab('artifacts')}
          style={{
            padding: '0.6rem 1.2rem',
            borderRadius: '0.5rem',
            border: 'none',
            fontWeight: '600',
            cursor: 'pointer',
            backgroundColor: activeTab === 'artifacts' ? '#0284c7' : '#1e293b',
            color: '#ffffff'
          }}
        >
          Artifact Viewer ({artifacts.length})
        </button>
      </div>

      {/* Subsystem Health Bar */}
      <div className="status-bar" style={{ marginBottom: '1.5rem' }}>
        <span className="status-label">System Architecture:</span>
        <div className="status-indicator"><div className="dot green"></div><span>REST API: /api/v1</span></div>
        <div className="status-indicator"><div className={`dot ${mlflowStatus?.enabled ? 'green' : 'yellow'}`}></div><span>MLFLOW: {mlflowStatus?.enabled ? 'ACTIVE' : 'FILE_LOCAL'}</span></div>
        <div className="status-indicator"><div className={`dot ${tbStatus?.has_active_runs ? 'green' : 'yellow'}`}></div><span>TENSORBOARD: {tbStatus?.has_active_runs ? 'LOGGING' : 'READY'}</span></div>
        <div className="status-indicator"><div className="dot green"></div><span>CHECKPOINTS: {checkpoints.length} Indexed</span></div>
        <div className="status-indicator"><div className="dot green"></div><span>ARTIFACTS: {artifacts.length} Files</span></div>
      </div>

      {/* TAB 1: OVERVIEW & TELEMETRY */}
      {activeTab === 'overview' && (
        <>
          <div className="kpi-grid">
            <div className="kpi-card">
              <div className="kpi-title">Completed FL Round</div>
              <div className="kpi-value-row">
                <div className="kpi-value">{latestRound}</div>
                <div className="kpi-trend neutral">{currentExp?.num_rounds || 3} Max Rounds</div>
              </div>
            </div>

            <div className="kpi-card">
              <div className="kpi-title">Global Training Loss</div>
              <div className="kpi-value-row">
                <div className="kpi-value" style={{ color: 'var(--accent-cyan)' }}>{latestLoss}</div>
                <div className="kpi-trend positive">↓ Decreasing</div>
              </div>
            </div>

            <div className="kpi-card">
              <div className="kpi-title">Dice Score (Similarity)</div>
              <div className="kpi-value-row">
                <div className="kpi-value" style={{ color: 'var(--accent-purple)' }}>{latestDice}</div>
                <div className="kpi-trend positive">↑ Improving</div>
              </div>
            </div>

            <div className="kpi-card">
              <div className="kpi-title">Participating Silos</div>
              <div className="kpi-value-row">
                <div className="kpi-value" style={{ color: 'var(--accent-green)' }}>{currentExp?.num_clients || 3}</div>
                <div className="kpi-trend positive">100% Online</div>
              </div>
            </div>
          </div>

          <div className="charts-grid">
            <div className="panel-card">
              <div className="panel-header">
                <div className="panel-title">Real-time Convergence Curves (Dice & Loss)</div>
                <div className="tag">LIVE TELEMETRY</div>
              </div>
              <div style={{ width: '100%', height: 320 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={metrics}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                    <XAxis dataKey="round_number" stroke="#64748b" tick={{ fill: '#94a3b8' }} />
                    <YAxis stroke="#64748b" tick={{ fill: '#94a3b8' }} domain={[0, 1]} />
                    <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderRadius: '0.75rem' }} />
                    <Legend />
                    <Line type="monotone" dataKey="training_loss" stroke="#38bdf8" name="Training Loss" strokeWidth={3} />
                    <Line type="monotone" dataKey="dice_score" stroke="#c084fc" name="Dice Score" strokeWidth={3} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="panel-card">
              <div className="panel-header">
                <div className="panel-title">Active Experiment Specs</div>
                <div className="tag">{expStatus}</div>
              </div>
              <div className="info-list">
                <div className="info-item"><span className="info-key">Experiment ID</span><span className="info-value">{currentExp?.experiment_id || 'default'}</span></div>
                <div className="info-item"><span className="info-key">Strategy</span><span className="info-value">{currentExp?.strategy_name || 'FedAvg'}</span></div>
                <div className="info-item"><span className="info-key">Learning Rate</span><span className="info-value">{currentExp?.learning_rate || '1e-4'}</span></div>
                <div className="info-item"><span className="info-key">Batch Size / Epochs</span><span className="info-value">{currentExp?.batch_size || 2} / {currentExp?.local_epochs || 1}</span></div>
                <div className="info-item"><span className="info-key">Random Seed</span><span className="info-value">{currentExp?.seed || 42}</span></div>
                <div className="info-item"><span className="info-key">Differential Privacy</span><span className="info-value">{currentExp?.dp_enabled ? 'Opacus DP' : 'Disabled'}</span></div>
                <div className="info-item"><span className="info-key">Homomorphic Enc.</span><span className="info-value">{currentExp?.he_enabled ? 'TenSEAL CKKS' : 'Disabled'}</span></div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* TAB 2: CHECKPOINT REGISTRY */}
      {activeTab === 'checkpoints' && (
        <div className="panel-card">
          <div className="panel-header">
            <div className="panel-title">Checkpoint Registry</div>
            <div className="tag">INDEXED STATE WEIGHTS</div>
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Checkpoint ID</th>
                <th>Strategy</th>
                <th>Round / Epoch</th>
                <th>Dice</th>
                <th>Loss</th>
                <th>SHA-256 Hash</th>
                <th>Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {checkpoints.map((c) => (
                <tr key={c.checkpoint_id}>
                  <td><code>{c.checkpoint_id}</code></td>
                  <td>{c.strategy}</td>
                  <td>{c.round ? `Round ${c.round}` : `Epoch ${c.epoch}`}</td>
                  <td style={{ color: 'var(--accent-purple)', fontWeight: 'bold' }}>{c.dice?.toFixed(4)}</td>
                  <td style={{ color: 'var(--accent-cyan)' }}>{c.loss?.toFixed(4)}</td>
                  <td><code>{c.file_hash?.slice(0, 12)}...</code></td>
                  <td>{c.timestamp}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* TAB 3: BENCHMARK EXPLORER */}
      {activeTab === 'benchmarks' && (
        <div className="panel-card">
          <div className="panel-header">
            <div className="panel-title">Benchmark Suite Leaderboard ({currentBench?.benchmark_id || 'bm_suite'})</div>
            <div className="tag">RANKED MATRIX SWEEPS</div>
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Experiment ID</th>
                <th>Strategy</th>
                <th>Partition</th>
                <th>Seed</th>
                <th>Best Dice</th>
                <th>Avg Loss</th>
                <th>Runtime (s)</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {leaderboard.map((row) => (
                <tr key={row.experiment_id}>
                  <td><span className="node-badge">#{row.rank}</span></td>
                  <td>{row.experiment_id}</td>
                  <td>{row.strategy_name}</td>
                  <td>{row.partition_strategy}</td>
                  <td>{row.seed}</td>
                  <td style={{ color: 'var(--accent-purple)', fontWeight: 'bold' }}>{row.best_dice}</td>
                  <td style={{ color: 'var(--accent-cyan)' }}>{row.avg_loss}</td>
                  <td>{row.runtime_sec}s</td>
                  <td><span style={{ color: row.status === 'completed' ? 'var(--accent-green)' : 'var(--accent-amber)' }}>● {row.status.toUpperCase()}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* TAB 4: REPRODUCIBILITY & MLFLOW */}
      {activeTab === 'reproducibility' && (
        <div className="charts-grid">
          <div className="panel-card">
            <div className="panel-header">
              <div className="panel-title">Git & Software Metadata</div>
              <div className="tag">REPRODUCIBILITY</div>
            </div>
            <div className="info-list">
              <div className="info-item"><span className="info-key">Git Branch</span><span className="info-value"><code>{reproducibility?.git?.git_branch}</code></span></div>
              <div className="info-item"><span className="info-key">Git Commit</span><span className="info-value"><code>{reproducibility?.git?.git_commit}</code></span></div>
              <div className="info-item"><span className="info-key">Python Version</span><span className="info-value">{reproducibility?.dependencies?.python_version}</span></div>
              <div className="info-item"><span className="info-key">PyTorch Version</span><span className="info-value">{reproducibility?.dependencies?.torch_version}</span></div>
              <div className="info-item"><span className="info-key">MONAI Version</span><span className="info-value">{reproducibility?.dependencies?.monai_version}</span></div>
              <div className="info-item"><span className="info-key">Flower Version</span><span className="info-value">{reproducibility?.dependencies?.flower_version}</span></div>
            </div>
          </div>

          <div className="panel-card">
            <div className="panel-header">
              <div className="panel-title">Hardware Context & Tracking Servers</div>
              <div className="tag">ENVIRONMENT</div>
            </div>
            <div className="info-list">
              <div className="info-item"><span className="info-key">Operating System</span><span className="info-value">{reproducibility?.hardware?.os_platform}</span></div>
              <div className="info-item"><span className="info-key">CPU Cores / Arch</span><span className="info-value">{reproducibility?.hardware?.cpu_count} Cores ({reproducibility?.hardware?.cpu_arch})</span></div>
              <div className="info-item"><span className="info-key">System RAM</span><span className="info-value">{reproducibility?.hardware?.ram_gb} GB</span></div>
              <div className="info-item"><span className="info-key">CUDA Available</span><span className="info-value">{reproducibility?.hardware?.cuda_available ? 'Yes' : 'No (CPU Mode)'}</span></div>
              <div className="info-item"><span className="info-key">MLflow URI</span><span className="info-value"><code>{mlflowStatus?.tracking_uri}</code></span></div>
              <div className="info-item"><span className="info-key">TensorBoard Logdir</span><span className="info-value"><code>{tbStatus?.logdir}</code></span></div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 5: ARTIFACT VIEWER */}
      {activeTab === 'artifacts' && (
        <div className="panel-card">
          <div className="panel-header">
            <div className="panel-title">Discovered Research Artifacts</div>
            <div className="tag">EXPORT ENGINE OUTPUTS</div>
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Artifact Name</th>
                <th>File Path</th>
                <th>Type</th>
                <th>Size (Bytes)</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {artifacts.map((a, i) => (
                <tr key={i}>
                  <td><b>{a.name}</b></td>
                  <td><code>{a.path}</code></td>
                  <td><span className="node-badge">{a.extension.toUpperCase()}</span></td>
                  <td>{a.size_bytes.toLocaleString()} B</td>
                  <td>
                    <a href={`/api/v1/artifacts/download?path=${encodeURIComponent(a.path)}`} target="_blank" rel="noreferrer" style={{ color: '#38bdf8', fontWeight: 'bold' }}>
                      Download ↓
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
