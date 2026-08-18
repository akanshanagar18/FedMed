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
  Bar,
  AreaChart,
  Area
} from 'recharts';
import {
  Activity,
  Shield,
  Brain,
  Database,
  Server,
  FileText,
  Lock,
  Sliders,
  Download,
  RefreshCw,
  Play,
  CheckCircle,
  AlertTriangle,
  TrendingUp,
  Terminal,
  Cpu,
  Layers,
  Search,
  Award,
  ChevronRight,
  Info,
  Users,
  Boxes,
  BarChart2,
  Settings,
  Zap,
  GitBranch,
  Crosshair,
  ClipboardList,
  CheckSquare,
  LayoutDashboard,
  Eye
} from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  
  // Backend State
  const [metrics, setMetrics] = useState([]);
  const [experiments, setExperiments] = useState([]);
  const [benchmarks, setBenchmarks] = useState([]);
  const [checkpoints, setCheckpoints] = useState([]);
  const [artifacts, setArtifacts] = useState([]);
  const [systemHealth, setSystemHealth] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('CONNECTING');
  
  // Real-time Telemetry State
  const [latestRound, setLatestRound] = useState(20);
  const [latestLoss, setLatestLoss] = useState('0.9604');
  const [latestDice, setLatestDice] = useState('0.0800');
  
  // Production Model Specification State
  const [modelInfo, setModelInfo] = useState(null);
  const [availableCases, setAvailableCases] = useState([]);

  // 3D Inference Interactive Form State
  const [infSubjectId, setInfSubjectId] = useState('BraTS-GLI-00005-100');
  const [infResult, setInfResult] = useState(null);
  const [infLoading, setInfLoading] = useState(false);
  const [infHistory, setInfHistory] = useState([]);
  
  // Model Export Interactive Form State
  const [expModelId, setExpModelId] = useState('brats_monai_3d_unet');
  const [expVersion, setExpVersion] = useState('v2.1.0-dp-prod');
  const [expResult, setExpResult] = useState(null);
  const [expLoading, setExpLoading] = useState(false);

  // Self-Healing State
  const [shNodeId, setShNodeId] = useState('hospital_alpha');
  const [shReason, setShReason] = useState('Feature Drift Compensation');
  const [shResult, setShResult] = useState(null);
  const [shLoading, setShLoading] = useState(false);

  // Governance & SLA State
  const [slaAudit, setSlaAudit] = useState(null);
  const [driftMetrics, setDriftMetrics] = useState(null);
  const [registeredDatasets, setRegisteredDatasets] = useState([]);
  const [auditLogs, setAuditLogs] = useState([]);
  const [currentUser, setCurrentUser] = useState(null);

  // Initial API Fetches
  useEffect(() => {
    fetchAllData();

    // WebSocket Telemetry
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/v1/telemetry/ws`;
    let ws;
    try {
      ws = new WebSocket(wsUrl);
      ws.onopen = () => setConnectionStatus('ONLINE');
      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.event === 'metrics_updated' && msg.data) {
            const newM = msg.data;
            setMetrics((prev) => {
              const filtered = prev.filter((m) => m.round_number !== newM.round_number);
              return [...filtered, newM].sort((a, b) => a.round_number - b.round_number);
            });
            setLatestRound(newM.round_number);
            setLatestLoss(newM.training_loss ? newM.training_loss.toFixed(4) : '0.9604');
            setLatestDice(newM.dice_score ? newM.dice_score.toFixed(4) : '0.0800');
          }
        } catch (e) {}
      };
      ws.onerror = () => setConnectionStatus('OFFLINE');
      ws.onclose = () => setConnectionStatus('OFFLINE');
    } catch (e) {
      setConnectionStatus('OFFLINE');
    }

    return () => {
      if (ws) ws.close();
    };
  }, []);

  const fetchAllData = () => {
    // Model Metadata Card
    fetch('/api/v1/inference/model')
      .then((res) => res.json())
      .then((res) => res.data && setModelInfo(res.data))
      .catch(() => {});

    // Demo Cases
    fetch('/api/v1/inference/cases')
      .then((res) => res.json())
      .then((res) => res.data?.cases && setAvailableCases(res.data.cases))
      .catch(() => {});

    // Metrics
    fetch('/api/v1/metrics/default')
      .then((res) => res.json())
      .then((res) => {
        if (res.data && Array.isArray(res.data)) {
          setMetrics(res.data);
          if (res.data.length > 0) {
            const last = res.data[res.data.length - 1];
            setLatestRound(last.round_number);
            setLatestLoss(last.training_loss ? last.training_loss.toFixed(4) : '0.9604');
            setLatestDice(last.dice_score ? last.dice_score.toFixed(4) : '0.0800');
          }
        }
      })
      .catch(() => {});

    // Experiments
    fetch('/api/v1/experiments')
      .then((res) => res.json())
      .then((res) => res.data && Array.isArray(res.data) && setExperiments(res.data))
      .catch(() => {});

    // Benchmarks
    fetch('/api/v1/benchmarks')
      .then((res) => res.json())
      .then((res) => res.data && Array.isArray(res.data) && setBenchmarks(res.data))
      .catch(() => {});

    // Checkpoints
    fetch('/api/v1/checkpoints')
      .then((r) => r.json())
      .then((d) => d.data && Array.isArray(d.data) && setCheckpoints(d.data))
      .catch(() => {});

    // Artifacts
    fetch('/api/v1/artifacts')
      .then((r) => r.json())
      .then((d) => d.data?.artifacts && Array.isArray(d.data.artifacts) && setArtifacts(d.data.artifacts))
      .catch(() => {});

    // System Health
    fetch('/api/v1/system/health')
      .then((r) => r.json())
      .then((d) => d.data && setSystemHealth(d.data))
      .catch(() => {});

    // Inference History
    fetch('/api/v1/inference/history')
      .then((r) => r.json())
      .then((d) => d.data?.history && Array.isArray(d.data.history) && setInfHistory(d.data.history))
      .catch(() => {});

    // Dataset Management
    fetch('/api/v1/dataset-management/datasets')
      .then((r) => r.json())
      .then((d) => d.data && Array.isArray(d.data) && setRegisteredDatasets(d.data))
      .catch(() => {});

    // Audit Logs
    fetch('/api/v1/auth/audit-logs')
      .then((r) => r.json())
      .then((d) => d.data && Array.isArray(d.data) && setAuditLogs(d.data))
      .catch(() => {});

    // Governance SLA
    fetch('/api/v1/governance/sla/audit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ run_id: 'run_dp_d2_1_final', epsilon_consumed: 2.8934, delta_consumed: 1e-5, participating_nodes: 4, total_nodes: 4, avg_latency_ms: 124.5 })
    })
      .then((r) => r.json())
      .then((d) => d.data && setSlaAudit(d.data))
      .catch(() => {});

    // Drift Detection
    fetch('/api/v1/governance/drift', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ node_id: 'hospital_alpha', reference_data: [1.0, 2.0, 3.0], current_data: [1.02, 2.01, 3.03] })
    })
      .then((r) => r.json())
      .then((d) => d.data && setDriftMetrics(d.data))
      .catch(() => {});

    // User Profile
    fetch('/api/v1/auth/me')
      .then((r) => r.json())
      .then((d) => d.data && setCurrentUser(d.data))
      .catch(() => {});
  };

  // Handlers for Interactive Actions
  const handleRunInference = async (e) => {
    if (e) e.preventDefault();
    setInfLoading(true);
    setInfResult(null);
    try {
      const res = await fetch('/api/v1/inference/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_id: infSubjectId,
          model_version: 'v2.1.0-dp-prod'
        })
      });
      const data = await res.json();
      if (data.data) {
        setInfResult(data.data);
        fetchAllData();
      } else {
        setInfResult({ error: data.detail || 'Inference execution failed' });
      }
    } catch (err) {
      setInfResult({ error: 'Failed to execute 3D MONAI inference request' });
    } finally {
      setInfLoading(false);
    }
  };

  const handleExportModel = async (e) => {
    e.preventDefault();
    setExpLoading(true);
    setExpResult(null);
    try {
      const res = await fetch('/api/v1/export/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: expModelId,
          version: expVersion
        })
      });
      const data = await res.json();
      if (data.data) {
        setExpResult(data.data);
      }
    } catch (err) {
      setExpResult({ error: 'Failed to export model' });
    } finally {
      setExpLoading(false);
    }
  };

  const handleTriggerSelfHealing = async (e) => {
    e.preventDefault();
    setShLoading(true);
    setShResult(null);
    try {
      const res = await fetch('/api/v1/autonomous/self-healing/recover', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ node_id: shNodeId, failure_type: shReason })
      });
      const data = await res.json();
      if (data.data) {
        setShResult(data.data);
      }
    } catch (err) {
      setShResult({ error: 'Failed to trigger self-healing recovery' });
    } finally {
      setShLoading(false);
    }
  };

  // Nav Items Definition
  const navItems = [
    { id: 'overview', label: 'Overview & Telemetry', icon: LayoutDashboard, category: 'CORE PLATFORM' },
    { id: 'fl_training', label: 'FL Training & Strategies', icon: Activity, category: 'CORE PLATFORM' },
    { id: 'inference_exporter', label: '3D Inference & Visualizer', icon: Brain, category: 'MODEL HUB' },
    { id: 'governance_sla', label: 'Governance & SLA Audit', icon: Shield, category: 'GOVERNANCE' },
    { id: 'autonomous_os', label: 'Autonomous OS & RCA', icon: Cpu, category: 'GOVERNANCE' },
    { id: 'dataset_lineage', label: 'Dataset & Lineage', icon: Database, category: 'DATA PLATFORM' },
    { id: 'benchmarks_artifacts', label: 'Benchmarks & Artifacts', icon: Award, category: 'DATA PLATFORM' },
    { id: 'security_audit', label: 'Security & Audit Trail', icon: Lock, category: 'SECURITY' }
  ];

  // Handlers for Operations Control Actions
  const handleStartExp = async (expId) => {
    try {
      await fetch(`/api/v1/experiments/${expId}/start`, { method: 'POST' });
      fetchAllData();
    } catch (e) {}
  };

  const handlePauseExp = async (expId) => {
    try {
      await fetch(`/api/v1/experiments/${expId}/pause`, { method: 'POST' });
      fetchAllData();
    } catch (e) {}
  };

  const handleResumeExp = async (expId) => {
    try {
      await fetch(`/api/v1/experiments/${expId}/resume`, { method: 'POST' });
      fetchAllData();
    } catch (e) {}
  };

  const handleCancelExp = async (expId) => {
    try {
      await fetch(`/api/v1/experiments/${expId}/cancel`, { method: 'POST' });
      fetchAllData();
    } catch (e) {}
  };

  const handleRestartOS = async () => {
    try {
      await fetch('/api/v1/runtime/restart', { method: 'POST' });
      fetchAllData();
    } catch (e) {}
  };

  return (
    <div className="app-container">
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="brand-icon">
            <Brain size={22} />
          </div>
          <div>
            <div className="brand-title">FedMed OS</div>
            <div className="brand-version">PRIVACY-PRESERVING BRAIN TUMOR AI</div>
          </div>
        </div>

        <div className="sidebar-menu">
          {['CORE PLATFORM', 'MODEL HUB', 'GOVERNANCE', 'DATA PLATFORM', 'SECURITY'].map((cat) => (
            <div key={cat}>
              <div className="menu-group-title">{cat}</div>
              {navItems
                .filter((item) => item.category === cat)
                .map((item) => {
                  const Icon = item.icon;
                  const isActive = activeTab === item.id;
                  return (
                    <div
                      key={item.id}
                      className={`menu-item ${isActive ? 'active' : ''}`}
                      onClick={() => setActiveTab(item.id)}
                    >
                      <Icon size={18} />
                      <span>{item.label}</span>
                    </div>
                  );
                })}
            </div>
          ))}
        </div>

        {/* User Badge */}
        <div style={{ padding: '1rem', borderTop: '1px solid var(--border-subtle)', background: 'rgba(0,0,0,0.2)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{ width: '32px', height: '32px', borderRadius: '50%', background: 'linear-gradient(135deg, #0284c7, #818cf8)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.8rem', fontWeight: '700' }}>
              FP
            </div>
            <div>
              <div style={{ fontSize: '0.8rem', fontWeight: '600' }}>{currentUser?.username || 'Clinician Admin'}</div>
              <div style={{ fontSize: '0.7rem', color: '#34d399' }}>● Verified Silo Access</div>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="main-wrapper">
        {/* Top Navbar */}
        <header className="top-navbar">
          <div className="page-title">
            <Layers size={18} color="#38bdf8" />
            <span>{navItems.find((item) => item.id === activeTab)?.label || 'Overview'}</span>
          </div>

          <div className="nav-actions">
            <div className="status-indicator">
              <span className="status-dot"></span>
              <span>DP-SGD PRIVACY LEDGER ACTIVE (ε = 2.8934, δ = 1e-5)</span>
            </div>

            <button className="btn-action" onClick={fetchAllData} style={{ background: 'rgba(255,255,255,0.06)', color: 'var(--text-main)', border: '1px solid var(--border-subtle)' }}>
              <RefreshCw size={14} /> Refresh
            </button>
          </div>
        </header>

        {/* Content Body */}
        <main className="content-area">
          {/* TAB 1: OVERVIEW & TELEMETRY */}
          {activeTab === 'overview' && (
            <div>
              {/* Stat Cards */}
              <div className="card-grid">
                <div className="stat-card">
                  <div className="stat-header">
                    <span className="stat-title">Privacy Guarantee</span>
                    <div className="stat-icon-wrap"><Shield size={16} color="#34d399" /></div>
                  </div>
                  <div className="stat-value" style={{ color: '#34d399' }}>ε = 2.8934</div>
                  <div className="stat-subtitle">Poisson DP-SGD (δ = 1e-5, T = 4,720)</div>
                </div>

                <div className="stat-card">
                  <div className="stat-header">
                    <span className="stat-title">Validation Dice</span>
                    <div className="stat-icon-wrap"><Award size={16} color="#38bdf8" /></div>
                  </div>
                  <div className="stat-value" style={{ color: '#38bdf8' }}>0.0800</div>
                  <div className="stat-subtitle">TC: 0.2177 | ET: 0.0147 | WT: 0.0076</div>
                </div>

                <div className="stat-card">
                  <div className="stat-header">
                    <span className="stat-title">Locked Test Dice</span>
                    <div className="stat-icon-wrap"><CheckCircle size={16} color="#c084fc" /></div>
                  </div>
                  <div className="stat-value" style={{ color: '#c084fc' }}>0.0741</div>
                  <div className="stat-subtitle">204 Subjects (TC: 0.2054, Loss: 0.9628)</div>
                </div>

                <div className="stat-card">
                  <div className="stat-header">
                    <span className="stat-title">Hospital Silos</span>
                    <div className="stat-icon-wrap"><Server size={16} /></div>
                  </div>
                  <div className="stat-value">4 / 4</div>
                  <div className="stat-subtitle">Alpha, Beta, Gamma, Delta (944 Train)</div>
                </div>
              </div>

              {/* Model Specification Banner */}
              <div className="panel-card" style={{ marginTop: '1.25rem' }}>
                <div className="panel-header">
                  <div className="panel-title">
                    <Brain size={18} color="#38bdf8" /> Production Model Specification & Clinical Lineage
                  </div>
                  <span className="tag-badge">FROZEN CANDIDATE (PHASE 11)</span>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem', fontSize: '0.85rem' }}>
                  <div>
                    <div style={{ color: 'var(--text-dim)', fontSize: '0.75rem', fontWeight: '600' }}>ARCHITECTURE</div>
                    <div style={{ fontWeight: '700', marginTop: '0.2rem' }}>MONAI 3D U-Net (128³)</div>
                    <div style={{ color: 'var(--text-dim)', fontSize: '0.75rem' }}>4,810,074 Trainable Params</div>
                  </div>
                  <div>
                    <div style={{ color: 'var(--text-dim)', fontSize: '0.75rem', fontWeight: '600' }}>CHECKPOINT SHA-256</div>
                    <div className="code-inline" style={{ fontSize: '0.725rem', marginTop: '0.2rem', display: 'inline-block' }}>f6cd18dc5e05595c...</div>
                    <div style={{ color: 'var(--text-dim)', fontSize: '0.75rem' }}>checkpoints/final/fedmed_dp_final_model.pt</div>
                  </div>
                  <div>
                    <div style={{ color: 'var(--text-dim)', fontSize: '0.75rem', fontWeight: '600' }}>OPTIMIZER & DP ENGINE</div>
                    <div style={{ fontWeight: '700', marginTop: '0.2rem' }}>SGD + Momentum (μ=0.9, η=1e-3)</div>
                    <div style={{ color: 'var(--text-dim)', fontSize: '0.75rem' }}>Clipping C=0.06, Noise σ=0.87</div>
                  </div>
                  <div>
                    <div style={{ color: 'var(--text-dim)', fontSize: '0.75rem', fontWeight: '600' }}>CLINICAL DATASET</div>
                    <div style={{ fontWeight: '700', marginTop: '0.2rem' }}>BraTS-GLI 2024 (1,350 Cohort)</div>
                    <div style={{ color: 'var(--text-dim)', fontSize: '0.75rem' }}>944 Train | 202 Val | 204 Test</div>
                  </div>
                </div>
              </div>

              {/* Training Loss & Dice Chart Panel */}
              <div className="panel-grid-2" style={{ marginTop: '1.25rem' }}>
                <div className="panel-card">
                  <div className="panel-header">
                    <div className="panel-title">
                      <Activity size={18} color="#38bdf8" /> 20-Round Federated Validation Loss
                    </div>
                    <span className="tag-badge">MONOTONIC CONVERGENCE</span>
                  </div>
                  <div style={{ height: '220px', width: '100%' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={[
                        { round: 1, val_loss: 0.9875 }, { round: 3, val_loss: 0.9832 },
                        { round: 5, val_loss: 0.9813 }, { round: 7, val_loss: 0.9798 },
                        { round: 10, val_loss: 0.9772 }, { round: 13, val_loss: 0.9740 },
                        { round: 16, val_loss: 0.9695 }, { round: 18, val_loss: 0.9649 },
                        { round: 20, val_loss: 0.9604 }
                      ]}>
                        <defs>
                          <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#38bdf8" stopOpacity={0.4} />
                            <stop offset="95%" stopColor="#38bdf8" stopOpacity={0.0} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                        <XAxis dataKey="round" stroke="#64748b" tick={{ fontSize: 11 }} />
                        <YAxis stroke="#64748b" domain={[0.95, 0.99]} tick={{ fontSize: 11 }} />
                        <Tooltip contentStyle={{ background: '#0f172a', borderColor: 'rgba(255,255,255,0.1)' }} />
                        <Area type="monotone" dataKey="val_loss" stroke="#38bdf8" strokeWidth={2} fill="url(#lossGrad)" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="panel-card">
                  <div className="panel-header">
                    <div className="panel-title">
                      <TrendingUp size={18} color="#34d399" /> Sub-Region Dice Progression (TC / ET / WT)
                    </div>
                    <span className="tag-badge">TC: 0.2177 (4.98x GAIN)</span>
                  </div>
                  <div style={{ height: '220px', width: '100%' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={[
                        { round: 1, tc: 0.0519, macro: 0.0219, et: 0.0092 },
                        { round: 5, tc: 0.0841, macro: 0.0338, et: 0.0115 },
                        { round: 10, tc: 0.1166, macro: 0.0453, et: 0.0128 },
                        { round: 15, tc: 0.1579, macro: 0.0597, et: 0.0140 },
                        { round: 20, tc: 0.2177, macro: 0.0800, et: 0.0147 }
                      ]}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                        <XAxis dataKey="round" stroke="#64748b" tick={{ fontSize: 11 }} />
                        <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                        <Tooltip contentStyle={{ background: '#0f172a', borderColor: 'rgba(255,255,255,0.1)' }} />
                        <Legend wrapperStyle={{ fontSize: '11px' }} />
                        <Line type="monotone" dataKey="tc" name="Tumor Core (TC)" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} />
                        <Line type="monotone" dataKey="macro" name="Macro Dice" stroke="#34d399" strokeWidth={2} dot={{ r: 3 }} />
                        <Line type="monotone" dataKey="et" name="Enhancing (ET)" stroke="#eab308" strokeWidth={2} dot={{ r: 2 }} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: FL TRAINING & STRATEGIES */}
          {activeTab === 'fl_training' && (
            <div className="panel-card">
              <div className="panel-header">
                <div className="panel-title">
                  <Activity size={18} color="#38bdf8" /> Multi-Silo Federated Learning Framework
                </div>
                <span className="tag-badge">FEDAVG + DP-SGD</span>
              </div>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '1rem' }}>
                FedMed coordinates privacy-preserving weight aggregation across 4 hospital nodes with strict per-sample gradient norm clipping (C = 0.06) and Gaussian noise addition (σ = 0.87).
              </p>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Silo ID</th>
                    <th>Cohort Size</th>
                    <th>Gradient Clipping (C)</th>
                    <th>Noise Multiplier (σ)</th>
                    <th>Poisson Rate (q)</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><b>Hospital Alpha</b></td>
                    <td>236 subjects</td>
                    <td><span className="code-inline">0.06</span></td>
                    <td><span className="code-inline">0.87</span></td>
                    <td>1 / 236</td>
                    <td><span style={{ color: '#34d399' }}>● COMPLETED (20/20)</span></td>
                  </tr>
                  <tr>
                    <td><b>Hospital Beta</b></td>
                    <td>236 subjects</td>
                    <td><span className="code-inline">0.06</span></td>
                    <td><span className="code-inline">0.87</span></td>
                    <td>1 / 236</td>
                    <td><span style={{ color: '#34d399' }}>● COMPLETED (20/20)</span></td>
                  </tr>
                  <tr>
                    <td><b>Hospital Gamma</b></td>
                    <td>236 subjects</td>
                    <td><span className="code-inline">0.06</span></td>
                    <td><span className="code-inline">0.87</span></td>
                    <td>1 / 236</td>
                    <td><span style={{ color: '#34d399' }}>● COMPLETED (20/20)</span></td>
                  </tr>
                  <tr>
                    <td><b>Hospital Delta</b></td>
                    <td>236 subjects</td>
                    <td><span className="code-inline">0.06</span></td>
                    <td><span className="code-inline">0.87</span></td>
                    <td>1 / 236</td>
                    <td><span style={{ color: '#34d399' }}>● COMPLETED (20/20)</span></td>
                  </tr>
                </tbody>
              </table>
            </div>
          )}

          {/* TAB 3: 3D MONAI INFERENCE & VISUALIZER */}
          {activeTab === 'inference_exporter' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              {/* Interactive 3D Inference & Case Selection */}
              <div className="panel-card">
                <div className="panel-header">
                  <div className="panel-title">
                    <Brain size={18} color="#34d399" /> Clinical Decision Support — 3D Brain Tumor Segmentation
                  </div>
                  <span className="tag-badge">MONAI 3D U-NET (DP-SGD)</span>
                </div>

                <form onSubmit={handleRunInference}>
                  <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '1rem', alignItems: 'flex-end' }}>
                    <div className="form-group" style={{ marginBottom: 0 }}>
                      <label className="form-label">Select Patient MRI Case (Safe Training Cohort)</label>
                      <select
                        className="form-select"
                        value={infSubjectId}
                        onChange={(e) => setInfSubjectId(e.target.value)}
                      >
                        {availableCases.length > 0 ? (
                          availableCases.map((c) => (
                            <option key={c.patient_id} value={c.patient_id}>
                              {c.patient_id} — {c.description} ({c.cohort})
                            </option>
                          ))
                        ) : (
                          <>
                            <option value="BraTS-GLI-00005-100">BraTS-GLI-00005-100 — Adult Glioma (Silo Alpha)</option>
                            <option value="BraTS-GLI-00006-100">BraTS-GLI-00006-100 — Adult Glioma (Silo Alpha)</option>
                            <option value="BraTS-GLI-00008-101">BraTS-GLI-00008-101 — Adult Glioma (Silo Beta)</option>
                            <option value="BraTS-GLI-00020-100">BraTS-GLI-00020-100 — Adult Glioma (Validation)</option>
                          </>
                        )}
                      </select>
                    </div>

                    <button className="btn-action primary" type="submit" disabled={infLoading} style={{ height: '42px', justifyContent: 'center' }}>
                      {infLoading ? <RefreshCw size={16} className="spin" /> : <Play size={16} />}
                      {infLoading ? 'Running 3D Inference...' : 'Run Segmentation Inference'}
                    </button>
                  </div>
                </form>

                {/* Inference Error */}
                {infResult?.error && (
                  <div style={{ marginTop: '1rem', padding: '0.75rem', background: 'rgba(251, 113, 133, 0.1)', border: '1px solid rgba(251, 113, 133, 0.3)', borderRadius: '0.5rem', color: '#fb7185', fontSize: '0.85rem' }}>
                    <AlertTriangle size={16} style={{ display: 'inline', marginRight: '0.5rem' }} />
                    {infResult.error}
                  </div>
                )}
              </div>

              {/* Segmentation Results & Visualizer */}
              {infResult?.status === 'SUCCESS' && (
                <div className="panel-card">
                  <div className="panel-header">
                    <div className="panel-title">
                      <Eye size={18} color="#38bdf8" /> Multi-Planar Segmentation Overlays ({infResult.patient_id})
                    </div>
                    <span className="tag-badge" style={{ color: '#34d399' }}>LATENCY: {infResult.inference_time_ms} MS</span>
                  </div>

                  {/* Volume Counters */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '1.25rem' }}>
                    <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', padding: '0.75rem', borderRadius: '0.5rem' }}>
                      <div style={{ color: '#ef4444', fontSize: '0.75rem', fontWeight: '700' }}>TUMOR CORE (TC)</div>
                      <div style={{ fontSize: '1.25rem', fontWeight: '800', fontFamily: 'var(--font-mono)' }}>{infResult.tumor_volumes_mm3?.tumor_core_tc?.toLocaleString() || 0} mm³</div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>Necrotic Core + Active Tumor</div>
                    </div>

                    <div style={{ background: 'rgba(234, 179, 8, 0.1)', border: '1px solid rgba(234, 179, 8, 0.3)', padding: '0.75rem', borderRadius: '0.5rem' }}>
                      <div style={{ color: '#eab308', fontSize: '0.75rem', fontWeight: '700' }}>ENHANCING TUMOR (ET)</div>
                      <div style={{ fontSize: '1.25rem', fontWeight: '800', fontFamily: 'var(--font-mono)' }}>{infResult.tumor_volumes_mm3?.enhancing_tumor_et?.toLocaleString() || 0} mm³</div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>Active Vascular Rim</div>
                    </div>

                    <div style={{ background: 'rgba(34, 197, 94, 0.1)', border: '1px solid rgba(34, 197, 94, 0.3)', padding: '0.75rem', borderRadius: '0.5rem' }}>
                      <div style={{ color: '#22c55e', fontSize: '0.75rem', fontWeight: '700' }}>WHOLE TUMOR (WT)</div>
                      <div style={{ fontSize: '1.25rem', fontWeight: '800', fontFamily: 'var(--font-mono)' }}>{infResult.tumor_volumes_mm3?.whole_tumor_wt?.toLocaleString() || 0} mm³</div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>Total Tumor + Peritumoral Edema</div>
                    </div>
                  </div>

                  {/* Multi-Planar Slice Images */}
                  {infResult.visual_slices && (
                    <div className="visualizer-grid">
                      <div className="slice-card">
                        <div className="slice-title">Axial Plane (Transverse)</div>
                        <img
                          src={`data:image/png;base64,${infResult.visual_slices.axial}`}
                          alt="Axial MRI Slice"
                          className="slice-img"
                        />
                      </div>

                      <div className="slice-card">
                        <div className="slice-title">Coronal Plane (Frontal)</div>
                        <img
                          src={`data:image/png;base64,${infResult.visual_slices.coronal}`}
                          alt="Coronal MRI Slice"
                          className="slice-img"
                        />
                      </div>

                      <div className="slice-card">
                        <div className="slice-title">Sagittal Plane (Lateral)</div>
                        <img
                          src={`data:image/png;base64,${infResult.visual_slices.sagittal}`}
                          alt="Sagittal MRI Slice"
                          className="slice-img"
                        />
                      </div>
                    </div>
                  )}

                  {/* Timing & Privacy Footer */}
                  <div style={{ marginTop: '1.25rem', paddingTop: '1rem', borderTop: '1px solid var(--border-subtle)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.75rem', color: 'var(--text-dim)' }}>
                    <div>
                      ⚡ Preprocessing: <b>{infResult.timing_ms?.preprocessing_time_ms} ms</b> | Neural Pass: <b>{infResult.timing_ms?.inference_time_ms} ms</b> | Total: <b>{infResult.timing_ms?.total_inference_time_ms} ms</b>
                    </div>
                    <div>
                      🔒 Evaluated with Frozen DP Model (<span className="code-inline">ε = 2.8934</span>)
                    </div>
                  </div>
                </div>
              )}

              {/* Production Model Exporter Form */}
              <div className="panel-card">
                <div className="panel-header">
                  <div className="panel-title">
                    <Download size={18} color="#38bdf8" /> Production Model Packaging & TorchScript Export
                  </div>
                  <span className="tag-badge">STANDALONE BUNDLE</span>
                </div>

                <form onSubmit={handleExportModel}>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr auto', gap: '1rem', alignItems: 'flex-end' }}>
                    <div className="form-group" style={{ marginBottom: 0 }}>
                      <label className="form-label">Model Identifier</label>
                      <input className="form-input" value={expModelId} onChange={(e) => setExpModelId(e.target.value)} required />
                    </div>

                    <div className="form-group" style={{ marginBottom: 0 }}>
                      <label className="form-label">Release Version Tag</label>
                      <input className="form-input" value={expVersion} onChange={(e) => setExpVersion(e.target.value)} required />
                    </div>

                    <button className="btn-action primary" type="submit" disabled={expLoading} style={{ height: '42px' }}>
                      {expLoading ? <RefreshCw size={14} className="spin" /> : <Download size={14} />}
                      {expLoading ? 'Exporting...' : 'Export Package (.pt)'}
                    </button>
                  </div>
                </form>

                {expResult && (
                  <div style={{ marginTop: '1rem', padding: '0.75rem', background: 'rgba(56, 189, 248, 0.08)', borderRadius: '0.5rem', border: '1px solid rgba(56, 189, 248, 0.25)', fontSize: '0.8rem' }}>
                    ✓ Production Package Exported: <code className="code-inline">{expResult.export_path}</code> (SHA256: <code>{expResult.sha256_checksum?.slice(0, 16)}...</code>)
                  </div>
                )}
              </div>
            </div>
          )}

          {/* TAB 4: GOVERNANCE & SLA AUDIT */}
          {activeTab === 'governance_sla' && (
            <div className="panel-card">
              <div className="panel-header">
                <div className="panel-title">
                  <Shield size={18} color="#34d399" /> Enterprise Privacy & Governance SLA Verification
                </div>
                <span className="tag-badge">RÉNYI DP LEDGER (POISSON RDP)</span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem', marginBottom: '1.25rem' }}>
                <div className="stat-card">
                  <div className="stat-title">Analytical Epsilon</div>
                  <div className="stat-value" style={{ color: '#34d399' }}>2.8934</div>
                  <div className="stat-subtitle">Strict (ε, δ)-DP Bound at α=7</div>
                </div>
                <div className="stat-card">
                  <div className="stat-title">Target Delta</div>
                  <div className="stat-value">1.0 × 10⁻⁵</div>
                  <div className="stat-subtitle">Cryptographic Failure Prob</div>
                </div>
                <div className="stat-card">
                  <div className="stat-title">Clipping Threshold (C)</div>
                  <div className="stat-value">0.06</div>
                  <div className="stat-subtitle">Calibrated Norm Bound</div>
                </div>
                <div className="stat-card">
                  <div className="stat-title">Noise Multiplier (σ)</div>
                  <div className="stat-value">0.87</div>
                  <div className="stat-subtitle">Physical Noise Std = 0.0522</div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 5: AUTONOMOUS OS */}
          {activeTab === 'autonomous_os' && (
            <div className="panel-card">
              <div className="panel-header">
                <div className="panel-title">
                  <Cpu size={18} color="#c084fc" /> Autonomous Self-Healing & Root-Cause Analysis
                </div>
                <span className="tag-badge">ACTIVE AGENTIC SYSTEM</span>
              </div>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '1rem' }}>
                FedMed monitors hospital silo heartbeat telemetry, stragglers, and feature drift, triggering automated healing workflows when anomalies are detected.
              </p>
              <form onSubmit={handleTriggerSelfHealing} style={{ display: 'grid', gridTemplateColumns: '1fr 1fr auto', gap: '1rem', alignItems: 'flex-end' }}>
                <div className="form-group" style={{ marginBottom: 0 }}>
                  <label className="form-label">Hospital Silo Node</label>
                  <select className="form-select" value={shNodeId} onChange={(e) => setShNodeId(e.target.value)}>
                    <option value="hospital_alpha">Hospital Alpha</option>
                    <option value="hospital_beta">Hospital Beta</option>
                    <option value="hospital_gamma">Hospital Gamma</option>
                    <option value="hospital_delta">Hospital Delta</option>
                  </select>
                </div>
                <div className="form-group" style={{ marginBottom: 0 }}>
                  <label className="form-label">Recovery Action</label>
                  <input className="form-input" value={shReason} onChange={(e) => setShReason(e.target.value)} />
                </div>
                <button className="btn-action primary" type="submit" disabled={shLoading} style={{ height: '42px' }}>
                  {shLoading ? 'Executing...' : 'Trigger Self-Healing'}
                </button>
              </form>
              {shResult && (
                <div style={{ marginTop: '1rem', padding: '0.75rem', background: 'rgba(192, 132, 252, 0.08)', borderRadius: '0.5rem', border: '1px solid rgba(192, 132, 252, 0.25)', fontSize: '0.8rem' }}>
                  ✓ Recovery Workflow Finished: Node <b>{shResult.node_id}</b> restored. Status: <span style={{ color: '#34d399' }}>HEALTHY</span>
                </div>
              )}
            </div>
          )}

          {/* TAB 6: DATASET & LINEAGE */}
          {activeTab === 'dataset_lineage' && (
            <div className="panel-card">
              <div className="panel-header">
                <div className="panel-title">
                  <Database size={18} color="#38bdf8" /> Dataset Registry & Split Cryptographic Hashes
                </div>
                <span className="tag-badge">BRATS-GLI 2024</span>
              </div>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Cohort Split</th>
                    <th>Subject Count</th>
                    <th>Spatial Resolution</th>
                    <th>Modalities</th>
                    <th>Firewall Status</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><b>Training Partition</b></td>
                    <td>944 subjects</td>
                    <td>128 × 128 × 128</td>
                    <td>T1, T1CE, T2, FLAIR</td>
                    <td><span style={{ color: '#34d399' }}>● 4 Silos (236 each)</span></td>
                  </tr>
                  <tr>
                    <td><b>Validation Cohort</b></td>
                    <td>202 subjects</td>
                    <td>128 × 128 × 128</td>
                    <td>T1, T1CE, T2, FLAIR</td>
                    <td><span style={{ color: '#38bdf8' }}>● Model Selection</span></td>
                  </tr>
                  <tr>
                    <td><b>Locked Test Cohort</b></td>
                    <td>204 subjects</td>
                    <td>128 × 128 × 128</td>
                    <td>T1, T1CE, T2, FLAIR</td>
                    <td><span style={{ color: '#c084fc' }}>● Evaluated Once (Macro: 0.0741)</span></td>
                  </tr>
                </tbody>
              </table>
            </div>
          )}

          {/* TAB 7: BENCHMARKS & ARTIFACTS */}
          {activeTab === 'benchmarks_artifacts' && (
            <div className="panel-card">
              <div className="panel-header">
                <div className="panel-title">
                  <Award size={18} color="#fbbf24" /> Scientific Benchmark Comparative Summary
                </div>
                <span className="tag-badge">EXPERIMENTS D → D2.1</span>
              </div>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Experiment</th>
                    <th>Optimizer</th>
                    <th>Clipping (C)</th>
                    <th>Epsilon (ε)</th>
                    <th>Val Loss</th>
                    <th>Macro Dice</th>
                    <th>TC Dice</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><b>Non-Private FedAvg</b></td>
                    <td>Adam (1e-4)</td>
                    <td>None</td>
                    <td>∞</td>
                    <td>0.5872</td>
                    <td>0.3815</td>
                    <td>0.1878</td>
                  </tr>
                  <tr>
                    <td><b>Experiment D (DP Base)</b></td>
                    <td>Adam (1e-4)</td>
                    <td>1.0</td>
                    <td>2.8934</td>
                    <td>0.9888</td>
                    <td>0.0190</td>
                    <td>0.0437</td>
                  </tr>
                  <tr>
                    <td><b>Experiment D1</b></td>
                    <td>Adam (1e-4)</td>
                    <td>0.06</td>
                    <td>1.9636</td>
                    <td>0.9883</td>
                    <td>0.0205</td>
                    <td>0.0479</td>
                  </tr>
                  <tr>
                    <td><b>Experiment D2</b></td>
                    <td>SGD+Mom (1e-4)</td>
                    <td>0.06</td>
                    <td>1.9636</td>
                    <td>0.9902</td>
                    <td>0.0210</td>
                    <td>0.0498</td>
                  </tr>
                  <tr style={{ background: 'rgba(52, 211, 153, 0.08)', fontWeight: '700' }}>
                    <td><b>Experiment D2.1 (Val)</b></td>
                    <td>SGD+Mom (1e-3)</td>
                    <td>0.06</td>
                    <td>2.8934</td>
                    <td>0.9604</td>
                    <td style={{ color: '#34d399' }}>0.0800 (4.2x)</td>
                    <td style={{ color: '#34d399' }}>0.2177 (5.0x)</td>
                  </tr>
                  <tr style={{ background: 'rgba(192, 132, 252, 0.08)', fontWeight: '700' }}>
                    <td><b>Experiment D2.1 (Locked Test)</b></td>
                    <td>SGD+Mom (1e-3)</td>
                    <td>0.06</td>
                    <td>2.8934</td>
                    <td>0.9628</td>
                    <td style={{ color: '#c084fc' }}>0.0741 (3.9x)</td>
                    <td style={{ color: '#c084fc' }}>0.2054 (4.7x)</td>
                  </tr>
                </tbody>
              </table>
            </div>
          )}

          {/* TAB 8: SECURITY & AUDIT TRAIL */}
          {activeTab === 'security_audit' && (
            <div className="panel-card">
              <div className="panel-header">
                <div className="panel-title">
                  <Lock size={18} color="#34d399" /> Immutable Security & Audit Trail
                </div>
                <span className="tag-badge">SHA-256 VERIFIED</span>
              </div>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Timestamp</th>
                    <th>Event Description</th>
                    <th>Target Artifact</th>
                    <th>Integrity Hash</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>2026-08-18T14:11:07Z</td>
                    <td>Locked Test Evaluation (204 Subjects)</td>
                    <td>reports/final/fedmed_final_test_results.json</td>
                    <td><code className="code-inline">88c679b37c...</code></td>
                  </tr>
                  <tr>
                    <td>2026-08-18T09:35:21Z</td>
                    <td>Final Candidate Frozen (Stage C Round 20)</td>
                    <td>checkpoints/final/fedmed_dp_final_model.pt</td>
                    <td><code className="code-inline">f6cd18dc5e...</code></td>
                  </tr>
                  <tr>
                    <td>2026-08-18T04:24:13Z</td>
                    <td>Pre-Launch Gate Passed (19 Protected Baselines)</td>
                    <td>reports/real_brats2024/dp_d2_1_stage_c_prelaunch_gate.json</td>
                    <td><code className="code-inline">c1229d6cb0...</code></td>
                  </tr>
                </tbody>
              </table>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
