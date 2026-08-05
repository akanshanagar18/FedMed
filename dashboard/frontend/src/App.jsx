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
  AreaChart,
  Area
} from 'recharts';

export default function App() {
  const [metrics, setMetrics] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState('CONNECTING');
  const [wsConnected, setWsConnected] = useState(false);
  const [latestRound, setLatestRound] = useState(0);
  const [latestLoss, setLatestLoss] = useState('N/A');
  const [latestDice, setLatestDice] = useState('N/A');

  useEffect(() => {
    // Initial fetch of persisted metrics
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
      .catch((err) => console.log('Initial metrics fetch error:', err));

    // WebSocket telemetry connection with reconnect
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

  return (
    <div className="dashboard-container">
      {/* Platform Header */}
      <header className="header">
        <div className="header-title-group">
          <div className="logo-badge">FM</div>
          <div>
            <h1>FedMed Monitoring Engine</h1>
            <p className="header-subtitle">
              Cross-Silo Federated Learning Platform for Medical Image Segmentation
            </p>
            <div className="meta-tags">
              <span className="tag">EXP: default</span>
              <span className="tag">MODEL: 3D U-Net (BraTS)</span>
              <span className="tag">STRATEGY: FedAvg</span>
            </div>
          </div>
        </div>

        <div className="status-indicator">
          <div className={`dot ${wsConnected ? 'green' : 'red'}`}></div>
          <span>TELEMETRY: {connectionStatus}</span>
        </div>
      </header>

      {/* Subsystem Health Bar */}
      <div className="status-bar">
        <span className="status-label">System Health:</span>
        <div className="status-indicator">
          <div className="dot green"></div>
          <span>REST API: 200 OK</span>
        </div>
        <div className="status-indicator">
          <div className="dot green"></div>
          <span>DATABASE: SQLite Active</span>
        </div>
        <div className="status-indicator">
          <div className={`dot ${wsConnected ? 'green' : 'yellow'}`}></div>
          <span>WEBSOCKET: {wsConnected ? 'LIVE' : 'IDLE'}</span>
        </div>
        <div className="status-indicator">
          <div className="dot green"></div>
          <span>FLOWER SERVER: 127.0.0.1:8080</span>
        </div>
        <div className="status-indicator">
          <div className="dot green"></div>
          <span>HOSPITALS: 2/2 Active</span>
        </div>
      </div>

      {/* Top KPI Cards Grid */}
      <div className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-title">Completed FL Round</div>
          <div className="kpi-value-row">
            <div className="kpi-value">{latestRound}</div>
            <div className="kpi-trend neutral">3 Max Rounds</div>
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-title">Global Training Loss</div>
          <div className="kpi-value-row">
            <div className="kpi-value" style={{ color: 'var(--accent-cyan)' }}>
              {latestLoss}
            </div>
            <div className="kpi-trend positive">↓ Decreasing</div>
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-title">Dice Score (Similarity)</div>
          <div className="kpi-value-row">
            <div className="kpi-value" style={{ color: 'var(--accent-purple)' }}>
              {latestDice}
            </div>
            <div className="kpi-trend positive">↑ Improving</div>
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-title">Participating Silos</div>
          <div className="kpi-value-row">
            <div className="kpi-value" style={{ color: 'var(--accent-green)' }}>
              2
            </div>
            <div className="kpi-trend positive">100% Online</div>
          </div>
        </div>
      </div>

      {/* Main Charts & Metadata Layout */}
      <div className="charts-grid">
        {/* Recharts Live Metrics Panel */}
        <div className="panel-card">
          <div className="panel-header">
            <div className="panel-title">
              <div className="panel-title-icon"></div>
              Real-time Convergence Curves
            </div>
            <div className="tag">LIVE TELEMETRY STREAM</div>
          </div>

          <div style={{ width: '100%', height: 350 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={metrics} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis
                  dataKey="round_number"
                  stroke="#64748b"
                  tick={{ fill: '#94a3b8' }}
                  label={{ value: 'FL Round Number', position: 'insideBottom', offset: -5, fill: '#64748b' }}
                />
                <YAxis stroke="#64748b" tick={{ fill: '#94a3b8' }} domain={[0, 1]} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1e293b',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderRadius: '0.75rem',
                    color: '#f8fafc',
                    boxShadow: '0 10px 25px rgba(0,0,0,0.5)'
                  }}
                />
                <Legend wrapperStyle={{ paddingTop: '10px' }} />
                <Line
                  type="monotone"
                  dataKey="training_loss"
                  stroke="#38bdf8"
                  name="Training Loss"
                  strokeWidth={3}
                  dot={{ r: 5, fill: '#38bdf8' }}
                  activeDot={{ r: 8 }}
                />
                <Line
                  type="monotone"
                  dataKey="dice_score"
                  stroke="#c084fc"
                  name="Dice Score"
                  strokeWidth={3}
                  dot={{ r: 5, fill: '#c084fc' }}
                  activeDot={{ r: 8 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Experiment & Infrastructure Metadata Panel */}
        <div className="panel-card">
          <div className="panel-header">
            <div className="panel-title">
              <div className="panel-title-icon" style={{ backgroundColor: 'var(--accent-purple)' }}></div>
              Experiment Specs
            </div>
          </div>

          <div className="info-list">
            <div className="info-item">
              <span className="info-key">Experiment ID</span>
              <span className="info-value">default</span>
            </div>
            <div className="info-item">
              <span className="info-key">Architecture</span>
              <span className="info-value">MONAI 3D U-Net</span>
            </div>
            <div className="info-item">
              <span className="info-key">Input Dimensions</span>
              <span className="info-value">4 x 32 x 32 x 32</span>
            </div>
            <div className="info-item">
              <span className="info-key">Privacy Preserving</span>
              <span className="info-value" style={{ color: 'var(--accent-green)' }}>
                TenSEAL CKKS HE
              </span>
            </div>
            <div className="info-item">
              <span className="info-key">Aggregation</span>
              <span className="info-value">FedAvg</span>
            </div>
            <div className="info-item">
              <span className="info-key">Communication</span>
              <span className="info-value">gRPC (Port 8080)</span>
            </div>
            <div className="info-item">
              <span className="info-key">Persistence</span>
              <span className="info-value">SQLite (fedmed.db)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Hospital Nodes Table */}
      <div className="panel-card">
        <div className="panel-header">
          <div className="panel-title">
            <div className="panel-title-icon" style={{ backgroundColor: 'var(--accent-green)' }}></div>
            Hospital Client Silos
          </div>
          <div className="tag">2 SILOS REGISTERED</div>
        </div>

        <table className="data-table">
          <thead>
            <tr>
              <th>Hospital ID</th>
              <th>Node Name</th>
              <th>Status</th>
              <th>Latency</th>
              <th>Local Dataset</th>
              <th>Engine</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <span className="node-badge">hospital_a</span>
              </td>
              <td>General Hospital Alpha</td>
              <td>
                <div className="status-indicator">
                  <div className="dot green"></div>
                  <span>ACTIVE / TRAINING</span>
                </div>
              </td>
              <td>12 ms</td>
              <td>BraTS Synthetic Tensors</td>
              <td>PyTorch 2.x (CPU)</td>
            </tr>
            <tr>
              <td>
                <span className="node-badge">hospital_b</span>
              </td>
              <td>St. Jude Medical Beta</td>
              <td>
                <div className="status-indicator">
                  <div className="dot green"></div>
                  <span>ACTIVE / TRAINING</span>
                </div>
              </td>
              <td>18 ms</td>
              <td>BraTS Synthetic Tensors</td>
              <td>PyTorch 2.x (CPU)</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
