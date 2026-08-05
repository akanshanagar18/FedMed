import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export default function App() {
  const [metrics, setMetrics] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState('Connecting...');
  const [latestRound, setLatestRound] = useState(0);
  const [latestLoss, setLatestLoss] = useState('N/A');
  const [latestDice, setLatestDice] = useState('N/A');

  useEffect(() => {
    // Initial fetch of existing metrics
    fetch('/api/v1/metrics/default')
      .then(res => res.json())
      .then(res => {
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
      .catch(err => console.log('Initial metrics fetch error:', err));

    // Connect to WebSocket telemetry stream
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/v1/telemetry/ws`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      setConnectionStatus('Live Telemetry Connected');
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.event === 'metrics_updated' && msg.data) {
          const newMetric = msg.data;
          setMetrics(prev => {
            const filtered = prev.filter(m => m.round_number !== newMetric.round_number);
            return [...filtered, newMetric].sort((a, b) => a.round_number - b.round_number);
          });
          setLatestRound(newMetric.round_number);
          setLatestLoss(newMetric.training_loss ? newMetric.training_loss.toFixed(4) : 'N/A');
          setLatestDice(newMetric.dice_score ? newMetric.dice_score.toFixed(4) : 'N/A');
        }
      } catch (err) {
        console.error('Error parsing WS message:', err);
      }
    };

    ws.onclose = () => {
      setConnectionStatus('Disconnected (Reconnecting...)');
    };

    return () => ws.close();
  }, []);

  return (
    <div className="dashboard-container">
      <header className="header">
        <div>
          <h1>FedMed Monitoring Engine</h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '0.2rem' }}>
            Cross-Silo Federated Learning Engine for Medical Image Segmentation
          </p>
        </div>
        <div className="status-badge">
          <div className="status-dot"></div>
          <span>{connectionStatus}</span>
        </div>
      </header>

      <div className="grid">
        <div className="card">
          <div className="card-title">Completed Round</div>
          <div className="card-value">{latestRound}</div>
        </div>
        <div className="card">
          <div className="card-title">Training Loss</div>
          <div className="card-value" style={{ color: 'var(--accent-cyan)' }}>{latestLoss}</div>
        </div>
        <div className="card">
          <div className="card-title">Dice Score</div>
          <div className="card-value" style={{ color: 'var(--accent-purple)' }}>{latestDice}</div>
        </div>
      </div>

      <div className="chart-card">
        <div className="chart-title">Real-time Federated Learning Metrics</div>
        <div style={{ width: '100%', height: 350 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={metrics}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="round_number" stroke="#94a3b8" label={{ value: 'FL Round', position: 'insideBottom', offset: -5 }} />
              <YAxis stroke="#94a3b8" />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', borderRadius: '0.5rem' }} />
              <Legend />
              <Line type="monotone" dataKey="training_loss" stroke="#38bdf8" name="Training Loss" strokeWidth={3} dot={{ r: 5 }} />
              <Line type="monotone" dataKey="dice_score" stroke="#c084fc" name="Dice Score" strokeWidth={3} dot={{ r: 5 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="table-card">
        <div className="chart-title">Hospital Client Nodes</div>
        <table>
          <thead>
            <tr>
              <th>Hospital ID</th>
              <th>Node Name</th>
              <th>Status</th>
              <th>Latency</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>hospital_a</td>
              <td>General Hospital Alpha</td>
              <td><span style={{ color: 'var(--accent-green)' }}>● Active / Training</span></td>
              <td>12 ms</td>
            </tr>
            <tr>
              <td>hospital_b</td>
              <td>St. Jude Medical Beta</td>
              <td><span style={{ color: 'var(--accent-green)' }}>● Active / Training</span></td>
              <td>18 ms</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
