import React, { useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import {
  AlertTriangle,
  AudioWaveform,
  BadgeCheck,
  BrainCircuit,
  Clock,
  FileAudio,
  Gauge,
  Lock,
  Radio,
  Radar,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Upload,
  Zap
} from 'lucide-react';

const intercepts = [
  {
    id: 'routine',
    title: 'Routine Base Update',
    subtitle: 'Supply check-in',
    authenticity: 'Likely Real',
    confidence: 0.18,
    latency: 112,
    risk: 'LOW',
    intent: 'Routine logistics update',
    transcript:
      'Routine supply check completed at Base Bravo. No additional support required.',
    recommendation: 'Log message. No escalation required.',
    chunks: [0.11, 0.18, 0.15, 0.09, 0.13],
    watch: ['Base Bravo']
  },
  {
    id: 'convoy',
    title: 'Convoy Reroute Order',
    subtitle: 'Movement command',
    authenticity: 'Likely Synthetic',
    confidence: 0.91,
    latency: 184,
    risk: 'CRITICAL',
    intent: 'Movement order / logistics reroute',
    transcript:
      'Convoy Alpha should reroute to Checkpoint Delta immediately. Repeat, reroute to Checkpoint Delta.',
    recommendation:
      'Do not execute automatically. Verify through secondary channel and escalate to communications officer.',
    chunks: [0.21, 0.44, 0.78, 0.91, 0.86],
    watch: ['Convoy Alpha', 'Checkpoint Delta', 'reroute']
  },
  {
    id: 'warning',
    title: 'Drone Activity Warning',
    subtitle: 'Threat alert',
    authenticity: 'Uncertain',
    confidence: 0.62,
    latency: 156,
    risk: 'HIGH',
    intent: 'Threat warning',
    transcript:
      'Drone activity detected near the north gate. All units should prepare for immediate response.',
    recommendation:
      'Escalate for confirmation. Cross-check with sensor feeds before operational action.',
    chunks: [0.32, 0.49, 0.62, 0.58, 0.46],
    watch: ['Drone activity', 'north gate', 'immediate response']
  }
];

const missionProfiles = [
  'Convoy Logistics',
  'Base Security',
  'Cyber Ops',
  'Medical Evac',
  'Supply Chain'
];

const operatorActions = ['Trust', 'Verify', 'Escalate', 'Block'];

function Card({ children, className = '' }) {
  return <div className={`card ${className}`}>{children}</div>;
}

function Button({ children, className = '', variant = 'solid', ...props }) {
  return (
    <button className={`btn ${variant === 'outline' ? 'btn-outline' : 'btn-solid'} ${className}`} {...props}>
      {children}
    </button>
  );
}

function RiskPill({ risk }) {
  return <span className={`risk-pill risk-${risk.toLowerCase()}`}>{risk}</span>;
}

function Waveform({ chunks }) {
  const bars = useMemo(
    () =>
      Array.from({ length: 56 }, (_, i) => {
        const chunkIndex = Math.min(chunks.length - 1, Math.floor((i / 56) * chunks.length));
        const base = chunks[chunkIndex];
        return 18 + Math.round(Math.abs(Math.sin(i * 0.72) + Math.cos(i * 0.31)) * 18 + base * 38);
      }),
    [chunks]
  );

  const maxChunk = chunks.indexOf(Math.max(...chunks));
  return (
    <div className="wave-wrap">
      <div className="wave-topline" />
      <div className="wave-meta">
        <span>Audio signal scan</span>
        <span>4s windows · 1s overlap</span>
      </div>
      <div className="wave-bars">
        {bars.map((h, i) => {
          const chunkIndex = Math.min(chunks.length - 1, Math.floor((i / 56) * chunks.length));
          const hot = chunkIndex === maxChunk;
          return (
            <motion.div
              key={i}
              initial={{ height: 4, opacity: 0.5 }}
              animate={{ height: h, opacity: 1 }}
              transition={{ delay: i * 0.008, duration: 0.35 }}
              className={`wave-bar ${hot ? 'wave-hot' : ''}`}
            />
          );
        })}
      </div>
      <div className="segment-grid">
        {chunks.map((score, idx) => (
          <div key={idx} className={`segment ${idx === maxChunk ? 'segment-hot' : ''}`}>
            <div className="segment-time">{idx * 3}-{idx * 3 + 4}s</div>
            <div className="segment-score">{Math.round(score * 100)}%</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function MetricCard({ icon: Icon, label, value, sub }) {
  return (
    <Card>
      <div className="metric-head">
        <div className="metric-icon">
          <Icon size={18} />
        </div>
        <div className="metric-dot" />
      </div>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {sub && <div className="metric-sub">{sub}</div>}
    </Card>
  );
}

export default function App() {
  const [active, setActive] = useState(intercepts[1]);
  const [mission, setMission] = useState(missionProfiles[0]);
  const [watchlist, setWatchlist] = useState('Convoy Alpha, Checkpoint Delta, Fuel Depot, Sector 7');
  const [operatorDecision, setOperatorDecision] = useState('Escalate');

  const fakeScore = Math.round(active.confidence * 100);
  const authenticityScore =
    active.authenticity === 'Likely Real' ? Math.round((1 - active.confidence) * 100) : fakeScore;

  const systemRecommendation =
    active.risk === 'LOW' ? 'TRUST' : active.risk === 'HIGH' ? 'VERIFY' : 'ESCALATE';

  return (
    <div className="app-root">
      <div className="bg-layer">
        <div className="blob blob1" />
        <div className="blob blob2" />
        <div className="blob blob3" />
        <div className="grid-overlay" />
      </div>

      <main className="container">
        <header className="hero card">
          <div>
            <div className="eyebrow">
              <Radar size={18} />
              <span>Mission Audio Triage</span>
            </div>
            <h1>
              SignalShield <span className="brand-grad">AI</span>
            </h1>
            <p>
              Detect synthetic mission audio, classify operational intent, and turn suspicious
              commands into clear verification decisions.
            </p>
          </div>
          <div className="hero-actions">
            <Button>
              <Upload size={16} /> Upload Audio
            </Button>
            <Button variant="outline">
              <Sparkles size={16} /> Generate Brief
            </Button>
          </div>
        </header>

        <div className="layout">
          <aside className="sidebar-col">
            <Card>
              <div className="panel-title">
                <Lock size={18} />
                <h2>Mission Setup</h2>
              </div>
              <label className="small-label">Profile</label>
              <div className="stack">
                {missionProfiles.map((profile) => (
                  <button
                    key={profile}
                    onClick={() => setMission(profile)}
                    className={`choice-btn ${mission === profile ? 'active-cyan' : ''}`}
                  >
                    {profile}
                  </button>
                ))}
              </div>
              <label className="small-label top-gap">Watched Assets</label>
              <textarea
                value={watchlist}
                onChange={(e) => setWatchlist(e.target.value)}
                rows={3}
                className="watch-input"
              />
            </Card>

            <Card>
              <div className="panel-title">
                <Radio size={18} />
                <h2>Sample Intercepts</h2>
              </div>
              <div className="stack">
                {intercepts.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => setActive(item)}
                    className={`choice-btn intercept ${active.id === item.id ? 'active-fuchsia' : ''}`}
                  >
                    <div>
                      <div className="intercept-title">{item.title}</div>
                      <div className="intercept-sub">{item.subtitle}</div>
                    </div>
                    <FileAudio size={18} />
                  </button>
                ))}
              </div>
            </Card>
          </aside>

          <section className="main-col">
            <div className="metric-grid">
              <MetricCard icon={ShieldAlert} label="Risk" value={active.risk} sub={`Mission: ${mission}`} />
              <MetricCard
                icon={Gauge}
                label="Authenticity"
                value={active.authenticity}
                sub={`${authenticityScore}% confidence`}
              />
              <MetricCard
                icon={BrainCircuit}
                label="Intent"
                value={active.intent.split(' /')[0]}
                sub="Transcript classifier"
              />
              <MetricCard icon={Clock} label="Latency" value={`${active.latency} ms`} sub="Warm inference" />
            </div>

            <Card className="analysis-card">
              <div className="analysis-head">
                <div>
                  <div className="analysis-eyebrow">
                    <AudioWaveform size={18} /> Active Intercept Analysis
                  </div>
                  <h2>{active.title}</h2>
                </div>
                <RiskPill risk={active.risk} />
              </div>

              <div className="analysis-grid">
                <div className="left-stack">
                  <Waveform chunks={active.chunks} />

                  <div className="two-col">
                    <div className="mini-card">
                      <div className="mini-title">
                        <Zap size={16} /> Authenticity Output
                      </div>
                      <div className="score-row">
                        <div className="score-value">{fakeScore}%</div>
                        <div className="score-meta">synthetic score</div>
                      </div>
                      <div className="progress-bg">
                        <motion.div
                          key={active.id}
                          initial={{ width: 0 }}
                          animate={{ width: `${fakeScore}%` }}
                          className="progress-fill"
                        />
                      </div>
                    </div>

                    <div className="mini-card">
                      <div className="mini-title">
                        <BadgeCheck size={16} /> Decision Band
                      </div>
                      <div className="decision-text">{systemRecommendation}</div>
                      <p className="small-note">
                        Generated from authenticity, intent, watchlist hits, and mission context.
                      </p>
                    </div>
                  </div>
                </div>

                <div className="right-stack">
                  <div className="mini-card">
                    <div className="label">Transcript</div>
                    <p className="quote">“{active.transcript}”</p>
                  </div>

                  <div className="mini-card">
                    <div className="label">Watchlist Matches</div>
                    <div className="chips">
                      {active.watch.map((term) => (
                        <span key={term} className="chip">
                          {term}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div className="mini-card alert">
                    <div className="mini-title">
                      {active.risk === 'LOW' ? <ShieldCheck size={18} /> : <AlertTriangle size={18} />}
                      Recommended Action
                    </div>
                    <p className="small-note">{active.recommendation}</p>
                  </div>

                  <div className="mini-card">
                    <div className="label">Decision Mode</div>
                    <div className="action-grid">
                      {operatorActions.map((action) => (
                        <button
                          key={action}
                          onClick={() => setOperatorDecision(action)}
                          className={`action-btn ${operatorDecision === action ? 'active-cyan' : ''}`}
                        >
                          {action}
                        </button>
                      ))}
                    </div>
                    <p className="small-note">
                      System recommendation: <strong>{systemRecommendation}</strong>
                    </p>
                  </div>

                  <div className="mini-card">
                    <div className="label">Analyst Summary</div>
                    <p className="small-note">
                      In a <strong>{mission}</strong> context, this intercept is classified as{' '}
                      <strong>{active.intent}</strong>. The system recommends{' '}
                      <strong>
                        {active.risk === 'LOW'
                          ? 'logging the message'
                          : active.risk === 'HIGH'
                            ? 'verification before action'
                            : 'immediate escalation'}
                      </strong>{' '}
                      based on authenticity confidence and operational impact.
                    </p>
                    <p className="small-note">Operator selected: {operatorDecision.toUpperCase()}</p>
                  </div>
                </div>
              </div>
            </Card>
          </section>
        </div>
      </main>
    </div>
  );
}
