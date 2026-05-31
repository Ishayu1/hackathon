import React, { useEffect, useMemo, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import {
  AlertTriangle,
  AudioWaveform,
  BadgeCheck,
  BrainCircuit,
  Clock,
  FileAudio,
  Gauge,
  Loader2,
  Lock,
  Mic,
  Pause,
  Play,
  Radio,
  Radar,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Square,
  Upload,
  Zap
} from 'lucide-react';
import {
  applyTranscriptionResult,
  classifyAudio,
  getHealth,
  mapClassifyResponse,
  transcribeAudio
} from './api';
import { createObjectUrl, recordingBlobToFile, revokeObjectUrl } from './audioUtils';

const intercepts = [
  {
    id: 'routine',
    title: 'Routine Base Update',
    subtitle: 'Supply check-in',
    authenticity: 'Likely Real',
    spoofScore: 0.18,
    confidence: 0.82,
    latency: 112,
    source: 'sample',
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
    spoofScore: 0.91,
    confidence: 0.91,
    latency: 184,
    source: 'sample',
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
    spoofScore: 0.62,
    confidence: 0.62,
    latency: 156,
    source: 'sample',
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

function Waveform({ chunks, illustrative = false }) {
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
        <span>{illustrative ? 'Sample scenario (illustrative)' : 'Audio signal scan'}</span>
        <span>{illustrative ? 'not model output' : '4s windows · 1s overlap'}</span>
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

function AudioPlayer({ url, label = 'Play clip' }) {
  const audioRef = useRef(null);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    setPlaying(false);
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
  }, [url]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return undefined;

    const onEnded = () => setPlaying(false);
    const onPause = () => setPlaying(false);
    const onPlay = () => setPlaying(true);

    audio.addEventListener('ended', onEnded);
    audio.addEventListener('pause', onPause);
    audio.addEventListener('play', onPlay);

    return () => {
      audio.removeEventListener('ended', onEnded);
      audio.removeEventListener('pause', onPause);
      audio.removeEventListener('play', onPlay);
    };
  }, [url]);

  if (!url) return null;

  const togglePlayback = () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (playing) {
      audio.pause();
    } else {
      audio.play().catch(() => setPlaying(false));
    }
  };

  return (
    <div className="audio-player">
      <audio ref={audioRef} src={url} preload="metadata" />
      <Button type="button" variant="outline" className="audio-play-btn" onClick={togglePlayback}>
        {playing ? <Pause size={16} /> : <Play size={16} />}
        {playing ? 'Pause' : label}
      </Button>
    </div>
  );
}

function AcousticRationale({ explanation }) {
  if (!explanation) {
    return (
      <div className="mini-card">
        <div className="mini-title">
          <BrainCircuit size={16} /> Acoustic Rationale
        </div>
        <p className="small-note">Interpretability available for fast MFCC model only.</p>
      </div>
    );
  }

  if (!explanation.top_signals?.length) {
    return (
      <div className="mini-card">
        <div className="mini-title">
          <BrainCircuit size={16} /> Acoustic Rationale
        </div>
        <p className="small-note">{explanation.note || 'No grounded feature signals were returned.'}</p>
      </div>
    );
  }

  const summary =
    explanation.summary ||
    'The acoustic profile is consistent with the predicted class in the training corpus.';

  return (
    <div className="mini-card rationale-card">
      <div className="mini-title">
        <BrainCircuit size={16} /> Acoustic Rationale
      </div>
      <p className="rationale-summary">{summary}</p>
      <p className="small-note disclaimer">{explanation.disclaimer}</p>
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
  const fileInputRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const recordingStreamRef = useRef(null);
  const recordingChunksRef = useRef([]);
  const audioUrlRef = useRef(null);
  const [active, setActive] = useState(intercepts[1]);
  const [uploadResult, setUploadResult] = useState(null);
  const [uploadError, setUploadError] = useState('');
  const [mission, setMission] = useState(missionProfiles[0]);
  const [watchlist, setWatchlist] = useState('Convoy Alpha, Checkpoint Delta, Fuel Depot, Sector 7');
  const [operatorDecision, setOperatorDecision] = useState('Escalate');
  const [backendHealth, setBackendHealth] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);

  useEffect(() => {
    return () => {
      revokeObjectUrl(audioUrlRef.current);
    };
  }, []);

  useEffect(() => {
    if (!isRecording) {
      setRecordingSeconds(0);
      return undefined;
    }

    const timer = setInterval(() => {
      setRecordingSeconds((seconds) => seconds + 1);
    }, 1000);

    return () => clearInterval(timer);
  }, [isRecording]);

  useEffect(() => {
    let cancelled = false;

    async function checkHealth() {
      try {
        const health = await getHealth();
        if (!cancelled) {
          setBackendHealth(health);
        }
      } catch {
        if (!cancelled) {
          setBackendHealth({ status: 'offline', model_loaded: false });
        }
      }
    }

    checkHealth();
    const timer = setInterval(checkHealth, 15000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  const handleUploadClick = () => {
    setUploadError('');
    fileInputRef.current?.click();
  };

  const stopRecordingTracks = () => {
    recordingStreamRef.current?.getTracks().forEach((track) => track.stop());
    recordingStreamRef.current = null;
  };

  const processAudioFile = async (file, sourceType = 'upload') => {
    setIsUploading(true);
    setUploadError('');

    revokeObjectUrl(audioUrlRef.current);
    const audioUrl = createObjectUrl(file);
    audioUrlRef.current = audioUrl;

    try {
      const result = await classifyAudio(file);
      const mapped = mapClassifyResponse(result, { sourceType, audioUrl });
      setUploadResult(mapped);
      setActive(mapped);
      setOperatorDecision(mapped.systemRecommendation);

      transcribeAudio(file, {
        customKeywords: watchlist,
        deepfakeProbability: mapped.spoofScore
      })
        .then((transcription) => {
          const updated = applyTranscriptionResult(mapped, transcription);
          setUploadResult((current) => (current?.id === mapped.id ? updated : current));
          setActive((current) => (current?.id === mapped.id ? updated : current));
          setOperatorDecision(updated.systemRecommendation);
        })
        .catch((error) => {
          const updated = applyTranscriptionResult(mapped, {
            available: false,
            error: error.message || 'Transcription failed'
          });
          setUploadResult((current) => (current?.id === mapped.id ? updated : current));
          setActive((current) => (current?.id === mapped.id ? updated : current));
        });
    } catch (error) {
      revokeObjectUrl(audioUrl);
      audioUrlRef.current = null;
      setUploadError(error.message || 'Analysis failed');
    } finally {
      setIsUploading(false);
    }
  };

  const handleFileSelected = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;

    await processAudioFile(file, 'upload');
  };

  const handleStartRecording = async () => {
    if (isUploading || isRecording) return;

    setUploadError('');

    if (!navigator.mediaDevices?.getUserMedia) {
      setUploadError('Microphone recording is not supported in this browser.');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recordingStreamRef.current = stream;
      recordingChunksRef.current = [];

      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : MediaRecorder.isTypeSupported('audio/webm')
          ? 'audio/webm'
          : '';

      const recorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);

      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordingChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        stopRecordingTracks();

        const blob = new Blob(recordingChunksRef.current, {
          type: recorder.mimeType || 'audio/webm'
        });
        recordingChunksRef.current = [];

        if (blob.size === 0) {
          setUploadError('Recording was empty. Try again and speak into the microphone.');
          return;
        }

        try {
          const file = await recordingBlobToFile(blob, recorder.mimeType || 'audio/webm');
          await processAudioFile(file, 'record');
        } catch (error) {
          setUploadError(error.message || 'Could not process recording');
        }
      };

      recorder.start();
      setIsRecording(true);
    } catch (error) {
      stopRecordingTracks();
      setUploadError(error.message || 'Microphone access was denied');
    }
  };

  const handleStopRecording = () => {
    if (!isRecording) return;

    mediaRecorderRef.current?.stop();
    mediaRecorderRef.current = null;
    setIsRecording(false);
  };

  const handleRecordClick = () => {
    if (isRecording) {
      handleStopRecording();
    } else {
      handleStartRecording();
    }
  };

  const formatRecordingTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const hasPlayableAudio = Boolean(active.audioUrl);
  const isBusy = isUploading || isRecording;

  const fakeScore = Math.round((active.spoofScore ?? active.syntheticScore ?? active.confidence) * 100);
  const authenticityScore =
    active.authenticity === 'Likely Real'
      ? Math.round((active.confidence ?? 1 - (active.spoofScore ?? 0)) * 100)
      : Math.round((active.confidence ?? active.spoofScore ?? 0) * 100);

  const systemRecommendation =
    active.systemRecommendation ||
    (active.risk === 'LOW' ? 'TRUST' : active.risk === 'HIGH' ? 'VERIFY' : 'ESCALATE');

  const backendLabel = backendHealth?.model_loaded
    ? `${backendHealth.backend} · ${backendHealth.device}`
    : backendHealth?.status === 'offline'
      ? 'API offline'
      : 'Connecting…';


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
              Detect synthetic mission audio, transcribe operational content, and combine both
              signals into clear verification decisions.
            </p>
            <div className={`backend-pill ${backendHealth?.model_loaded ? 'online' : 'offline'}`}>
              <span className="backend-dot" />
              {backendLabel}
            </div>
          </div>
          <div className="hero-actions">
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*,.flac,.wav,.mp3,.m4a,.ogg"
              hidden
              onChange={handleFileSelected}
            />
            <Button onClick={handleUploadClick} disabled={isBusy}>
              {isUploading ? <Loader2 size={16} className="spin" /> : <Upload size={16} />}
              {isUploading ? 'Analyzing…' : 'Upload Audio'}
            </Button>
            <Button
              variant="outline"
              className={isRecording ? 'btn-recording' : ''}
              onClick={handleRecordClick}
              disabled={isUploading}
            >
              {isRecording ? <Square size={16} /> : <Mic size={16} />}
              {isRecording ? `Stop · ${formatRecordingTime(recordingSeconds)}` : 'Record Live'}
            </Button>
            <Button variant="outline" onClick={() => setActive(intercepts[1])} disabled={isBusy}>
              <Sparkles size={16} /> Load Sample
            </Button>
          </div>
        </header>

        {uploadError && (
          <div className="error-banner card">
            <AlertTriangle size={18} />
            <span>{uploadError}</span>
          </div>
        )}

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
                {uploadResult && (
                  <button
                    onClick={() => setActive(uploadResult)}
                    className={`choice-btn intercept ${active.id === uploadResult.id ? 'active-fuchsia' : ''}`}
                  >
                    <div>
                      <div className="intercept-title">
                        {uploadResult.source === 'record' ? 'Latest Recording' : 'Latest Upload'}
                      </div>
                      <div className="intercept-sub">{uploadResult.title}</div>
                    </div>
                    <FileAudio size={18} />
                  </button>
                )}
                {intercepts.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => {
                      setActive(item);
                      setUploadError('');
                    }}
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
              {uploadError && <p className="small-note upload-error">{uploadError}</p>}
            </Card>
          </aside>

          <section className="main-col">
            <div className="metric-grid">
              <MetricCard icon={ShieldAlert} label="Overall Risk" value={active.risk} sub="Authenticity + severity" />
              <MetricCard
                icon={Gauge}
                label="Authenticity"
                value={active.authenticity}
                sub={`${authenticityScore}% confidence`}
              />
              <MetricCard
                icon={BrainCircuit}
                label="Category"
                value={active.intent.split(' /')[0]}
                sub="From transcript"
              />
              <MetricCard icon={Clock} label="Latency" value={`${active.latency} ms`} sub="Live inference" />
            </div>

            <Card className="analysis-card">
              <div className="analysis-head">
                <div>
                  <div className="analysis-eyebrow">
                    <AudioWaveform size={18} /> Audio + Transcript Analysis
                  </div>
                  <h2>{active.title}</h2>
                  {hasPlayableAudio && (
                    <AudioPlayer
                      url={active.audioUrl}
                      label={active.source === 'record' ? 'Play recording' : 'Play clip'}
                    />
                  )}
                </div>
                <RiskPill risk={active.risk} />
              </div>

              <div className="analysis-grid">
                <div className="left-stack">
                  {active.source === 'upload' || active.source === 'record' ? (
                    <AcousticRationale explanation={active.explanation} />
                  ) : (
                    <Waveform chunks={active.chunks} illustrative />
                  )}

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
                    {active.isTranscribing && <p className="small-note">Authenticity is ready. Transcription is still running.</p>}
                  </div>

                  <div className="mini-card">
                    <div className="label">Watchlist Matches</div>
                    <div className="chips">
                      {active.watch.length === 0 && <span className="chip muted-chip">No extracted matches</span>}
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
                    <div className="label">Analysis Summary</div>
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
                      based on authenticity confidence, transcript severity, and operational impact.
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
