import React, { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import {
  AlertTriangle,
  AudioWaveform,
  BrainCircuit,
  Clock,
  Gauge,
  HeartPulse,
  Loader2,
  Mic,
  Pause,
  Play,
  Radar,
  ShieldAlert,
  ShieldCheck,
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
import { highlightTranscript, parseWatchlistKeywords } from './textUtils';

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

function AudioPlayer({ url, label = 'Play clip' }) {
  const audioRef = useRef(null);
  const progressRef = useRef(null);
  const [playing, setPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [waveformBars, setWaveformBars] = useState([]);

  useEffect(() => {
    setPlaying(false);
    setDuration(0);
    setCurrentTime(0);
    setWaveformBars([]);
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
    const onLoadedMetadata = () => setDuration(Number.isFinite(audio.duration) ? audio.duration : 0);
    const onTimeUpdate = () => setCurrentTime(audio.currentTime || 0);

    audio.addEventListener('ended', onEnded);
    audio.addEventListener('pause', onPause);
    audio.addEventListener('play', onPlay);
    audio.addEventListener('loadedmetadata', onLoadedMetadata);
    audio.addEventListener('timeupdate', onTimeUpdate);

    return () => {
      audio.removeEventListener('ended', onEnded);
      audio.removeEventListener('pause', onPause);
      audio.removeEventListener('play', onPlay);
      audio.removeEventListener('loadedmetadata', onLoadedMetadata);
      audio.removeEventListener('timeupdate', onTimeUpdate);
    };
  }, [url]);

  useEffect(() => {
    if (!url) {
      return undefined;
    }

    let cancelled = false;
    const context = new (window.AudioContext || window.webkitAudioContext)();

    async function buildWaveform() {
      try {
        const response = await fetch(url);
        const audioBuffer = await response.arrayBuffer();
        const decoded = await context.decodeAudioData(audioBuffer);
        if (cancelled) return;

        const data = decoded.getChannelData(0);
        const bars = 72;
        const samplesPerBar = Math.max(1, Math.floor(data.length / bars));
        const nextBars = [];

        for (let index = 0; index < bars; index += 1) {
          const start = index * samplesPerBar;
          const end = Math.min(data.length, start + samplesPerBar);
          let peak = 0;
          for (let sample = start; sample < end; sample += 1) {
            const amplitude = Math.abs(data[sample]);
            if (amplitude > peak) peak = amplitude;
          }
          nextBars.push(peak);
        }

        const maxPeak = Math.max(...nextBars, 0.01);
        const normalized = nextBars.map((value) => Math.max(0.08, value / maxPeak));
        setWaveformBars(normalized);
      } catch {
        setWaveformBars([]);
      }
    }

    buildWaveform();
    return () => {
      cancelled = true;
      context.close().catch(() => {});
    };
  }, [url]);

  const formatTime = (seconds) => {
    if (!Number.isFinite(seconds) || seconds <= 0) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${String(secs).padStart(2, '0')}`;
  };

  const seekToPosition = (clientX) => {
    const audio = audioRef.current;
    const progress = progressRef.current;
    if (!audio || !progress || !duration) return;

    const rect = progress.getBoundingClientRect();
    const clampedX = Math.min(rect.right, Math.max(rect.left, clientX));
    const ratio = (clampedX - rect.left) / rect.width;
    audio.currentTime = ratio * duration;
    setCurrentTime(audio.currentTime);
  };

  const progressRatio = duration > 0 ? Math.min(1, currentTime / duration) : 0;

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
      <div className="audio-player-controls">
        <Button type="button" variant="outline" className="audio-play-btn" onClick={togglePlayback}>
          {playing ? <Pause size={16} /> : <Play size={16} />}
          {playing ? 'Pause' : label}
        </Button>
        <span className="audio-time">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
      </div>
      <div
        ref={progressRef}
        className="waveform-track"
        role="slider"
        aria-label="Audio timeline"
        aria-valuemin={0}
        aria-valuemax={Math.round(duration)}
        aria-valuenow={Math.round(currentTime)}
        tabIndex={0}
        onClick={(event) => seekToPosition(event.clientX)}
        onKeyDown={(event) => {
          const audio = audioRef.current;
          if (!audio || !duration) return;
          if (event.key === 'ArrowRight') {
            audio.currentTime = Math.min(duration, audio.currentTime + 2);
            setCurrentTime(audio.currentTime);
          } else if (event.key === 'ArrowLeft') {
            audio.currentTime = Math.max(0, audio.currentTime - 2);
            setCurrentTime(audio.currentTime);
          }
        }}
      >
        <div className="waveform-bars" aria-hidden="true">
          {(waveformBars.length ? waveformBars : new Array(72).fill(0.22)).map((bar, index) => {
            const active = index / 72 <= progressRatio;
            return (
              <span
                key={`${index}-${bar}`}
                className={`waveform-bar ${active ? 'active' : ''}`}
                style={{ height: `${Math.round(14 + bar * 38)}px` }}
              />
            );
          })}
        </div>
        <div className="waveform-progress" style={{ width: `${progressRatio * 100}%` }} />
      </div>
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

function HighlightedTranscript({ text, watchlist }) {
  const parts = highlightTranscript(text, watchlist);

  return (
    <p className="quote">
      “
      {parts.map((part, index) =>
        part.highlight ? (
          <span key={index} className="keyword-color">
            {part.text}
          </span>
        ) : (
          <span key={index}>{part.text}</span>
        )
      )}
      ”
    </p>
  );
}

function EmptyState() {
  return (
    <Card className="empty-state">
      <AudioWaveform size={28} />
      <h2>No audio analyzed yet</h2>
      <p>Upload a clip or record live audio to run deepfake detection, duress analysis, and transcription.</p>
    </Card>
  );
}

export default function App() {
  const fileInputRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const recordingStreamRef = useRef(null);
  const recordingChunksRef = useRef([]);
  const audioUrlRef = useRef(null);
  const [active, setActive] = useState(null);
  const [uploadError, setUploadError] = useState('');
  const [watchlist, setWatchlist] = useState('');
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
      setActive(mapped);

      transcribeAudio(file, {
        customKeywords: watchlist,
        deepfakeProbability: mapped.spoofScore,
        duressProbability: mapped.duressScore ?? 0
      })
        .then((transcription) => {
          try {
            const updated = applyTranscriptionResult(mapped, transcription);
            setActive((current) => (current?.id === mapped.id ? updated : current));
          } catch (error) {
            setActive((current) =>
              current?.id === mapped.id
                ? {
                    ...current,
                    isTranscribing: false,
                    transcript: `Transcription update failed: ${error.message || 'unknown error'}`
                  }
                : current
            );
          }
        })
        .catch((error) => {
          const updated = applyTranscriptionResult(mapped, {
            available: false,
            error: error.message || 'Transcription failed'
          });
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

  const isBusy = isUploading || isRecording;

  const fakeScore = active
    ? Math.round((active.spoofScore ?? active.syntheticScore ?? active.confidence) * 100)
    : 0;
  const duressScore = active?.duressAvailable
    ? active.duressPercent ?? Math.round((active.duressScore ?? 0) * 100)
    : null;
  const authenticityScore = active
    ? active.authenticity === 'Likely Real'
      ? Math.round((active.confidence ?? 1 - (active.spoofScore ?? 0)) * 100)
      : Math.round((active.confidence ?? active.spoofScore ?? 0) * 100)
    : 0;

  const systemRecommendation =
    active?.systemRecommendation ||
    (active?.risk === 'LOW' ? 'TRUST' : active?.risk === 'HIGH' ? 'VERIFY' : 'ESCALATE');

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
              Detect synthetic speech, acoustic duress, and operational content from uploaded or live audio.
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
                <ShieldAlert size={18} />
                <h2>Watchlist Keywords</h2>
              </div>
              <p className="small-note">
                Optional comma-separated terms to flag in the transcript during analysis.
              </p>
              <label className="small-label top-gap">Keywords</label>
              <textarea
                value={watchlist}
                onChange={(e) => setWatchlist(e.target.value)}
                rows={4}
                placeholder="e.g. convoy, checkpoint, medevac"
                className="watch-input"
              />
            </Card>
          </aside>

          <section className="main-col">
            {!active ? (
              <EmptyState />
            ) : (
              <>
                <div className="metric-grid metric-grid-5">
                  <MetricCard icon={ShieldAlert} label="Overall Risk" value={active.risk} sub="Authenticity + duress + severity" />
                  <MetricCard
                    icon={Gauge}
                    label="Authenticity"
                    value={active.authenticity}
                    sub={`${authenticityScore}% confidence`}
                  />
                  <MetricCard
                    icon={HeartPulse}
                    label="Acoustic Duress"
                    value={active.duressAvailable === false ? 'Unavailable' : active.duressLabel || 'Pending'}
                    sub={
                      active.duressAvailable === false
                        ? active.duressError || 'Model not loaded'
                        : `${duressScore ?? 0}% stress probability`
                    }
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
                        <AudioWaveform size={18} /> Analysis Results
                      </div>
                      <h2>{active.title}</h2>
                      {active.audioUrl && (
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
                      <AcousticRationale explanation={active.explanation} />

                      <div className="two-col">
                        <div className="mini-card">
                          <div className="mini-title">
                            <Zap size={16} /> Authenticity
                          </div>
                          <div className="score-row">
                            <div className="score-value">{fakeScore}%</div>
                            <div className="score-meta">synthetic score</div>
                          </div>
                          <div className="progress-bg">
                            <motion.div
                              key={`${active.id}-spoof`}
                              initial={{ width: 0 }}
                              animate={{ width: `${fakeScore}%` }}
                              className="progress-fill"
                            />
                          </div>
                        </div>

                        <div className={`mini-card ${active.isDuress ? 'duress-alert' : ''}`}>
                          <div className="mini-title">
                            <HeartPulse size={16} /> Acoustic Duress
                          </div>
                          {active.duressAvailable === false ? (
                            <p className="small-note">
                              {active.duressError || 'Duress model unavailable. Place temporal_bilstm_duress.pth in the project root.'}
                            </p>
                          ) : (
                            <>
                              <div className="score-row">
                                <div className="score-value">{duressScore ?? 0}%</div>
                                <div className="score-meta">{active.duressLabel || 'Pending analysis'}</div>
                              </div>
                              <div className="progress-bg">
                                <motion.div
                                  key={`${active.id}-duress`}
                                  initial={{ width: 0 }}
                                  animate={{ width: `${duressScore ?? 0}%` }}
                                  className={`progress-fill ${active.isDuress ? 'progress-duress' : 'progress-calm'}`}
                                />
                              </div>
                            </>
                          )}
                        </div>
                      </div>
                    </div>

                    <div className="right-stack">
                      <div className="mini-card">
                        <div className="label">Transcript</div>
                        {parseWatchlistKeywords(watchlist).length > 0 &&
                        !active.isTranscribing &&
                        !active.transcript.startsWith('Transcription') ? (
                          <HighlightedTranscript text={active.transcript} watchlist={watchlist} />
                        ) : (
                          <p className="quote">“{active.transcript}”</p>
                        )}
                        {active.isTranscribing && (
                          <p className="small-note">Authenticity is ready. Transcription is still running.</p>
                        )}
                      </div>

                      {watchlist.trim() && (
                        <div className="mini-card">
                          <div className="label">Watchlist Matches</div>
                          <div className="chips">
                            {active.watch.length === 0 && (
                              <span className="chip muted-chip">No matches in transcript</span>
                            )}
                            {active.watch.map((term) => (
                              <span key={term} className="chip">
                                {term}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}

                      <div className="mini-card alert">
                        <div className="mini-title">
                          {active.risk === 'LOW' ? <ShieldCheck size={18} /> : <AlertTriangle size={18} />}
                          Recommendation · {systemRecommendation}
                        </div>
                        <p className="small-note">{active.recommendation}</p>
                      </div>
                    </div>
                  </div>
                </Card>
              </>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}
