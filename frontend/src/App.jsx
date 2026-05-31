import React, { useEffect, useRef, useState } from 'react';
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
  Square,
  Upload
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

const WAVEFORM_HEIGHT = 56;
const WAVEFORM_MIN_BARS = 120;
const WAVEFORM_MAX_BARS = 480;

function formatAudioTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${String(secs).padStart(2, '0')}`;
}

function computePeaks(channelData, barCount) {
  const samplesPerBar = Math.max(1, Math.floor(channelData.length / barCount));
  const peaks = new Float32Array(barCount);
  let maxPeak = 0.01;

  for (let index = 0; index < barCount; index += 1) {
    const start = index * samplesPerBar;
    const end = Math.min(channelData.length, start + samplesPerBar);
    let peak = 0;
    for (let sample = start; sample < end; sample += 1) {
      const amplitude = Math.abs(channelData[sample]);
      if (amplitude > peak) peak = amplitude;
    }
    peaks[index] = peak;
    if (peak > maxPeak) maxPeak = peak;
  }

  for (let index = 0; index < barCount; index += 1) {
    peaks[index] = Math.max(0.08, peaks[index] / maxPeak);
  }

  return peaks;
}

function drawWaveformCanvas(ctx, peaks, width, height) {
  const barCount = peaks.length;
  const gap = 1;
  const barWidth = Math.max(1, (width - gap * (barCount - 1)) / barCount);
  const centerY = height / 2;

  ctx.clearRect(0, 0, width, height);

  for (let index = 0; index < barCount; index += 1) {
    const barHeight = Math.max(4, peaks[index] * (height - 8));
    const x = index * (barWidth + gap);
    const y = centerY - barHeight / 2;
    const radius = Math.min(barWidth / 2, barHeight / 2);
    ctx.fillStyle = 'rgba(148, 163, 184, 0.72)';
    ctx.beginPath();
    if (typeof ctx.roundRect === 'function') {
      ctx.roundRect(x, y, barWidth, barHeight, radius);
    } else {
      ctx.rect(x, y, barWidth, barHeight);
    }
    ctx.fill();
  }
}

function AudioPlayer({ url, label = 'Play clip' }) {
  const audioRef = useRef(null);
  const trackRef = useRef(null);
  const canvasRef = useRef(null);
  const playheadRef = useRef(null);
  const peaksRef = useRef(null);
  const durationRef = useRef(0);
  const rafRef = useRef(null);
  const draggingRef = useRef(false);
  const lastLabelUpdateRef = useRef(0);

  const [playing, setPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [displayTime, setDisplayTime] = useState(0);
  const [waveformReady, setWaveformReady] = useState(false);

  const setPlayheadRatio = (ratio) => {
    const clamped = Math.min(1, Math.max(0, ratio));
    if (playheadRef.current) {
      playheadRef.current.style.left = `${clamped * 100}%`;
    }
  };

  const syncDisplayTime = (seconds, force = false) => {
    const now = performance.now();
    if (!force && now - lastLabelUpdateRef.current < 120) return;
    lastLabelUpdateRef.current = now;
    setDisplayTime(seconds);
  };

  const seekFromClientX = (clientX) => {
    const audio = audioRef.current;
    const track = trackRef.current;
    const trackDuration = durationRef.current;
    if (!audio || !track || !trackDuration) return;

    const rect = track.getBoundingClientRect();
    const clampedX = Math.min(rect.right, Math.max(rect.left, clientX));
    const ratio = (clampedX - rect.left) / rect.width;
    audio.currentTime = ratio * trackDuration;
    setPlayheadRatio(ratio);
    syncDisplayTime(audio.currentTime, true);
  };

  const redrawWaveform = () => {
    const canvas = canvasRef.current;
    const track = trackRef.current;
    const peaks = peaksRef.current;
    if (!canvas || !track || !peaks?.length) return;

    const width = Math.max(1, Math.floor(track.clientWidth));
    const barCount = Math.min(
      WAVEFORM_MAX_BARS,
      Math.max(WAVEFORM_MIN_BARS, Math.floor(width / 2))
    );

    let drawPeaks = peaks;
    if (peaks.length !== barCount) {
      const resampled = new Float32Array(barCount);
      const step = peaks.length / barCount;
      for (let index = 0; index < barCount; index += 1) {
        resampled[index] = peaks[Math.min(peaks.length - 1, Math.floor(index * step))];
      }
      drawPeaks = resampled;
    }

    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(WAVEFORM_HEIGHT * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${WAVEFORM_HEIGHT}px`;

    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      drawWaveformCanvas(ctx, drawPeaks, width, WAVEFORM_HEIGHT);
    }
  };

  useEffect(() => {
    setPlaying(false);
    setDuration(0);
    setDisplayTime(0);
    setWaveformReady(false);
    peaksRef.current = null;
    durationRef.current = 0;
    setPlayheadRatio(0);

    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }

    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, [url]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return undefined;

    const onEnded = () => {
      setPlaying(false);
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
    const onPause = () => {
      setPlaying(false);
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      syncDisplayTime(audio.currentTime, true);
    };
    const onPlay = () => setPlaying(true);
    const onLoadedMetadata = () => {
      const nextDuration = Number.isFinite(audio.duration) ? audio.duration : 0;
      durationRef.current = nextDuration;
      setDuration(nextDuration);
    };

    audio.addEventListener('ended', onEnded);
    audio.addEventListener('pause', onPause);
    audio.addEventListener('play', onPlay);
    audio.addEventListener('loadedmetadata', onLoadedMetadata);

    return () => {
      audio.removeEventListener('ended', onEnded);
      audio.removeEventListener('pause', onPause);
      audio.removeEventListener('play', onPlay);
      audio.removeEventListener('loadedmetadata', onLoadedMetadata);
    };
  }, [url]);

  useEffect(() => {
    if (!playing) return undefined;

    const tick = () => {
      const audio = audioRef.current;
      if (audio && !draggingRef.current && durationRef.current > 0) {
        const ratio = audio.currentTime / durationRef.current;
        setPlayheadRatio(ratio);
        syncDisplayTime(audio.currentTime);
      }
      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    };
  }, [playing]);

  useEffect(() => {
    if (!url) return undefined;

    let cancelled = false;
    const context = new (window.AudioContext || window.webkitAudioContext)();

    async function buildWaveform() {
      try {
        const response = await fetch(url);
        const audioBuffer = await response.arrayBuffer();
        const decoded = await context.decodeAudioData(audioBuffer);
        if (cancelled) return;

        peaksRef.current = computePeaks(decoded.getChannelData(0), WAVEFORM_MAX_BARS);
        setWaveformReady(true);
        requestAnimationFrame(() => redrawWaveform());
      } catch {
        peaksRef.current = null;
        setWaveformReady(false);
      }
    }

    buildWaveform();
    return () => {
      cancelled = true;
      context.close().catch(() => {});
    };
  }, [url]);

  useEffect(() => {
    const track = trackRef.current;
    if (!track || !waveformReady) return undefined;

    const observer = new ResizeObserver(() => {
      requestAnimationFrame(redrawWaveform);
    });
    observer.observe(track);
    return () => observer.disconnect();
  }, [url, waveformReady]);

  useEffect(() => {
    const onPointerMove = (event) => {
      if (!draggingRef.current) return;
      seekFromClientX(event.clientX);
    };

    const onPointerUp = () => {
      if (!draggingRef.current) return;
      draggingRef.current = false;
      document.body.classList.remove('waveform-dragging');
    };

    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
    window.addEventListener('pointercancel', onPointerUp);

    return () => {
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
      window.removeEventListener('pointercancel', onPointerUp);
      document.body.classList.remove('waveform-dragging');
    };
  }, []);

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

  const startDrag = (event) => {
    if (!durationRef.current) return;
    draggingRef.current = true;
    document.body.classList.add('waveform-dragging');
    event.currentTarget.setPointerCapture?.(event.pointerId);
    seekFromClientX(event.clientX);
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
          {formatAudioTime(displayTime)} / {formatAudioTime(duration)}
        </span>
      </div>
      <div
        ref={trackRef}
        className="waveform-track"
        role="slider"
        aria-label="Audio timeline"
        aria-valuemin={0}
        aria-valuemax={Math.round(duration)}
        aria-valuenow={Math.round(displayTime)}
        tabIndex={0}
        onPointerDown={startDrag}
        onKeyDown={(event) => {
          const audio = audioRef.current;
          const trackDuration = durationRef.current;
          if (!audio || !trackDuration) return;
          if (event.key === 'ArrowRight') {
            audio.currentTime = Math.min(trackDuration, audio.currentTime + 1);
            setPlayheadRatio(audio.currentTime / trackDuration);
            syncDisplayTime(audio.currentTime, true);
          } else if (event.key === 'ArrowLeft') {
            audio.currentTime = Math.max(0, audio.currentTime - 1);
            setPlayheadRatio(audio.currentTime / trackDuration);
            syncDisplayTime(audio.currentTime, true);
          }
        }}
      >
        <canvas ref={canvasRef} className="waveform-canvas" aria-hidden="true" />
        <div ref={playheadRef} className="waveform-playhead" aria-hidden="true">
          <div className="waveform-playhead-line" />
          <div className="waveform-playhead-knob" />
        </div>
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

  const duressScore = active?.duressAvailable
    ? active.duressPercent ?? Math.round((active.duressScore ?? 0) * 100)
    : null;
  const authenticityScore = active
    ? active.authenticity === 'Likely Real'
      ? Math.round((active.confidence ?? 1 - (active.spoofScore ?? 0)) * 100)
      : Math.round((active.confidence ?? active.spoofScore ?? 0) * 100)
    : 0;

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
                    <div className="analysis-head-top">
                      <div>
                        <div className="analysis-eyebrow">
                          <AudioWaveform size={18} /> Analysis Results
                        </div>
                        <h2>{active.title}</h2>
                      </div>
                      <RiskPill risk={active.risk} />
                    </div>
                    {active.audioUrl && (
                      <AudioPlayer
                        url={active.audioUrl}
                        label={active.source === 'record' ? 'Play recording' : 'Play clip'}
                      />
                    )}
                  </div>

                  <div className="analysis-details">
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
