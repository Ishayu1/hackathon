const API_BASE = import.meta.env.VITE_API_BASE || '/api';

async function parseJsonResponse(response) {
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = body.detail || body.message || response.statusText;
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return body;
}

export async function getHealth() {
  const response = await fetch(`${API_BASE}/health`);
  return parseJsonResponse(response);
}

export async function classifyAudio(file, { signal } = {}) {
  const form = new FormData();
  form.append('file', file);

  const response = await fetch(`${API_BASE}/classify`, {
    method: 'POST',
    body: form,
    signal
  });

  return parseJsonResponse(response);
}

export function mapClassifyResponse(result) {
  const spoofScore = Math.min(1, Math.max(0, result.score_spoof ?? (result.is_spoof ? result.confidence : 1 - result.confidence)));
  const isUncertain = result.confidence < 0.85;

  let authenticity;
  let risk;
  let recommendation;
  let systemRecommendation;

  if (result.label === 'bonafide') {
    authenticity = isUncertain ? 'Uncertain' : 'Likely Real';
    if (isUncertain) {
      risk = 'HIGH';
      recommendation = 'Authenticity confidence is moderate. Verify through a secondary channel before acting.';
      systemRecommendation = 'VERIFY';
    } else {
      risk = 'LOW';
      recommendation = 'Audio appears authentic. Log and continue routine monitoring.';
      systemRecommendation = 'TRUST';
    }
  } else {
    authenticity = isUncertain ? 'Uncertain' : 'Likely Synthetic';
    if (isUncertain) {
      risk = 'HIGH';
      recommendation = 'Possible synthetic speech. Escalate for human confirmation before action.';
      systemRecommendation = 'VERIFY';
    } else {
      risk = 'CRITICAL';
      recommendation = 'Synthetic speech detected. Do not execute commands automatically. Escalate immediately.';
      systemRecommendation = 'ESCALATE';
    }
  }

  const base = Math.max(0.05, spoofScore);
  const chunks = [0.85, 0.92, 1.04, 0.97, 0.88].map((m) => Math.min(0.99, base * m));

  return {
    id: `upload-${Date.now()}`,
    title: result.filename || 'Uploaded Audio',
    subtitle: `${result.backend} · ${result.label}`,
    authenticity,
    spoofScore,
    confidence: result.confidence,
    latency: Math.round(result.total_ms ?? 0),
    risk,
    intent: result.is_spoof ? 'Synthetic speech detected' : 'Human speech detected',
    transcript: `Screened with ${result.backend} backend. Spoof probability ${Math.round(spoofScore * 100)}%.`,
    recommendation,
    systemRecommendation,
    chunks,
    watch: result.backend === 'fast' ? ['MFCC RBF'] : ['Spectra-AASIST3'],
    raw: result
  };
}
