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

export async function classifyAudio(file, { signal, customKeywords = '' } = {}) {
  const form = new FormData();
  form.append('file', file);

  const response = await fetch(`${API_BASE}/classify`, {
    method: 'POST',
    body: form,
    signal
  });

  return parseJsonResponse(response);
}

export async function transcribeAudio(file, { signal, customKeywords = '', deepfakeProbability = 0 } = {}) {
  const form = new FormData();
  form.append('file', file);
  form.append('custom_keywords', customKeywords);
  form.append('deepfake_probability', String(deepfakeProbability));

  const response = await fetch(`${API_BASE}/transcribe`, {
    method: 'POST',
    body: form,
    signal
  });

  return parseJsonResponse(response);
}

export function mapClassifyResponse(result) {
  const spoofScore = Math.min(
    1,
    Math.max(0, result.score_spoof ?? (result.is_spoof ? result.confidence : 1 - result.confidence))
  );
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
      recommendation = 'Audio is consistent with authentic training examples. Log and continue routine monitoring.';
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
      recommendation = 'Audio is consistent with spoof training examples. Do not execute commands automatically; escalate immediately.';
      systemRecommendation = 'ESCALATE';
    }
  }

  const transcription = result.transcription;
  const transcriptionAvailable = Boolean(transcription?.available);
  const transcript = transcriptionAvailable
    ? transcription.transcript || 'No speech detected.'
    : `Transcription unavailable: ${transcription?.error || 'not returned by API'}`;
  const segmentCategory = transcriptionAvailable ? transcription.category : null;
  const segmentSeverity = transcriptionAvailable ? transcription.severity : null;
  const severityScore = transcriptionAvailable ? transcription.severity_score ?? 0 : 0;
  const severityRank = { low: 0, medium: 1, high: 2, critical: 3 };
  const severityRisk = {
    low: 'LOW',
    medium: 'HIGH',
    high: 'HIGH',
    critical: 'CRITICAL'
  }[segmentSeverity] || null;

  if (severityRisk && severityRank[segmentSeverity] > severityRank.low) {
    const riskRank = { LOW: 0, HIGH: 1, CRITICAL: 2 };
    if (riskRank[severityRisk] > riskRank[risk]) {
      risk = severityRisk;
    }
  }

  const terms = transcriptionAvailable
    ? Array.from(new Set([
        ...(transcription.chunks || []).flatMap((chunk) => chunk.matched_terms || []),
        ...(transcription.terms || [])
      ]))
    : [];
  const customMatches = terms.filter((term) => term.startsWith('custom: ')).map((term) => term.replace('custom: ', ''));
  const watch = customMatches.length
    ? customMatches
    : terms.length
      ? terms.slice(0, 6)
      : result.backend === 'fast'
        ? ['Fast MFCC model']
        : ['Spectra-AASIST3'];

  const intent = transcriptionAvailable
    ? `${segmentCategory} / ${segmentSeverity} (${severityScore})`
    : 'Pending transcript';

  if (transcriptionAvailable && segmentSeverity === 'critical') {
    recommendation = `${recommendation} Transcript triage is critical; escalate and verify content before action.`;
    systemRecommendation = 'ESCALATE';
  } else if (transcriptionAvailable && ['medium', 'high'].includes(segmentSeverity)) {
    recommendation = `${recommendation} Transcript triage found operational terms; verify before execution.`;
    if (systemRecommendation === 'TRUST') {
      systemRecommendation = 'VERIFY';
    }
  }

  return {
    id: `upload-${Date.now()}`,
    title: result.filename || 'Uploaded Audio',
    subtitle: `${result.backend} · ${result.label}`,
    authenticity,
    spoofScore,
    confidence: result.confidence,
    latency: Math.round(result.total_ms ?? 0),
    risk,
    intent,
    transcript: transcriptionAvailable ? transcript : 'Transcription running...',
    recommendation,
    systemRecommendation,
    source: 'upload',
    chunks: transcriptionAvailable && transcription.chunks?.length
      ? transcription.chunks.map((chunk) => Math.min(1, Math.max(0.05, (chunk.severity_score || 0) / 100)))
      : [],
    watch,
    explanation: result.explanation,
    transcription,
    transcriptionTerms: terms,
    isTranscribing: !transcriptionAvailable,
    raw: result
  };
}

export function applyTranscriptionResult(active, transcription) {
  const transcriptionAvailable = Boolean(transcription?.available);
  if (!transcriptionAvailable) {
    return {
      ...active,
      isTranscribing: false,
      transcript: `Transcription unavailable: ${transcription?.error || 'not returned by API'}`,
      transcription
    };
  }

  const segmentSeverity = transcription.severity;
  const severityScore = transcription.severity_score ?? 0;
  const terms = Array.from(new Set((transcription.chunks || []).flatMap((chunk) => chunk.matched_terms || [])));
  const customMatches = terms.filter((term) => term.startsWith('custom: ')).map((term) => term.replace('custom: ', ''));
  const riskRank = { LOW: 0, HIGH: 1, CRITICAL: 2 };
  const severityRisk = {
    low: 'LOW',
    medium: 'HIGH',
    high: 'HIGH',
    critical: 'CRITICAL'
  }[segmentSeverity] || 'LOW';
  const risk = riskRank[severityRisk] > riskRank[active.risk] ? severityRisk : active.risk;

  let recommendation = active.recommendation;
  let systemRecommendation = active.systemRecommendation;
  if (segmentSeverity === 'critical') {
    recommendation = `${recommendation} Transcript severity is critical; escalate and verify content before action.`;
    systemRecommendation = 'ESCALATE';
  } else if (['medium', 'high'].includes(segmentSeverity) && systemRecommendation === 'TRUST') {
    recommendation = `${recommendation} Transcript contains operational terms; verify before execution.`;
    systemRecommendation = 'VERIFY';
  }

  return {
    ...active,
    risk,
    transcript: transcription.transcript || 'No speech detected.',
    intent: `${transcription.category} / ${segmentSeverity} (${severityScore})`,
    recommendation,
    systemRecommendation,
    chunks: transcription.chunks?.length
      ? transcription.chunks.map((chunk) => Math.min(1, Math.max(0.05, (chunk.severity_score || 0) / 100)))
      : active.chunks,
    watch: customMatches.length ? customMatches : terms.length ? terms.slice(0, 6) : active.watch,
    transcription,
    transcriptionTerms: terms,
    isTranscribing: false
  };
}
