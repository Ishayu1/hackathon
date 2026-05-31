export function createObjectUrl(blob) {
  return URL.createObjectURL(blob);
}

export function revokeObjectUrl(url) {
  if (url) {
    URL.revokeObjectURL(url);
  }
}

function floatTo16BitPCM(output, offset, input) {
  for (let i = 0; i < input.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, input[i]));
    output.setInt16(offset + i * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }
}

function writeWavHeader(view, { numChannels, sampleRate, dataLength }) {
  const blockAlign = numChannels * 2;
  const byteRate = sampleRate * blockAlign;

  const writeString = (offset, str) => {
    for (let i = 0; i < str.length; i += 1) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + dataLength, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, dataLength, true);
}

function encodeWav(audioBuffer) {
  const numChannels = audioBuffer.numberOfChannels;
  const sampleRate = audioBuffer.sampleRate;
  const length = audioBuffer.length;
  const interleaved = new Float32Array(length * numChannels);

  for (let channel = 0; channel < numChannels; channel += 1) {
    const channelData = audioBuffer.getChannelData(channel);
    for (let i = 0; i < length; i += 1) {
      interleaved[i * numChannels + channel] = channelData[i];
    }
  }

  const dataLength = interleaved.length * 2;
  const buffer = new ArrayBuffer(44 + dataLength);
  const view = new DataView(buffer);

  writeWavHeader(view, { numChannels, sampleRate, dataLength });
  floatTo16BitPCM(view, 44, interleaved);

  return buffer;
}

export async function blobToWav(blob) {
  const arrayBuffer = await blob.arrayBuffer();
  const audioContext = new AudioContext();

  try {
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
    const wavBuffer = encodeWav(audioBuffer);
    return new Blob([wavBuffer], { type: 'audio/wav' });
  } finally {
    await audioContext.close();
  }
}

export async function recordingBlobToFile(blob, mimeType) {
  const isWav = mimeType.includes('wav');
  const wavBlob = isWav ? blob : await blobToWav(blob);
  const filename = `recording-${Date.now()}.wav`;
  return new File([wavBlob], filename, { type: 'audio/wav' });
}
