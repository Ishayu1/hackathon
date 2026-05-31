function escapeRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

export function parseWatchlistKeywords(watchlist) {
  if (!watchlist?.trim()) return [];
  return Array.from(
    new Set(
      watchlist
        .split(/[,;\n]/)
        .map((item) => item.trim())
        .filter(Boolean)
    )
  );
}

function keywordToPattern(keyword) {
  const parts = keyword.trim().split(/\s+/).filter(Boolean).map(escapeRegex);
  if (!parts.length) return null;
  return `\\b${parts.join('\\s+')}\\b`;
}

export function highlightTranscript(text, watchlist) {
  if (!text) return [{ text: '', highlight: false }];

  const patterns = parseWatchlistKeywords(watchlist)
    .sort((a, b) => b.length - a.length)
    .map(keywordToPattern)
    .filter(Boolean);

  if (!patterns.length) return [{ text, highlight: false }];

  const regex = new RegExp(patterns.join('|'), 'gi');
  const parts = [];
  let lastIndex = 0;

  for (const match of text.matchAll(regex)) {
    const start = match.index ?? 0;
    if (start > lastIndex) {
      parts.push({ text: text.slice(lastIndex, start), highlight: false });
    }
    parts.push({ text: match[0], highlight: true });
    lastIndex = start + match[0].length;
  }

  if (lastIndex < text.length) {
    parts.push({ text: text.slice(lastIndex), highlight: false });
  }

  return parts.length ? parts : [{ text, highlight: false }];
}
