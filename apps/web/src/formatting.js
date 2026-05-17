export function formatSeconds(seconds) {
  if (seconds === null || seconds === undefined) {
    return "0s";
  }
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return minutes > 0 ? `${minutes}m ${remainder}s` : `${remainder}s`;
}

export function percent(value) {
  return `${Math.round((value ?? 0) * 100)}%`;
}
