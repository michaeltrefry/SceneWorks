export function sortNewest(a, b) {
  return b.createdAt.localeCompare(a.createdAt);
}

export function sortWorkers(a, b) {
  return `${a.gpuId}-${a.id}`.localeCompare(`${b.gpuId}-${b.id}`);
}
