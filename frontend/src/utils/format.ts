export function fmtMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return '—'
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(2)}s`
}

export function fmtUsd(v: number | null | undefined): string {
  if (v === null || v === undefined) return '—'
  return `$${v.toFixed(6)}`
}

export function fmtIso(iso: string): string {
  try {
    const d = new Date(iso)
    return d.toLocaleString()
  } catch {
    return iso
  }
}

