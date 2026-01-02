const DEFAULT_BACKEND = (() => {
  try {
    const proto = window.location.protocol || 'http:'
    const host = window.location.hostname || '127.0.0.1'
    return `${proto}//${host}:8000`
  } catch {
    return 'http://127.0.0.1:8000'
  }
})()

export const BACKEND_URL: string = (import.meta as any).env?.VITE_BACKEND_URL || DEFAULT_BACKEND

function withBase(path: string): string {
  if (path.startsWith('http://') || path.startsWith('https://')) return path
  if (!path.startsWith('/')) return `${BACKEND_URL}/${path}`
  return `${BACKEND_URL}${path}`
}

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(withBase(path), { method: 'GET' })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return (await res.json()) as T
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(withBase(path), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(await safeErr(res))
  return (await res.json()) as T
}

export async function apiPut<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(withBase(path), {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(await safeErr(res))
  return (await res.json()) as T
}

export async function apiDelete<T>(path: string): Promise<T> {
  const res = await fetch(withBase(path), { method: 'DELETE' })
  if (!res.ok) throw new Error(await safeErr(res))
  return (await res.json()) as T
}

async function safeErr(res: Response): Promise<string> {
  try {
    const j = await res.json()
    if (j && typeof j === 'object' && 'detail' in j) return String((j as any).detail)
  } catch {
    // ignore
  }
  return `HTTP ${res.status}`
}
