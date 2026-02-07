import { authHeader } from '../auth'

const DEFAULT_BACKEND = (() => {
  try {
    const proto = window.location.protocol || 'http:'
    const host = window.location.hostname || '127.0.0.1'
    const port = window.location.port || ''
    const origin = window.location.origin || ''
    const isHttp = origin.startsWith('http://') || origin.startsWith('https://')
    if (port === '5173' || port === '4173') {
      return `${proto}//${host}:8000`
    }
    if (isHttp) return origin
    return 'http://127.0.0.1:8000'
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
  const res = await fetch(withBase(path), { method: 'GET', headers: { ...authHeader() } })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return (await res.json()) as T
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(withBase(path), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeader() },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(await safeErr(res))
  return (await res.json()) as T
}

export async function apiPut<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(withBase(path), {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json', ...authHeader() },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(await safeErr(res))
  return (await res.json()) as T
}

export async function apiDelete<T>(path: string): Promise<T> {
  const res = await fetch(withBase(path), { method: 'DELETE', headers: { ...authHeader() } })
  if (!res.ok) throw new Error(await safeErr(res))
  return (await res.json()) as T
}

export async function downloadFile(path: string, filename?: string): Promise<void> {
  const res = await fetch(withBase(path), { method: 'GET', headers: { ...authHeader() } })
  if (!res.ok) throw new Error(await safeErr(res))
  const blob = await res.blob()
  const objectUrl = window.URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = objectUrl
  a.download = filename || 'download'
  a.rel = 'noreferrer'
  document.body.appendChild(a)
  a.click()
  a.remove()
  window.URL.revokeObjectURL(objectUrl)
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
