import { authHeader } from '../auth'

const DEFAULT_BACKEND = (() => {
  try {
    const proto = window.location.protocol || 'http:'
    const host = window.location.hostname || '127.0.0.1'
    const port = window.location.port || ''
    if (proto === 'tauri:' || proto === 'file:') {
      return 'http://127.0.0.1:8000'
    }
    if (port === '5173' || port === '4173') {
      return `${proto}//${host}:8000`
    }
    return window.location.origin || `${proto}//${host}${port ? `:${port}` : ''}`
  } catch {
    return 'http://127.0.0.1:8000'
  }
})()

function normalizeBackendUrl(raw: string): string {
  const trimmed = (raw || '').trim()
  if (!trimmed) return ''
  return trimmed.replace(/\/+$/, '')
}

export function getBackendUrl(): string {
  try {
    const globalUrl = (window as any).__IGX_BACKEND_URL__
    if (typeof globalUrl === 'string' && globalUrl.trim()) {
      return normalizeBackendUrl(globalUrl)
    }
    const fromStorage = window.localStorage.getItem('igx_backend_url') || ''
    if (fromStorage.trim()) {
      return normalizeBackendUrl(fromStorage)
    }
  } catch {
    // ignore
  }
  const envUrl = (import.meta as any).env?.VITE_BACKEND_URL
  return normalizeBackendUrl(envUrl) || DEFAULT_BACKEND
}

export function setBackendUrl(url: string) {
  const next = normalizeBackendUrl(url)
  if (!next) return
  try {
    window.localStorage.setItem('igx_backend_url', next)
  } catch {
    // ignore
  }
  ;(window as any).__IGX_BACKEND_URL__ = next
}

export const BACKEND_URL: string = getBackendUrl()

function withBase(path: string): string {
  const base = getBackendUrl()
  if (path.startsWith('http://') || path.startsWith('https://')) return path
  if (!path.startsWith('/')) return `${base}/${path}`
  return `${base}${path}`
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

async function safeErr(res: Response): Promise<string> {
  try {
    const j = await res.json()
    if (j && typeof j === 'object' && 'detail' in j) return String((j as any).detail)
  } catch {
    // ignore
  }
  return `HTTP ${res.status}`
}
