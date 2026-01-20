const AUTH_STORAGE_KEY = 'igx_basic_auth'

export function getBasicAuthToken(): string | null {
  try {
    return window.localStorage.getItem(AUTH_STORAGE_KEY)
  } catch {
    return null
  }
}

export function setBasicAuthToken(username: string, password: string): string {
  const token = btoa(`${username}:${password}`)
  window.localStorage.setItem(AUTH_STORAGE_KEY, token)
  return token
}

export function clearBasicAuthToken(): void {
  try {
    window.localStorage.removeItem(AUTH_STORAGE_KEY)
  } catch {
    // ignore
  }
}

export function authHeader(): Record<string, string> {
  const token = getBasicAuthToken()
  if (!token) return {}
  return { Authorization: `Basic ${token}` }
}
