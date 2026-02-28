import type { HostAction } from '../types'

export function formatHostActionLabel(action: HostAction): { title: string; detail: string } {
  const payload = action.payload || {}
  switch (action.action_type) {
    case 'run_shell':
      return {
        title: 'Shell command',
        detail: String(payload.command || ''),
      }
    case 'run_applescript':
      return {
        title: 'AppleScript',
        detail: String(payload.script || ''),
      }
    case 'run_powershell':
      return {
        title: 'PowerShell',
        detail: String(payload.script || payload.command || ''),
      }
    default:
      return { title: action.action_type || 'Host action', detail: '' }
  }
}
