import type { LocalModel } from '../types'

export function formatLocalModelToolSupport(model?: LocalModel | null): string {
  if (!model) return 'Tool calls: unknown'
  if (model.supports_tools === false) return 'Tool calls: not supported'
  const support = String(model.tool_support || '').trim().toLowerCase()
  if (support === 'native') return 'Tool calls: supported (native)'
  if (support === 'generic') return 'Tool calls: supported (generic)'
  return 'Tool calls: supported'
}
