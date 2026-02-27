export function formatProviderLabel(provider: string) {
  const p = (provider || '').toLowerCase()
  if (p === 'openrouter') return 'OpenRouter'
  if (p === 'local') return 'Local'
  if (p === 'chatgpt') return 'ChatGPT (OAuth)'
  if (p === 'openai') return 'OpenAI'
  return provider || 'OpenAI'
}

export function orderProviderList(list: string[]) {
  const order = ['chatgpt', 'openai', 'openrouter', 'local']
  const seen = new Set<string>()
  const unique: string[] = []
  for (const raw of list || []) {
    const val = String(raw || '').trim()
    if (!val || seen.has(val)) continue
    seen.add(val)
    unique.push(val)
  }
  return unique
    .map((provider, index) => {
      const rank = order.indexOf(provider.toLowerCase())
      return { provider, index, rank: rank === -1 ? order.length + 1 : rank }
    })
    .sort((a, b) => (a.rank - b.rank) || (a.index - b.index))
    .map((item) => item.provider)
}
