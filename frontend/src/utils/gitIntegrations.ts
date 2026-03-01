export type GitProvider = 'github' | 'gitlab'
export type GitAuthMode = 'pat' | 'oauth' | 'ssh'

export type GitProviderConfig = {
  auth_mode: GitAuthMode
  token: string
  oauth_account: string
  repo_url: string
  repo_cache_path: string
  repo_source_path: string
}

export type GitIntegrationsState = {
  provider: GitProvider
  ssh_key_path: string
  providers: Record<GitProvider, GitProviderConfig>
}

const DEFAULT_PROVIDER_CFG: GitProviderConfig = {
  auth_mode: 'pat',
  token: '',
  oauth_account: '',
  repo_url: '',
  repo_cache_path: '',
  repo_source_path: '',
}

const DEFAULT_STATE: GitIntegrationsState = {
  provider: 'github',
  ssh_key_path: '',
  providers: {
    github: { ...DEFAULT_PROVIDER_CFG },
    gitlab: { ...DEFAULT_PROVIDER_CFG },
  },
}

function safeParse(raw: string | undefined): Record<string, any> {
  try {
    const obj = JSON.parse((raw || '').trim() || '{}')
    if (!obj || typeof obj !== 'object' || Array.isArray(obj)) return {}
    return obj as Record<string, any>
  } catch {
    return {}
  }
}

function providerToken(obj: Record<string, any>, provider: GitProvider): string {
  if (provider === 'github') {
    return String(obj.github_token || obj.GITHUB_TOKEN || obj.git_token || obj.GIT_TOKEN || '').trim()
  }
  return String(obj.gitlab_token || obj.GITLAB_TOKEN || obj.git_token || obj.GIT_TOKEN || '').trim()
}

function providerCfg(raw: any): GitProviderConfig {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return { ...DEFAULT_PROVIDER_CFG }
  const modeRaw = String(raw.auth_mode || '').trim().toLowerCase()
  const auth_mode: GitAuthMode = modeRaw === 'oauth' ? 'oauth' : modeRaw === 'ssh' ? 'ssh' : 'pat'
  return {
    auth_mode,
    token: String(raw.token || '').trim(),
    oauth_account: String(raw.oauth_account || '').trim(),
    repo_url: String(raw.repo_url || '').trim(),
    repo_cache_path: String(raw.repo_cache_path || '').trim(),
    repo_source_path: String(raw.repo_source_path || '').trim(),
  }
}

export function readGitIntegrationsState(authJson: string | undefined): GitIntegrationsState {
  const obj = safeParse(authJson)
  const state: GitIntegrationsState = JSON.parse(JSON.stringify(DEFAULT_STATE))

  const p = String(obj.git_provider || '').trim().toLowerCase()
  state.provider = p === 'gitlab' ? 'gitlab' : 'github'
  state.ssh_key_path = String(obj.ssh_private_key_path || obj.ssh_key_path || '').trim()

  const nested = obj.git_integrations
  if (nested && typeof nested === 'object' && !Array.isArray(nested)) {
    state.providers.github = providerCfg((nested as Record<string, any>).github)
    state.providers.gitlab = providerCfg((nested as Record<string, any>).gitlab)
  }

  for (const provider of ['github', 'gitlab'] as GitProvider[]) {
    if (!state.providers[provider].token) {
      state.providers[provider].token = providerToken(obj, provider)
    }
  }

  const selected = state.providers[state.provider]
  if (!selected.repo_url) {
    selected.repo_url = String(obj.preferred_repo_url || obj.git_preferred_repo_url || obj.git_repo_url || obj.preferred_repo || '').trim()
  }
  if (!selected.repo_cache_path) {
    selected.repo_cache_path = String(obj.preferred_repo_cache_path || obj.git_repo_cache_path || obj.preferred_repo_path || '').trim()
  }
  if (!selected.repo_source_path) {
    selected.repo_source_path = String(obj.preferred_repo_source_path || obj.git_repo_source_path || obj.preferred_repo_working_path || '').trim()
  }

  if (!selected.token) {
    const mode = String(obj.git_auth_method || '').trim().toLowerCase()
    if (mode === 'ssh') selected.auth_mode = 'ssh'
  }

  return state
}

export function buildGitIntegrationsAuthJson(rawAuthJson: string | undefined, state: GitIntegrationsState): string {
  const base = safeParse(rawAuthJson)

  const provider: GitProvider = state.provider === 'gitlab' ? 'gitlab' : 'github'
  const selected = state.providers[provider] || { ...DEFAULT_PROVIDER_CFG }

  base.git_provider = provider
  base.git_integrations = {
    github: {
      auth_mode: state.providers.github.auth_mode,
      token: state.providers.github.token.trim(),
      oauth_account: state.providers.github.oauth_account.trim(),
      repo_url: state.providers.github.repo_url.trim(),
      repo_cache_path: state.providers.github.repo_cache_path.trim(),
      repo_source_path: state.providers.github.repo_source_path.trim(),
    },
    gitlab: {
      auth_mode: state.providers.gitlab.auth_mode,
      token: state.providers.gitlab.token.trim(),
      oauth_account: state.providers.gitlab.oauth_account.trim(),
      repo_url: state.providers.gitlab.repo_url.trim(),
      repo_cache_path: state.providers.gitlab.repo_cache_path.trim(),
      repo_source_path: state.providers.gitlab.repo_source_path.trim(),
    },
  }

  const sshPath = state.ssh_key_path.trim()
  if (sshPath) {
    base.ssh_private_key_path = sshPath
    delete base.ssh_key_path
  } else {
    delete base.ssh_private_key_path
    delete base.ssh_key_path
  }

  const mode = selected.auth_mode
  const token = selected.token.trim()
  if (mode === 'ssh') {
    base.git_auth_method = 'ssh'
    delete base.github_token
    delete base.GITHUB_TOKEN
    delete base.gitlab_token
    delete base.GITLAB_TOKEN
    delete base.git_token
    delete base.GIT_TOKEN
  } else {
    base.git_auth_method = mode === 'oauth' ? 'oauth' : 'token'
    if (token) {
      base.GIT_TOKEN = token
      base.git_token = token
      if (provider === 'github') {
        base.github_token = token
        base.GITHUB_TOKEN = token
        delete base.gitlab_token
        delete base.GITLAB_TOKEN
      } else {
        base.gitlab_token = token
        base.GITLAB_TOKEN = token
        delete base.github_token
        delete base.GITHUB_TOKEN
      }
    } else {
      delete base.github_token
      delete base.GITHUB_TOKEN
      delete base.gitlab_token
      delete base.GITLAB_TOKEN
      delete base.git_token
      delete base.GIT_TOKEN
    }
  }

  const repoUrl = selected.repo_url.trim()
  const repoCache = selected.repo_cache_path.trim()
  const repoSource = selected.repo_source_path.trim()
  if (repoUrl) base.preferred_repo_url = repoUrl
  else delete base.preferred_repo_url
  if (repoCache) base.preferred_repo_cache_path = repoCache
  else delete base.preferred_repo_cache_path
  if (repoSource) base.preferred_repo_source_path = repoSource
  else delete base.preferred_repo_source_path

  delete base.git_preferred_repo_url
  delete base.git_repo_url
  delete base.git_repo_cache_path
  delete base.preferred_repo
  delete base.preferred_repo_path
  delete base.git_repo_source_path
  delete base.preferred_repo_working_path

  return JSON.stringify(base, null, 2)
}
