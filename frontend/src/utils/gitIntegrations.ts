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

export type JiraConfig = {
  domain: string
  email: string
  api_token: string
  default_project_key: string
  default_issue_type: string
  default_jql: string
}

export type DatabaseCredential = {
  id: string
  nickname: string
  engine: string
  host: string
  port: string
  database: string
  user: string
  password: string
  options: string
  server_ca: string
  client_cert: string
  client_key: string
}

export type GmailConfig = {
  connected: boolean
  account_email: string
  scope: string
  token_type: string
  access_token: string
  refresh_token: string
  expires_at: number | null
  connected_at: string
  error: string
}

export type SlackConfig = {
  connected: boolean
  workspace_name: string
  workspace_id: string
  bot_user_id: string
  oauth_client_id: string
  oauth_client_secret: string
  oauth_redirect_uri: string
  oauth_scope: string
  scope: string
  access_token: string
  refresh_token: string
  expires_at: number | null
  connected_at: string
  error: string
}

export type GitIntegrationsState = {
  provider: GitProvider
  ssh_key_path: string
  providers: Record<GitProvider, GitProviderConfig>
  jira: JiraConfig
  gmail: GmailConfig
  slack: SlackConfig
  db_credentials: DatabaseCredential[]
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
  jira: {
    domain: '',
    email: '',
    api_token: '',
    default_project_key: '',
    default_issue_type: '',
    default_jql: '',
  },
  gmail: {
    connected: false,
    account_email: '',
    scope: '',
    token_type: '',
    access_token: '',
    refresh_token: '',
    expires_at: null,
    connected_at: '',
    error: '',
  },
  slack: {
    connected: false,
    workspace_name: '',
    workspace_id: '',
    bot_user_id: '',
    oauth_client_id: '',
    oauth_client_secret: '',
    oauth_redirect_uri: 'http://localhost:1467/auth/callback',
    oauth_scope: 'chat:write,channels:read,groups:read,im:read,mpim:read',
    scope: '',
    access_token: '',
    refresh_token: '',
    expires_at: null,
    connected_at: '',
    error: '',
  },
  db_credentials: [],
}

const DEFAULT_DB_CREDENTIAL: DatabaseCredential = {
  id: '',
  nickname: '',
  engine: 'postgresql',
  host: '',
  port: '',
  database: '',
  user: '',
  password: '',
  options: '',
  server_ca: '',
  client_cert: '',
  client_key: '',
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

function jiraCfg(raw: any): JiraConfig {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return {
      domain: '',
      email: '',
      api_token: '',
      default_project_key: '',
      default_issue_type: '',
      default_jql: '',
    }
  }
  return {
    domain: String(raw.domain || raw.site || raw.base_url || '').trim(),
    email: String(raw.email || '').trim(),
    api_token: String(raw.api_token || raw.token || '').trim(),
    default_project_key: String(raw.default_project_key || raw.project_key || '').trim(),
    default_issue_type: String(raw.default_issue_type || raw.issue_type || '').trim(),
    default_jql: String(raw.default_jql || raw.jql || '').trim(),
  }
}

function gmailCfg(raw: any): GmailConfig {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return {
      connected: false,
      account_email: '',
      scope: '',
      token_type: '',
      access_token: '',
      refresh_token: '',
      expires_at: null,
      connected_at: '',
      error: '',
    }
  }
  const scope = String(raw.scope || raw.scopes || '').trim()
  const refreshToken = String(raw.refresh_token || '').trim()
  const accessToken = String(raw.access_token || '').trim()
  const connectedRaw = raw.connected
  const connected = typeof connectedRaw === 'boolean' ? connectedRaw : Boolean(refreshToken || accessToken)
  const expiresRaw = raw.expires_at
  const expires_at = typeof expiresRaw === 'number' && Number.isFinite(expiresRaw) ? Number(expiresRaw) : null
  return {
    connected,
    account_email: String(raw.account_email || raw.email || '').trim(),
    scope,
    token_type: String(raw.token_type || '').trim(),
    access_token: accessToken,
    refresh_token: refreshToken,
    expires_at,
    connected_at: String(raw.connected_at || '').trim(),
    error: String(raw.error || '').trim(),
  }
}

function slackCfg(raw: any): SlackConfig {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return {
      connected: false,
      workspace_name: '',
      workspace_id: '',
      bot_user_id: '',
      oauth_client_id: '',
      oauth_client_secret: '',
      oauth_redirect_uri: 'http://localhost:1467/auth/callback',
      oauth_scope: 'chat:write,channels:read,groups:read,im:read,mpim:read',
      scope: '',
      access_token: '',
      refresh_token: '',
      expires_at: null,
      connected_at: '',
      error: '',
    }
  }
  const accessToken = String(raw.access_token || raw.bot_token || '').trim()
  const refreshToken = String(raw.refresh_token || '').trim()
  const connectedRaw = raw.connected
  const connected = typeof connectedRaw === 'boolean' ? connectedRaw : Boolean(accessToken || refreshToken)
  const expiresRaw = raw.expires_at
  const expires_at = typeof expiresRaw === 'number' && Number.isFinite(expiresRaw) ? Number(expiresRaw) : null
  return {
    connected,
    workspace_name: String(raw.workspace_name || raw.team_name || raw.workspace || '').trim(),
    workspace_id: String(raw.workspace_id || raw.team_id || '').trim(),
    bot_user_id: String(raw.bot_user_id || raw.user_id || '').trim(),
    oauth_client_id: String(raw.oauth_client_id || raw.client_id || '').trim(),
    oauth_client_secret: String(raw.oauth_client_secret || raw.client_secret || '').trim(),
    oauth_redirect_uri: String(raw.oauth_redirect_uri || raw.redirect_uri || '').trim() || 'http://localhost:1467/auth/callback',
    oauth_scope: String(raw.oauth_scope || '').trim() || 'chat:write,channels:read,groups:read,im:read,mpim:read',
    scope: String(raw.scope || raw.scopes || '').trim(),
    access_token: accessToken,
    refresh_token: refreshToken,
    expires_at,
    connected_at: String(raw.connected_at || '').trim(),
    error: String(raw.error || '').trim(),
  }
}

function mergeGmailFallbacks(gmail: GmailConfig, obj: Record<string, any>): GmailConfig {
  const next = { ...gmail }
  if (!next.account_email) next.account_email = String(obj.gmail_account_email || obj.gmail_email || '').trim()
  if (!next.scope) next.scope = String(obj.gmail_scope || '').trim()
  if (!next.token_type) next.token_type = String(obj.gmail_token_type || '').trim()
  if (!next.access_token) next.access_token = String(obj.gmail_access_token || '').trim()
  if (!next.refresh_token) next.refresh_token = String(obj.gmail_refresh_token || '').trim()
  if (!next.connected_at) next.connected_at = String(obj.gmail_connected_at || '').trim()
  if (!next.error) next.error = String(obj.gmail_error || '').trim()
  if (next.expires_at == null) {
    const expRaw = obj.gmail_expires_at
    if (typeof expRaw === 'number' && Number.isFinite(expRaw)) next.expires_at = Number(expRaw)
  }
  if (!next.connected) next.connected = Boolean(next.refresh_token || next.access_token || obj.gmail_connected)
  return next
}

function mergeSlackFallbacks(slack: SlackConfig, obj: Record<string, any>): SlackConfig {
  const next = { ...slack }
  if (!next.workspace_name) next.workspace_name = String(obj.slack_workspace || obj.slack_team_name || '').trim()
  if (!next.workspace_id) next.workspace_id = String(obj.slack_workspace_id || obj.slack_team_id || '').trim()
  if (!next.bot_user_id) next.bot_user_id = String(obj.slack_bot_user_id || '').trim()
  if (!next.oauth_client_id) next.oauth_client_id = String(obj.slack_oauth_client_id || obj.slack_client_id || '').trim()
  if (!next.oauth_client_secret) next.oauth_client_secret = String(obj.slack_oauth_client_secret || obj.slack_client_secret || '').trim()
  if (!next.oauth_redirect_uri) next.oauth_redirect_uri = String(obj.slack_oauth_redirect_uri || obj.slack_redirect_uri || '').trim()
  if (!next.oauth_scope) next.oauth_scope = String(obj.slack_oauth_scope || '').trim()
  if (!next.oauth_redirect_uri) next.oauth_redirect_uri = 'http://localhost:1467/auth/callback'
  if (!next.oauth_scope) next.oauth_scope = 'chat:write,channels:read,groups:read,im:read,mpim:read'
  if (!next.scope) next.scope = String(obj.slack_scope || '').trim()
  if (!next.access_token) next.access_token = String(obj.slack_access_token || '').trim()
  if (!next.refresh_token) next.refresh_token = String(obj.slack_refresh_token || '').trim()
  if (next.expires_at == null) {
    const expRaw = obj.slack_expires_at
    if (typeof expRaw === 'number' && Number.isFinite(expRaw)) next.expires_at = Number(expRaw)
  }
  if (!next.connected_at) next.connected_at = String(obj.slack_connected_at || '').trim()
  if (!next.error) next.error = String(obj.slack_error || '').trim()
  if (!next.connected) next.connected = Boolean(next.access_token || next.refresh_token || obj.slack_connected)
  return next
}

function disconnectedGmailState(): GmailConfig {
  return {
    connected: false,
    account_email: '',
    scope: '',
    token_type: '',
    access_token: '',
    refresh_token: '',
    expires_at: null,
    connected_at: '',
    error: '',
  }
}

function disconnectedSlackState(): SlackConfig {
  return {
    connected: false,
    workspace_name: '',
    workspace_id: '',
    bot_user_id: '',
    oauth_client_id: '',
    oauth_client_secret: '',
    oauth_redirect_uri: 'http://localhost:1467/auth/callback',
    oauth_scope: 'chat:write,channels:read,groups:read,im:read,mpim:read',
    scope: '',
    access_token: '',
    refresh_token: '',
    expires_at: null,
    connected_at: '',
    error: '',
  }
}

export function gmailConnected(gmail: GmailConfig): boolean {
  return Boolean(gmail.connected || gmail.refresh_token.trim() || gmail.access_token.trim())
}

export function clearGmailConfig(gmail: GmailConfig): GmailConfig {
  return {
    ...disconnectedGmailState(),
    scope: gmail.scope,
  }
}

export function slackConnected(slack: SlackConfig): boolean {
  return Boolean(slack.connected || slack.access_token.trim() || slack.refresh_token.trim())
}

export function clearSlackConfig(slack: SlackConfig): SlackConfig {
  return {
    ...disconnectedSlackState(),
    oauth_client_id: slack.oauth_client_id,
    oauth_client_secret: slack.oauth_client_secret,
    oauth_redirect_uri: slack.oauth_redirect_uri,
    oauth_scope: slack.oauth_scope,
    scope: slack.scope,
  }
}

function dbCredential(raw: any, index: number): DatabaseCredential {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return {
      ...DEFAULT_DB_CREDENTIAL,
      id: `db_${index + 1}`,
    }
  }
  return {
    id: String(raw.id || `db_${index + 1}`).trim(),
    nickname: String(raw.nickname || raw.name || '').trim(),
    engine: String(raw.engine || raw.driver || 'postgresql').trim() || 'postgresql',
    host: String(raw.host || '').trim(),
    port: String(raw.port || '').trim(),
    database: String(raw.database || raw.db || '').trim(),
    user: String(raw.user || raw.username || '').trim(),
    password: String(raw.password || raw.pass || '').trim(),
    options: String(raw.options || raw.params || '').trim(),
    server_ca: String(raw.server_ca || raw.serverCa || '').trim(),
    client_cert: String(raw.client_cert || raw.clientCert || '').trim(),
    client_key: String(raw.client_key || raw.clientKey || '').trim(),
  }
}

function dbCredentials(raw: any): DatabaseCredential[] {
  if (!Array.isArray(raw)) return []
  const used = new Set<string>()
  return raw.map((item, index) => {
    const next = dbCredential(item, index)
    const baseId = next.id || `db_${index + 1}`
    let id = baseId
    let suffix = 2
    while (used.has(id)) {
      id = `${baseId}_${suffix}`
      suffix += 1
    }
    used.add(id)
    return { ...next, id }
  })
}

export function newDatabaseCredential(): DatabaseCredential {
  const rand = Math.random().toString(36).slice(2, 8)
  return {
    ...DEFAULT_DB_CREDENTIAL,
    id: `db_${Date.now()}_${rand}`,
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

  const connectedApps = obj.connected_apps
  if (connectedApps && typeof connectedApps === 'object' && !Array.isArray(connectedApps)) {
    state.jira = jiraCfg((connectedApps as Record<string, any>).jira)
    state.gmail = gmailCfg((connectedApps as Record<string, any>).gmail)
    state.slack = slackCfg((connectedApps as Record<string, any>).slack)
  } else {
    state.jira = jiraCfg(obj.jira_integration)
    state.gmail = gmailCfg(obj.gmail_integration)
    state.slack = slackCfg(obj.slack_integration)
  }
  state.gmail = mergeGmailFallbacks(state.gmail, obj)
  state.slack = mergeSlackFallbacks(state.slack, obj)
  const connectedDbCreds =
    connectedApps && typeof connectedApps === 'object' && !Array.isArray(connectedApps)
      ? dbCredentials(
          (connectedApps as Record<string, any>).database_credentials || (connectedApps as Record<string, any>).db_credentials
        )
      : []
  const topLevelDbCreds = dbCredentials(obj.db_credentials || obj.database_credentials)
  state.db_credentials = connectedDbCreds.length ? connectedDbCreds : topLevelDbCreds

  if (!state.jira.domain) state.jira.domain = String(obj.jira_domain || obj.atlassian_domain || '').trim()
  if (!state.jira.email) state.jira.email = String(obj.jira_email || obj.atlassian_email || '').trim()
  if (!state.jira.api_token) state.jira.api_token = String(obj.jira_api_token || obj.atlassian_api_token || '').trim()
  if (!state.jira.default_project_key) state.jira.default_project_key = String(obj.jira_project_key || '').trim()
  if (!state.jira.default_issue_type) state.jira.default_issue_type = String(obj.jira_issue_type || '').trim()
  if (!state.jira.default_jql) state.jira.default_jql = String(obj.jira_default_jql || obj.jira_jql || '').trim()

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

  const connectedApps =
    base.connected_apps && typeof base.connected_apps === 'object' && !Array.isArray(base.connected_apps)
      ? { ...base.connected_apps }
      : {}
  connectedApps.jira = {
    domain: state.jira.domain.trim(),
    email: state.jira.email.trim(),
    api_token: state.jira.api_token.trim(),
    default_project_key: state.jira.default_project_key.trim(),
    default_issue_type: state.jira.default_issue_type.trim(),
    default_jql: state.jira.default_jql.trim(),
  }
  connectedApps.gmail = gmailConnected(state.gmail)
    ? {
        connected: true,
        account_email: state.gmail.account_email.trim(),
        scope: state.gmail.scope.trim(),
        token_type: state.gmail.token_type.trim(),
        access_token: state.gmail.access_token.trim(),
        refresh_token: state.gmail.refresh_token.trim(),
        expires_at: state.gmail.expires_at,
        connected_at: state.gmail.connected_at.trim(),
        error: state.gmail.error.trim(),
      }
    : disconnectedGmailState()
  connectedApps.slack = slackConnected(state.slack)
    ? {
        connected: true,
        workspace_name: state.slack.workspace_name.trim(),
        workspace_id: state.slack.workspace_id.trim(),
        bot_user_id: state.slack.bot_user_id.trim(),
        oauth_client_id: state.slack.oauth_client_id.trim(),
        oauth_client_secret: state.slack.oauth_client_secret.trim(),
        oauth_redirect_uri: state.slack.oauth_redirect_uri.trim(),
        oauth_scope: state.slack.oauth_scope.trim(),
        scope: state.slack.scope.trim(),
        access_token: state.slack.access_token.trim(),
        refresh_token: state.slack.refresh_token.trim(),
        expires_at: state.slack.expires_at,
        connected_at: state.slack.connected_at.trim(),
        error: state.slack.error.trim(),
      }
    : disconnectedSlackState()
  const dbCreds = (state.db_credentials || []).map((item, index) => {
    const next = dbCredential(item, index)
    return {
      id: next.id,
      nickname: next.nickname,
      engine: next.engine,
      host: next.host,
      port: next.port,
      database: next.database,
      user: next.user,
      password: next.password,
      options: next.options,
      server_ca: next.server_ca,
      client_cert: next.client_cert,
      client_key: next.client_key,
    }
  })
  connectedApps.database_credentials = dbCreds
  base.connected_apps = connectedApps
  base.jira_integration = connectedApps.jira
  base.gmail_integration = connectedApps.gmail
  base.slack_integration = connectedApps.slack
  base.db_credentials = dbCreds

  const jiraDomain = state.jira.domain.trim()
  const jiraEmail = state.jira.email.trim()
  const jiraApiToken = state.jira.api_token.trim()
  const jiraProjectKey = state.jira.default_project_key.trim()
  const jiraIssueType = state.jira.default_issue_type.trim()
  const jiraJql = state.jira.default_jql.trim()
  if (jiraDomain) base.jira_domain = jiraDomain
  else delete base.jira_domain
  if (jiraEmail) base.jira_email = jiraEmail
  else delete base.jira_email
  if (jiraApiToken) base.jira_api_token = jiraApiToken
  else delete base.jira_api_token
  if (jiraProjectKey) base.jira_project_key = jiraProjectKey
  else delete base.jira_project_key
  if (jiraIssueType) base.jira_issue_type = jiraIssueType
  else delete base.jira_issue_type
  if (jiraJql) base.jira_default_jql = jiraJql
  else delete base.jira_default_jql

  const gmailAccountEmail = state.gmail.account_email.trim()
  const gmailScope = state.gmail.scope.trim()
  const gmailTokenType = state.gmail.token_type.trim()
  const gmailAccessToken = state.gmail.access_token.trim()
  const gmailRefreshToken = state.gmail.refresh_token.trim()
  const gmailConnectedAt = state.gmail.connected_at.trim()
  const gmailError = state.gmail.error.trim()
  const gmailExpiresAt = state.gmail.expires_at
  const isGmailConnected = gmailConnected(state.gmail)
  if (gmailAccountEmail) base.gmail_account_email = gmailAccountEmail
  else delete base.gmail_account_email
  if (gmailScope) base.gmail_scope = gmailScope
  else delete base.gmail_scope
  if (gmailTokenType) base.gmail_token_type = gmailTokenType
  else delete base.gmail_token_type
  if (gmailAccessToken) base.gmail_access_token = gmailAccessToken
  else delete base.gmail_access_token
  if (gmailRefreshToken) base.gmail_refresh_token = gmailRefreshToken
  else delete base.gmail_refresh_token
  if (gmailConnectedAt) base.gmail_connected_at = gmailConnectedAt
  else delete base.gmail_connected_at
  if (gmailError) base.gmail_error = gmailError
  else delete base.gmail_error
  if (typeof gmailExpiresAt === 'number' && Number.isFinite(gmailExpiresAt)) base.gmail_expires_at = gmailExpiresAt
  else delete base.gmail_expires_at
  if (isGmailConnected) base.gmail_connected = true
  else delete base.gmail_connected
  delete base.gmail_client_id
  delete base.gmail_client_secret
  delete base.gmail_sender_email
  delete base.gmail_reply_to_email

  const slackWorkspace = state.slack.workspace_name.trim()
  const slackWorkspaceId = state.slack.workspace_id.trim()
  const slackBotUserId = state.slack.bot_user_id.trim()
  const slackOauthClientId = state.slack.oauth_client_id.trim()
  const slackOauthClientSecret = state.slack.oauth_client_secret.trim()
  const slackOauthRedirectUri = state.slack.oauth_redirect_uri.trim()
  const slackOauthScope = state.slack.oauth_scope.trim()
  const slackScope = state.slack.scope.trim()
  const slackAccessToken = state.slack.access_token.trim()
  const slackRefreshToken = state.slack.refresh_token.trim()
  const slackExpiresAt = state.slack.expires_at
  const slackConnectedAt = state.slack.connected_at.trim()
  const slackError = state.slack.error.trim()
  const isSlackConnected = slackConnected(state.slack)
  if (slackWorkspace) base.slack_workspace = slackWorkspace
  else delete base.slack_workspace
  if (slackWorkspaceId) base.slack_workspace_id = slackWorkspaceId
  else delete base.slack_workspace_id
  if (slackBotUserId) base.slack_bot_user_id = slackBotUserId
  else delete base.slack_bot_user_id
  if (slackOauthClientId) base.slack_oauth_client_id = slackOauthClientId
  else delete base.slack_oauth_client_id
  if (slackOauthClientSecret) base.slack_oauth_client_secret = slackOauthClientSecret
  else delete base.slack_oauth_client_secret
  if (slackOauthRedirectUri) base.slack_oauth_redirect_uri = slackOauthRedirectUri
  else delete base.slack_oauth_redirect_uri
  if (slackOauthScope) base.slack_oauth_scope = slackOauthScope
  else delete base.slack_oauth_scope
  if (slackScope) base.slack_scope = slackScope
  else delete base.slack_scope
  if (slackAccessToken) base.slack_access_token = slackAccessToken
  else delete base.slack_access_token
  if (slackRefreshToken) base.slack_refresh_token = slackRefreshToken
  else delete base.slack_refresh_token
  if (typeof slackExpiresAt === 'number' && Number.isFinite(slackExpiresAt)) base.slack_expires_at = slackExpiresAt
  else delete base.slack_expires_at
  if (slackConnectedAt) base.slack_connected_at = slackConnectedAt
  else delete base.slack_connected_at
  if (slackError) base.slack_error = slackError
  else delete base.slack_error
  if (isSlackConnected) base.slack_connected = true
  else delete base.slack_connected
  delete base.slack_client_id
  delete base.slack_client_secret

  delete base.git_preferred_repo_url
  delete base.git_repo_url
  delete base.git_repo_cache_path
  delete base.preferred_repo
  delete base.preferred_repo_path
  delete base.git_repo_source_path
  delete base.preferred_repo_working_path

  return JSON.stringify(base, null, 2)
}
