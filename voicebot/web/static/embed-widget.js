(function () {
  function uuid() {
    if (crypto && crypto.randomUUID) return crypto.randomUUID()
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
      var r = (Math.random() * 16) | 0
      var v = c === 'x' ? r : (r & 0x3) | 0x8
      return v.toString(16)
    })
  }

  function getBaseOriginFromScript(scriptEl) {
    try {
      var u = new URL(scriptEl.src)
      return u.origin
    } catch {
      return window.location.origin
    }
  }

  function injectStyles() {
    if (document.getElementById('igx-voicebot-style')) return
    var style = document.createElement('style')
    style.id = 'igx-voicebot-style'
    style.textContent =
      '.igxvb{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;font-size:14px;color:#fff}' +
      '.igxvbCard{width:360px;max-width:92vw;border-radius:14px;background:linear-gradient(180deg,rgba(10,14,24,.98),rgba(6,8,14,.98));border:1px solid rgba(255,255,255,.12);box-shadow:0 18px 40px rgba(0,0,0,.55);overflow:hidden}' +
      '.igxvbHeader{padding:10px 12px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid rgba(255,255,255,.08)}' +
      '.igxvbTitle{font-weight:700;font-size:13px;opacity:.9}' +
      '.igxvbMeta{font-size:12px;opacity:.65}' +
      '.igxvbBody{height:320px;overflow:auto;padding:10px 10px 12px}' +
      '.igxvbRow{display:flex;margin:6px 0}' +
      '.igxvbBubble{max-width:85%;padding:10px 12px;border-radius:14px;white-space:pre-wrap;word-break:break-word;line-height:1.35}' +
      '.igxvbUser{justify-content:flex-end}' +
      '.igxvbUser .igxvbBubble{background:rgba(124,108,255,.18);border:1px solid rgba(124,108,255,.28)}' +
      '.igxvbAsst .igxvbBubble{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.10)}' +
      '.igxvbFooter{display:flex;gap:8px;padding:10px;border-top:1px solid rgba(255,255,255,.08)}' +
      '.igxvbInput{flex:1;min-width:0;padding:10px 12px;border-radius:12px;border:1px solid rgba(255,255,255,.12);background:rgba(255,255,255,.04);color:#fff;outline:none}' +
      '.igxvbBtn{padding:10px 12px;border-radius:12px;border:1px solid rgba(124,108,255,.35);background:rgba(124,108,255,.22);color:#fff;font-weight:700;cursor:pointer}' +
      '.igxvbBtn:disabled{opacity:.5;cursor:not-allowed}' +
      '.igxvbError{padding:8px 10px;margin:8px 10px;border-radius:12px;background:rgba(255,77,109,.16);border:1px solid rgba(255,77,109,.28);font-size:12px;white-space:pre-wrap;word-break:break-word}'
    document.head.appendChild(style)
  }

  function createWidget(opts) {
    injectStyles()
    var botId = String(opts.botId || '')
    var apiKey = String(opts.apiKey || '')
    var userConversationId = String(opts.userConversationId || '')
    var baseOrigin = String(opts.baseOrigin || window.location.origin)
    var title = String(opts.title || 'Intelligravex Bot')
    var target = opts.target || null

    if (!botId) throw new Error('botId is required')
    if (!apiKey) throw new Error('apiKey is required')
    if (!userConversationId) throw new Error('userConversationId is required')

    var root = document.createElement('div')
    root.className = 'igxvb'
    var card = document.createElement('div')
    card.className = 'igxvbCard'
    root.appendChild(card)

    var header = document.createElement('div')
    header.className = 'igxvbHeader'
    var titleEl = document.createElement('div')
    titleEl.className = 'igxvbTitle'
    titleEl.textContent = title
    var metaEl = document.createElement('div')
    metaEl.className = 'igxvbMeta'
    metaEl.textContent = 'connecting…'
    header.appendChild(titleEl)
    header.appendChild(metaEl)
    card.appendChild(header)

    var body = document.createElement('div')
    body.className = 'igxvbBody'
    card.appendChild(body)

    var footer = document.createElement('div')
    footer.className = 'igxvbFooter'
    var input = document.createElement('input')
    input.className = 'igxvbInput'
    input.placeholder = 'Type a message…'
    var sendBtn = document.createElement('button')
    sendBtn.className = 'igxvbBtn'
    sendBtn.textContent = 'Send'
    sendBtn.disabled = true
    footer.appendChild(input)
    footer.appendChild(sendBtn)
    card.appendChild(footer)

    if (target) {
      var el = typeof target === 'string' ? document.querySelector(target) : target
      if (el) el.appendChild(root)
      else document.body.appendChild(root)
    } else {
      document.body.appendChild(root)
    }

    var ws = null
    var conversationId = null
    var draft = null
    var stage = 'disconnected'

    function scrollToBottom() {
      body.scrollTop = body.scrollHeight
    }

    function addMsg(role, text) {
      var row = document.createElement('div')
      row.className = 'igxvbRow ' + (role === 'user' ? 'igxvbUser' : 'igxvbAsst')
      var bubble = document.createElement('div')
      bubble.className = 'igxvbBubble'
      bubble.textContent = text
      row.appendChild(bubble)
      body.appendChild(row)
      scrollToBottom()
      return bubble
    }

    function showError(text) {
      var e = document.createElement('div')
      e.className = 'igxvbError'
      e.textContent = text
      body.appendChild(e)
      scrollToBottom()
    }

    function setMeta(s) {
      metaEl.textContent = s
    }

    function connect() {
      var proto = baseOrigin.startsWith('https:') ? 'wss:' : 'ws:'
      var host = new URL(baseOrigin).host
      var url =
        proto +
        '//' +
        host +
        '/public/v1/ws/bots/' +
        encodeURIComponent(botId) +
        '/chat?key=' +
        encodeURIComponent(apiKey) +
        '&user_conversation_id=' +
        encodeURIComponent(userConversationId)
      ws = new WebSocket(url)
      stage = 'connecting'
      setMeta('connecting…')
      sendBtn.disabled = true
      ws.onopen = function () {
        stage = 'idle'
        setMeta('ready')
        sendBtn.disabled = false
        ws.send(JSON.stringify({ type: 'start', req_id: uuid() }))
      }
      ws.onclose = function () {
        stage = 'disconnected'
        setMeta('disconnected')
        sendBtn.disabled = true
      }
      ws.onerror = function () {
        setMeta('error')
      }
      ws.onmessage = function (ev) {
        var msg
        try {
          msg = JSON.parse(ev.data)
        } catch {
          return
        }
        if (msg.type === 'conversation') {
          conversationId = msg.conversation_id || msg.id || null
          return
        }
        if (msg.type === 'status') {
          if (msg.stage) setMeta(String(msg.stage))
          return
        }
        if (msg.type === 'error') {
          showError(String(msg.error || 'Unknown error'))
          return
        }
        if (msg.type === 'text_delta') {
          var d = String(msg.delta || '')
          if (!d) return
          if (!draft) draft = addMsg('assistant', d)
          else draft.textContent = (draft.textContent || '') + d
          return
        }
        if (msg.type === 'done') {
          draft = null
          var m = msg.metrics || null
          if (m && typeof m === 'object') {
            var line =
              'model ' +
              (m.model || '?') +
              ' | in ' +
              (m.input_tokens_est || 0) +
              ' out ' +
              (m.output_tokens_est || 0) +
              ' | $' +
              Number(m.cost_usd_est || 0).toFixed(6) +
              ' | ttfb ' +
              (m.llm_ttfb_ms || '-') +
              'ms total ' +
              (m.llm_total_ms || '-') +
              'ms'
            setMeta(line)
          } else {
            setMeta('ready')
          }
          return
        }
      }
    }

    function send(text) {
      var t = String(text || '').trim()
      if (!t) return
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        showError('Not connected')
        return
      }
      addMsg('user', t)
      draft = null
      ws.send(JSON.stringify({ type: 'chat', req_id: uuid(), conversation_id: conversationId, text: t }))
    }

    sendBtn.onclick = function () {
      var t = input.value
      input.value = ''
      send(t)
    }
    input.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') {
        e.preventDefault()
        var t = input.value
        input.value = ''
        send(t)
      }
    })

    connect()

    return {
      root: root,
      send: send,
      close: function () {
        try {
          if (ws) ws.close()
        } catch {}
      },
    }
  }

  window.IntelligravexVoiceBot = {
    create: createWidget,
    autoInit: function () {
      var script = document.currentScript
      if (!script) return
      var botId = script.getAttribute('data-bot-id') || ''
      var apiKey = script.getAttribute('data-api-key') || ''
      var userConversationId = script.getAttribute('data-user-conversation-id') || ''
      var target = script.getAttribute('data-target') || ''
      var title = script.getAttribute('data-title') || 'Intelligravex Bot'
      var baseOrigin = script.getAttribute('data-backend-origin') || getBaseOriginFromScript(script)
      if (!botId || !apiKey || !userConversationId) return
      createWidget({ botId: botId, apiKey: apiKey, userConversationId: userConversationId, target: target || null, title: title, baseOrigin: baseOrigin })
    },
  }

  try {
    window.IntelligravexVoiceBot.autoInit()
  } catch {}
})()

