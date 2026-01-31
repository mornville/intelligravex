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
      '.igxvbHeaderRight{display:flex;align-items:center;gap:8px}' +
      '.igxvbMeta{font-size:12px;opacity:.65;max-width:220px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}' +
      '.igxvbHeaderBtn{padding:6px 8px;border-radius:10px;border:1px solid rgba(255,255,255,.12);background:rgba(255,255,255,.06);color:#fff;font-weight:700;font-size:12px;cursor:pointer}' +
      '.igxvbHeaderBtn:disabled{opacity:.5;cursor:not-allowed}' +
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
      '.igxvbError{padding:8px 10px;margin:8px 10px;border-radius:12px;background:rgba(255,77,109,.16);border:1px solid rgba(255,77,109,.28);font-size:12px;white-space:pre-wrap;word-break:break-word}' +
      '.igxvbOverlay{position:fixed;inset:0;background:rgba(0,0,0,.55);display:none;align-items:center;justify-content:center;z-index:2147483647}' +
      '.igxvbModal{width:min(980px,94vw);height:min(720px,86vh);border-radius:14px;background:rgba(10,14,24,.98);border:1px solid rgba(255,255,255,.12);box-shadow:0 18px 40px rgba(0,0,0,.65);overflow:hidden;display:flex;flex-direction:column}' +
      '.igxvbModalHead{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:10px 12px;border-bottom:1px solid rgba(255,255,255,.08)}' +
      '.igxvbModalTitle{font-weight:800;font-size:13px;opacity:.9}' +
      '.igxvbModalClose{padding:8px 10px;border-radius:10px;border:1px solid rgba(255,255,255,.12);background:rgba(255,255,255,.06);color:#fff;font-weight:800;cursor:pointer}' +
      '.igxvbIframe{flex:1;border:0;width:100%;background:#0b1020}'
    document.head.appendChild(style)
  }

  var defaultBackendOrigin = null
  try {
    if (document && document.currentScript && document.currentScript.src) {
      defaultBackendOrigin = getBaseOriginFromScript(document.currentScript)
    }
  } catch {}

  function createWidget(opts) {
    injectStyles()
    var botId = String(opts.botId || '')
    var apiKey = String(opts.apiKey || '')
    // Default to a new conversation per page load (no cross-tab persistence).
    // If callers explicitly provide a stable id, we still accept it.
    var userConversationId = String(opts.userConversationId || '') || uuid()
    var baseOrigin = String(opts.baseOrigin || defaultBackendOrigin || window.location.origin)
    var title = String(opts.title || 'GravexStudio Bot')
    var target = opts.target || null

    if (!botId) throw new Error('botId is required')
    if (!apiKey) throw new Error('apiKey is required')

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
    var right = document.createElement('div')
    right.className = 'igxvbHeaderRight'
    var metaEl = document.createElement('div')
    metaEl.className = 'igxvbMeta'
    metaEl.textContent = 'connecting…'
    header.appendChild(titleEl)
    right.appendChild(metaEl)
    var filesBtn = document.createElement('button')
    filesBtn.className = 'igxvbHeaderBtn'
    filesBtn.textContent = 'Files'
    filesBtn.disabled = true
    right.appendChild(filesBtn)
    header.appendChild(right)
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
    var overlay = document.createElement('div')
    overlay.className = 'igxvbOverlay'
    var modal = document.createElement('div')
    modal.className = 'igxvbModal'
    var modalHead = document.createElement('div')
    modalHead.className = 'igxvbModalHead'
    var modalTitle = document.createElement('div')
    modalTitle.className = 'igxvbModalTitle'
    modalTitle.textContent = 'Files'
    var modalClose = document.createElement('button')
    modalClose.className = 'igxvbModalClose'
    modalClose.textContent = 'Close'
    modalHead.appendChild(modalTitle)
    modalHead.appendChild(modalClose)
    var iframe = document.createElement('iframe')
    iframe.className = 'igxvbIframe'
    iframe.referrerPolicy = 'no-referrer'
    modal.appendChild(modalHead)
    modal.appendChild(iframe)
    overlay.appendChild(modal)
    document.body.appendChild(overlay)

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
      metaEl.title = s
    }

    function httpOrigin() {
      var b = String(baseOrigin || '').trim()
      if (!b) return window.location.origin
      if (b.indexOf('http://') !== 0 && b.indexOf('https://') !== 0) b = 'http://' + b
      return b.replace(/\/+$/, '')
    }

    function openFilesModal() {
      if (!conversationId) {
        showError('Conversation not ready yet')
        return
      }
      var url =
        httpOrigin() +
        '/conversations/' +
        encodeURIComponent(conversationId) +
        '/files?key=' +
        encodeURIComponent(apiKey)
      iframe.src = url
      overlay.style.display = 'flex'
    }

    function closeFilesModal() {
      overlay.style.display = 'none'
      try {
        iframe.src = 'about:blank'
      } catch {}
    }

    function connect() {
      var b = String(baseOrigin || '').trim()
      if (b.indexOf('http://') !== 0 && b.indexOf('https://') !== 0) b = 'http://' + b
      var proto = b.startsWith('https:') ? 'wss:' : 'ws:'
      var host = new URL(b).host
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
          filesBtn.disabled = !conversationId
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
        if (msg.type === 'interim') {
          var t = String(msg.text || '').trim()
          if (!t) return
          draft = null
          addMsg('assistant', t)
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
    filesBtn.onclick = function () {
      openFilesModal()
    }
    modalClose.onclick = function () {
      closeFilesModal()
    }
    overlay.addEventListener('click', function (e) {
      if (e && e.target === overlay) closeFilesModal()
    })
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
        try {
          overlay.remove()
        } catch {}
      },
    }
  }

  window.GravexStudioVoiceBot = {
    create: createWidget,
    autoInit: function () {
      var script = document.currentScript
      if (!script) return
      var botId = script.getAttribute('data-bot-id') || ''
      var apiKey = script.getAttribute('data-api-key') || ''
      var userConversationId = script.getAttribute('data-user-conversation-id') || ''
      var target = script.getAttribute('data-target') || ''
      var title = script.getAttribute('data-title') || 'GravexStudio Bot'
      var baseOrigin = script.getAttribute('data-backend-origin') || getBaseOriginFromScript(script)
      if (!botId || !apiKey || !userConversationId) return
      createWidget({ botId: botId, apiKey: apiKey, userConversationId: userConversationId, target: target || null, title: title, baseOrigin: baseOrigin })
    },
  }

  try {
    window.GravexStudioVoiceBot.autoInit()
  } catch {}
})()
