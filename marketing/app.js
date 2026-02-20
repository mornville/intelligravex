const cards = Array.from(document.querySelectorAll('.card.reveal'))
if ('IntersectionObserver' in window && cards.length > 0) {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('in')
          observer.unobserve(entry.target)
        }
      })
    },
    { threshold: 0.15 }
  )

  cards.forEach((card) => observer.observe(card))
} else {
  cards.forEach((card) => card.classList.add('in'))
}

const parallaxLayers = Array.from(document.querySelectorAll('[data-parallax]'))
if (parallaxLayers.length > 0) {
  let ticking = false
  const onScroll = () => {
    if (ticking) return
    ticking = true
    window.requestAnimationFrame(() => {
      const y = window.scrollY || window.pageYOffset || 0
      parallaxLayers.forEach((layer) => {
        const speed = Number(layer.dataset.parallax || '0')
        layer.style.transform = `translate3d(0, ${y * speed}px, 0)`
      })
      ticking = false
    })
  }
  window.addEventListener('scroll', onScroll, { passive: true })
  onScroll()
}

const modal = document.getElementById('download-modal')
const downloadButtons = Array.from(document.querySelectorAll('.download-btn'))

function openModal() {
  if (!modal) return
  modal.classList.add('active')
  modal.setAttribute('aria-hidden', 'false')
}

function closeModal() {
  if (!modal) return
  modal.classList.remove('active')
  modal.setAttribute('aria-hidden', 'true')
}

downloadButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    window.setTimeout(openModal, 200)
  })
})

document.addEventListener('click', (event) => {
  if (!modal || !modal.classList.contains('active')) return
  const target = event.target
  if (!(target instanceof HTMLElement)) return
  if (target.dataset.close === 'true') {
    closeModal()
  }
})

document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') {
    closeModal()
  }
})

const featureButtons = Array.from(document.querySelectorAll('[data-feature]'))
const featurePanel = document.querySelector('[data-feature-panel]')
const featureContainer = document.querySelector('.feature-showcase')
if (featureButtons.length > 0 && featurePanel) {
  const titleEl = featurePanel.querySelector('[data-feature-title]')
  const bodyEl = featurePanel.querySelector('[data-feature-body]')
  const tagsEl = featurePanel.querySelector('[data-feature-tags]')
  const bulletsEl = featurePanel.querySelector('[data-feature-bullets]')
  const iconEl = featurePanel.querySelector('[data-feature-icon]')
  let currentIndex = 0
  let paused = false

  const setActive = (index) => {
    const btn = featureButtons[index]
    if (!btn) return
    currentIndex = index
    featureButtons.forEach((button, i) => {
      button.classList.toggle('active', i === index)
      if (button.dataset.accent) {
        button.style.setProperty('--accent', button.dataset.accent)
      }
    })
    const accent = btn.dataset.accent || '#64f0ff'
    featurePanel.style.setProperty('--accent', accent)
    if (titleEl) titleEl.textContent = btn.dataset.title || ''
    if (bodyEl) bodyEl.textContent = btn.dataset.body || ''
    if (iconEl) {
      const iconSource = btn.querySelector('.logo-icon')
      iconEl.innerHTML = iconSource ? iconSource.innerHTML : ''
    }
    if (tagsEl) {
      tagsEl.innerHTML = ''
      const tags = (btn.dataset.tags || '').split(',').map((t) => t.trim()).filter(Boolean)
      tags.forEach((tag) => {
        const span = document.createElement('span')
        span.textContent = tag
        tagsEl.appendChild(span)
      })
    }
    if (bulletsEl) {
      bulletsEl.innerHTML = ''
      const bullets = (btn.dataset.bullets || '').split('|').map((b) => b.trim()).filter(Boolean)
      bullets.forEach((bullet) => {
        const li = document.createElement('li')
        li.textContent = bullet
        bulletsEl.appendChild(li)
      })
    }
  }

  featureButtons.forEach((btn, index) => {
    btn.addEventListener('click', () => setActive(index))
    btn.addEventListener('focus', () => setActive(index))
  })

  if (featureContainer) {
    featureContainer.addEventListener('mouseenter', () => {
      paused = true
    })
    featureContainer.addEventListener('mouseleave', () => {
      paused = false
    })
  }

  setActive(0)
  window.setInterval(() => {
    if (paused) return
    setActive((currentIndex + 1) % featureButtons.length)
  }, 5200)
}

const mediaTabs = Array.from(document.querySelectorAll('[data-media-tab]'))
const mediaPanel = document.querySelector('[data-media-panel]')
const mediaCarousel = document.querySelector('[data-carousel="media"]')
const mediaPrev = document.querySelector('[data-carousel-prev="media"]')
const mediaNext = document.querySelector('[data-carousel-next="media"]')
if (mediaTabs.length > 0 && mediaPanel) {
  const imgEl = mediaPanel.querySelector('[data-media-image]')
  const videoEl = mediaPanel.querySelector('[data-media-video]')
  const titleEl = mediaPanel.querySelector('[data-media-title]')
  const bodyEl = mediaPanel.querySelector('[data-media-body]')
  let mediaIndex = 0
  let mediaPaused = false
  let mediaTimer = null

  const setActive = (index, options = {}) => {
    const tab = mediaTabs[index]
    if (!tab) return
    const { scroll = true } = options
    mediaIndex = index
    mediaTabs.forEach((btn, i) => btn.classList.toggle('active', i === index))
    const title = tab.dataset.title || ''
    const body = tab.dataset.body || ''
    const type = tab.dataset.type || 'image'
    const src = tab.dataset.src || ''
    const poster = tab.dataset.poster || ''

    if (titleEl) titleEl.textContent = title
    if (bodyEl) bodyEl.textContent = body

    if (type === 'video') {
      mediaPanel.classList.add('video-active')
      if (videoEl instanceof HTMLVideoElement) {
        if (poster) videoEl.poster = poster
        if (src) {
          videoEl.pause()
          videoEl.src = src
          videoEl.load()
        }
      }
      if (imgEl instanceof HTMLImageElement && src) {
        imgEl.src = poster || src
      }
    } else {
      mediaPanel.classList.remove('video-active')
      if (imgEl instanceof HTMLImageElement && src) {
        imgEl.src = src
      }
      if (videoEl instanceof HTMLVideoElement) {
        videoEl.pause()
      }
    }
    if (scroll && tab instanceof HTMLElement) {
      tab.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' })
    }
  }

  mediaTabs.forEach((tab, index) => {
    tab.addEventListener('click', () => setActive(index))
    tab.addEventListener('focus', () => setActive(index))
  })

  const initial = mediaTabs.findIndex((tab) => tab.classList.contains('active'))
  setActive(initial >= 0 ? initial : 0, { scroll: false })

  const runMedia = () => {
    if (mediaPaused) return
    setActive((mediaIndex + 1) % mediaTabs.length)
  }

  const restartMediaTimer = () => {
    if (mediaTimer) window.clearInterval(mediaTimer)
    mediaTimer = window.setInterval(runMedia, 5000)
  }

  if (mediaPrev) {
    mediaPrev.addEventListener('click', () => {
      setActive((mediaIndex - 1 + mediaTabs.length) % mediaTabs.length)
      restartMediaTimer()
    })
  }

  if (mediaNext) {
    mediaNext.addEventListener('click', () => {
      setActive((mediaIndex + 1) % mediaTabs.length)
      restartMediaTimer()
    })
  }

  if (mediaCarousel) {
    mediaCarousel.addEventListener('mouseenter', () => {
      mediaPaused = true
    })
    mediaCarousel.addEventListener('mouseleave', () => {
      mediaPaused = false
    })
  }

  restartMediaTimer()
}

const devTabs = Array.from(document.querySelectorAll('[data-dev-tab]'))
const devPanel = document.querySelector('[data-dev-panel]')
const devCarousel = document.querySelector('[data-carousel="devs"]')
const devPrev = document.querySelector('[data-carousel-prev="devs"]')
const devNext = document.querySelector('[data-carousel-next="devs"]')
if (devTabs.length > 0 && devPanel) {
  const imgEl = devPanel.querySelector('[data-dev-image]')
  const titleEl = devPanel.querySelector('[data-dev-title]')
  const bodyEl = devPanel.querySelector('[data-dev-body]')
  let devIndex = 0
  let devPaused = false
  let devTimer = null

  const setActive = (index, options = {}) => {
    const tab = devTabs[index]
    if (!tab) return
    const { scroll = true } = options
    devIndex = index
    devTabs.forEach((btn, i) => btn.classList.toggle('active', i === index))
    const title = tab.dataset.title || ''
    const body = tab.dataset.body || ''
    const src = tab.dataset.src || ''

    if (titleEl) titleEl.textContent = title
    if (bodyEl) bodyEl.textContent = body
    if (imgEl instanceof HTMLImageElement && src) {
      imgEl.src = src
      imgEl.alt = title || 'Preview'
    }
    if (scroll && tab instanceof HTMLElement) {
      tab.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' })
    }
  }

  devTabs.forEach((tab, index) => {
    tab.addEventListener('click', () => setActive(index))
    tab.addEventListener('focus', () => setActive(index))
  })

  const initial = devTabs.findIndex((tab) => tab.classList.contains('active'))
  setActive(initial >= 0 ? initial : 0, { scroll: false })

  const runDev = () => {
    if (devPaused) return
    setActive((devIndex + 1) % devTabs.length)
  }

  const restartDevTimer = () => {
    if (devTimer) window.clearInterval(devTimer)
    devTimer = window.setInterval(runDev, 5000)
  }

  if (devPrev) {
    devPrev.addEventListener('click', () => {
      setActive((devIndex - 1 + devTabs.length) % devTabs.length)
      restartDevTimer()
    })
  }

  if (devNext) {
    devNext.addEventListener('click', () => {
      setActive((devIndex + 1) % devTabs.length)
      restartDevTimer()
    })
  }

  if (devCarousel) {
    devCarousel.addEventListener('mouseenter', () => {
      devPaused = true
    })
    devCarousel.addEventListener('mouseleave', () => {
      devPaused = false
    })
  }

  restartDevTimer()
}
