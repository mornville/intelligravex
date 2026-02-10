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
