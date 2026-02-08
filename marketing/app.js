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
