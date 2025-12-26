export type Recorder = {
  start: () => Promise<void>
  stop: () => Promise<void>
  close: () => Promise<void>
}

const TARGET_SR = 16000

function clamp01(v: number): number {
  if (v > 1) return 1
  if (v < -1) return -1
  return v
}

function floatToPcm16(samples: Float32Array): ArrayBuffer {
  const out = new Int16Array(samples.length)
  for (let i = 0; i < samples.length; i++) {
    const s = clamp01(samples[i])
    out[i] = s < 0 ? Math.round(s * 0x8000) : Math.round(s * 0x7fff)
  }
  return out.buffer
}

function resampleLinear(input: Float32Array, inSr: number, outSr: number): Float32Array {
  if (inSr === outSr) return input
  const ratio = inSr / outSr
  const outLen = Math.max(1, Math.floor(input.length / ratio))
  const out = new Float32Array(outLen)
  for (let i = 0; i < outLen; i++) {
    const t = i * ratio
    const i0 = Math.floor(t)
    const i1 = Math.min(input.length - 1, i0 + 1)
    const frac = t - i0
    out[i] = input[i0] * (1 - frac) + input[i1] * frac
  }
  return out
}

export async function createRecorder(onPcm16: (pcm16: ArrayBuffer) => void): Promise<Recorder> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    } as any,
    video: false,
  })

  const ctx = new AudioContext()
  await ctx.audioWorklet.addModule('/recorder-worklet.js')
  const source = ctx.createMediaStreamSource(stream)
  const node = new AudioWorkletNode(ctx, 'recorder-worklet')
  const sink = ctx.createGain()
  sink.gain.value = 0
  source.connect(node)
  node.connect(sink)
  sink.connect(ctx.destination)

  const inSr = ctx.sampleRate

  node.port.onmessage = (ev) => {
    const f32 = ev.data as Float32Array
    const resampled = resampleLinear(f32, inSr, TARGET_SR)
    onPcm16(floatToPcm16(resampled))
  }

  async function start() {
    if (ctx.state !== 'running') await ctx.resume()
    node.port.postMessage({ cmd: 'start' })
  }

  async function stop() {
    node.port.postMessage({ cmd: 'stop' })
  }

  async function close() {
    try {
      stream.getTracks().forEach((t) => t.stop())
    } catch {
      // ignore
    }
    try {
      await ctx.close()
    } catch {
      // ignore
    }
  }

  return { start, stop, close }
}

