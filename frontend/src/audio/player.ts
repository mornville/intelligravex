export class WavQueuePlayer {
  private ctx: AudioContext
  private playAt: number

  constructor() {
    this.ctx = new AudioContext()
    this.playAt = this.ctx.currentTime
  }

  async playWavBytes(wavBytes: Uint8Array): Promise<void> {
    if (this.ctx.state !== 'running') await this.ctx.resume()
    const copy = new Uint8Array(wavBytes.byteLength)
    copy.set(wavBytes)
    const audio = await this.ctx.decodeAudioData(copy.buffer)
    const src = this.ctx.createBufferSource()
    src.buffer = audio
    src.connect(this.ctx.destination)
    const startAt = Math.max(this.playAt, this.ctx.currentTime + 0.01)
    src.start(startAt)
    this.playAt = startAt + audio.duration
  }

  async close(): Promise<void> {
    try {
      await this.ctx.close()
    } catch {
      // ignore
    }
  }
}
