class RecorderWorkletProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._enabled = false;
    this._chunks = [];
    this._len = 0;
    this.port.onmessage = (ev) => {
      const cmd = ev?.data?.cmd;
      if (cmd === "start") {
        this._enabled = true;
      } else if (cmd === "stop") {
        this._enabled = false;
        this._flush();
      }
    };
  }

  _flush() {
    if (!this._len) return;
    const out = new Float32Array(this._len);
    let offset = 0;
    for (const c of this._chunks) {
      out.set(c, offset);
      offset += c.length;
    }
    this._chunks = [];
    this._len = 0;
    this.port.postMessage(out, [out.buffer]);
  }

  process(inputs) {
    if (!this._enabled) return true;
    const input = inputs && inputs[0] && inputs[0][0];
    if (!input) return true;

    const copy = input.slice(0);
    this._chunks.push(copy);
    this._len += copy.length;
    if (this._len >= 8192) this._flush();
    return true;
  }
}

registerProcessor("recorder-worklet", RecorderWorkletProcessor);

